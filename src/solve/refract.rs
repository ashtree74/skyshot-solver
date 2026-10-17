//! Atmospheric refraction measured from a solved wide field with the horizon in it.
//!
//! Refraction lifts every star towards the zenith by R(h), a steep function of the altitude h
//! (Saemundsson: 29' at the horizon, 5.4' at 10 deg, 1' at 45 deg). A free SIP polynomial
//! absorbs most of it, but a lens is symmetric about its optical axis and refraction about the
//! zenith: a rigid lens model (linear plus lens-like radial terms) fitted to the catalogue
//! positions refracted towards a trial zenith Z with a trial scale k (1 = 1010 hPa, 10 C) fits
//! best at the true Z and k. The search: a coarse grid over the zeniths that keep every matched
//! star above the horizon and some below 20 deg, then a simplex over (Z, ln k). The zenith in
//! ICRS is the observer's latitude (its declination) and local sidereal time (its right
//! ascension, up to precession and nutation of the equinox of date); k is the air density
//! relative to the standard. The WCS is not changed (it maps pixels to catalogue positions as
//! seen, refraction included, which is what annotations need); this is a measurement on top.

use super::*;
use crate::{
    sphere::dot,
    types::{Proj, Refraction},
};

/// Saemundsson's refraction (rad) at true altitude `h` (deg), clamped below -1 deg.
fn saemundsson(h_deg: f64) -> f64 {
    let h = h_deg.max(-1.0);
    (1.02 / (h + 10.3 / (h + 5.11)).to_radians().tan() / 60.0).to_radians()
}

/// `t` lifted towards the zenith `z` by `scale` x Saemundsson.
fn refract(t: Vec3, z: Vec3, scale: f64) -> Vec3 {
    let c = dot(t, z).clamp(-1.0, 1.0);
    let h = 90.0 - c.acos().to_degrees();
    let r = scale * saemundsson(h);
    let d = [z[0] - t[0] * c, z[1] - t[1] * c, z[2] - t[2] * c];
    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if n < 1e-12 {
        return t;
    }
    let (s, co) = r.sin_cos();
    crate::sphere::normalize([
        t[0] * co + d[0] / n * s,
        t[1] * co + d[1] / n * s,
        t[2] * co + d[2] / n * s,
    ])
}

/// Unit vector `d` (rad) east and `e` north of `c` on the sphere (tangent-plane step).
fn offset(c: Vec3, east: f64, north: f64) -> Vec3 {
    let (e, n) = tangent_basis(c);
    crate::sphere::normalize([
        c[0] + east * e[0] + north * n[0],
        c[1] + east * e[1] + north * n[1],
        c[2] + east * e[2] + north * n[2],
    ])
}

struct Problem<'a> {
    pairs: &'a [[f64; 4]],
    t: Vec<Vec3>,
    weights: Vec<f64>,
    crpix: [f64; 2],
    tangent: [f64; 2],
    order: u32,
    radial: u32,
    proj: Proj,
}

impl Problem<'_> {
    /// Weighted rms (px) of the lens model fitted to the pairs refracted towards `z` by `k`
    /// (`k` = 0: no refraction).
    fn rms(&self, z: Vec3, k: f64) -> f64 {
        let refr: Vec<[f64; 4]> = self
            .pairs
            .iter()
            .zip(&self.t)
            .map(|(p, &t)| {
                let a = if k > 0.0 { refract(t, z, k) } else { t };
                let (ra, dec) = radec(a);
                [p[0], p[1], ra, dec]
            })
            .collect();
        let Some(fit) = crate::wcsfit::fit_model(
            &refr,
            &self.weights,
            self.crpix,
            self.tangent,
            self.order,
            self.radial,
            self.proj,
        ) else {
            return f64::INFINITY;
        };
        crate::wcsfit::rms(&crate::wcsfit::residuals(&fit, &refr))
    }

    /// Lowest altitude (deg) of the pairs under zenith `z`.
    fn min_alt(&self, z: Vec3) -> f64 {
        self.t
            .iter()
            .map(|&t| 90.0 - dot(t, z).clamp(-1.0, 1.0).acos().to_degrees())
            .fold(f64::INFINITY, f64::min)
    }
}

/// Fewest matched stars for a measurement.
const MIN_PAIRS: usize = 60;
/// Stars considered (an even subsample of the pairs beyond this).
const MAX_PAIRS: usize = 250;
/// Refraction is measured only when some star is this low (deg): above, it is smooth enough
/// for the lens model to absorb.
const MAX_MIN_ALT: f64 = 20.0;
/// Largest rms (px) of the rigid lens with refraction for a reported measurement.
const MAX_RIGID_RMS: f64 = 1.0;
/// Accepted refraction scales (physical air densities plus measurement scatter).
const SCALE_RANGE: (f64, f64) = (0.5, 1.6);

/// Measure the refraction from the inlier `pairs` (x, y, ra, dec) of a solution with model
/// `wcs`. `None` when the frame shows no low stars or the refraction does not explain the
/// residuals (rms down by less than 15 %).
pub(super) fn measure(wcs: &Wcs, pairs: &[[f64; 4]], w: f64, h: f64) -> Option<Refraction> {
    if wcs.proj.is_cylindrical() || pairs.len() < MIN_PAIRS {
        return None;
    }
    let step = pairs.len().div_ceil(MAX_PAIRS);
    let sub: Vec<[f64; 4]> = pairs.iter().step_by(step).copied().collect();
    let crpix = [(w - 1.0) / 2.0, (h - 1.0) / 2.0];
    let n = sub.len();
    // a rigid lens: linear plus lens-like radial terms about the frame centre, no free SIP
    // polynomial (which would absorb the refraction, symmetric about the zenith instead)
    let p = Problem {
        pairs: &sub,
        t: sub.iter().map(|q| unit(q[2], q[3])).collect(),
        weights: vec![],
        crpix,
        tangent: wcs.pixel_to_sky(crpix[0], crpix[1]),
        order: 1,
        radial: if n >= 120 { 7 } else { 5 },
        proj: wcs.proj,
    };
    let rms0 = p.rms([0.0, 0.0, 1.0], 0.0);
    if !rms0.is_finite() {
        return None;
    }
    // coarse search: zeniths that keep every star above -1.5 deg and some star below 20 deg
    let c0 = unit(p.tangent[0], p.tangent[1]);
    let n_grid = 1200;
    let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
    let mut best = (f64::INFINITY, c0);
    let mut tried = 0;
    for i in 0..n_grid {
        let zc = 1.0 - 2.0 * (i as f64 + 0.5) / n_grid as f64;
        let r = (1.0 - zc * zc).sqrt();
        let z = [
            r * (golden * i as f64).cos(),
            r * (golden * i as f64).sin(),
            zc,
        ];
        // the zenith lies within 100 deg of the frame centre for any landscape
        if dot(z, c0) < (-10f64).to_radians().sin() {
            continue;
        }
        let lo = p.min_alt(z);
        if !(-1.5..=MAX_MIN_ALT).contains(&lo) {
            continue;
        }
        tried += 1;
        let e = p.rms(z, 1.0);
        if e < best.0 {
            best = (e, z);
        }
    }
    if tried == 0 || !best.0.is_finite() {
        return None;
    }
    // simplex over (east, north offsets of the zenith in rad, ln k)
    let z0 = best.1;
    let f = |x: [f64; 3]| -> f64 {
        let z = offset(z0, x[0], x[1]);
        let k = x[2].exp();
        if !(0.2..=4.0).contains(&k) || p.min_alt(z) < -1.5 {
            return f64::INFINITY;
        }
        p.rms(z, k)
    };
    let x = nelder_mead(f, [0.0, 0.0, 0.0], [0.035, 0.035, 0.25], 120);
    let z = offset(z0, x[0], x[1]);
    let k = x[2].exp();
    let rms1 = f(x);
    let lo = p.min_alt(z);
    if debug_on() {
        eprintln!(
            "  refraction: rms {rms0:.3} -> {rms1:.3} px, k {k:.2}, min alt {lo:.1} deg, {tried} zeniths tried"
        );
    }
    // report only a measurement the rigid lens supports: a good fit and a physical scale. The
    // air density at an observer is 0.55 (5 km up) to 1.25 (1050 hPa, -30 C) of the standard;
    // with the scatter of the measurement 0.5-1.6 is accepted. Larger scales are the search
    // exploiting a lens model that does not fit (in the synthetic set every scale above 2 came
    // with a zenith 80-110 deg off).
    let supported = rms1.is_finite()
        && rms1 < 0.85 * rms0
        && rms1 <= MAX_RIGID_RMS
        && (-1.0..=MAX_MIN_ALT).contains(&lo)
        && (SCALE_RANGE.0..=SCALE_RANGE.1).contains(&k);
    if !supported {
        return None;
    }
    let (ra, dec) = radec(z);
    Some(Refraction {
        zenith: [ra, dec],
        scale: k,
        rms_px: rms0,
        rms_refracted_px: rms1,
        min_alt_deg: lo,
        n_pairs: n,
    })
}

/// Minimise `f` from `x0` with initial steps `step` (Nelder-Mead, standard coefficients).
fn nelder_mead(
    f: impl Fn([f64; 3]) -> f64,
    x0: [f64; 3],
    step: [f64; 3],
    iters: usize,
) -> [f64; 3] {
    let mut s: Vec<([f64; 3], f64)> = (0..4)
        .map(|i| {
            let mut x = x0;
            if i > 0 {
                x[i - 1] += step[i - 1];
            }
            (x, f(x))
        })
        .collect();
    let comb = |a: [f64; 3], b: [f64; 3], t: f64| [0, 1, 2].map(|k| a[k] + t * (b[k] - a[k]));
    for _ in 0..iters {
        s.sort_by(|a, b| a.1.total_cmp(&b.1));
        let c = [0, 1, 2].map(|k| (s[0].0[k] + s[1].0[k] + s[2].0[k]) / 3.0);
        let worst = s[3];
        let xr = comb(c, worst.0, -1.0);
        let fr = f(xr);
        if fr < s[0].1 {
            let xe = comb(c, worst.0, -2.0);
            let fe = f(xe);
            s[3] = if fe < fr { (xe, fe) } else { (xr, fr) };
        } else if fr < s[2].1 {
            s[3] = (xr, fr);
        } else {
            let xc = comb(c, worst.0, 0.5);
            let fc = f(xc);
            if fc < worst.1 {
                s[3] = (xc, fc);
            } else {
                let b = s[0].0;
                for v in s.iter_mut().skip(1) {
                    v.0 = comb(b, v.0, 0.5);
                    v.1 = f(v.0);
                }
            }
        }
        let spread = s.iter().map(|v| v.1).fold(f64::MIN, f64::max) - s[0].1;
        if spread.abs() < 1e-6 {
            break;
        }
    }
    s.sort_by(|a, b| a.1.total_cmp(&b.1));
    s[0].0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saemundsson_matches_its_table() {
        // at true altitude 0 deg: 29.0' (Bennett's 34.5' is at apparent altitude 0), 10 deg:
        // 5.4', 45 deg: 1.01'
        let arcmin = |h: f64| saemundsson(h).to_degrees() * 60.0;
        assert!((arcmin(0.0) - 29.0).abs() < 0.2, "{}", arcmin(0.0));
        assert!((arcmin(10.0) - 5.40).abs() < 0.05, "{}", arcmin(10.0));
        assert!((arcmin(45.0) - 1.013).abs() < 0.01, "{}", arcmin(45.0));
    }

    #[test]
    fn refraction_lifts_towards_the_zenith() {
        let z = [0.0, 0.0, 1.0];
        let t = unit(30.0, 5.0); // 5 deg up
        let a = refract(t, z, 1.0);
        let up = |v: Vec3| v[2].asin().to_degrees();
        let lift = (up(a) - up(t)) * 60.0;
        assert!((lift - 9.9).abs() < 0.3, "{lift}");
        assert_eq!(radec(a).0.round(), 30.0);
    }

    /// A synthetic landscape: an ARC lens looking at the horizon, stars refracted with k = 1.15.
    /// The zenith comes back within a degree, k within 0.1.
    #[test]
    fn recovers_zenith_and_scale_from_refracted_stars() {
        let (w, h) = (2000.0, 1333.0);
        let zen = unit(120.0, 50.0);
        // boresight 25 deg above the horizon, towards azimuth "east" of the zenith frame
        let (e, _) = tangent_basis(zen);
        let b = crate::sphere::normalize([
            zen[0] * 25f64.to_radians().sin() + e[0] * 25f64.to_radians().cos(),
            zen[1] * 25f64.to_radians().sin() + e[1] * 25f64.to_radians().cos(),
            zen[2] * 25f64.to_radians().sin() + e[2] * 25f64.to_radians().cos(),
        ]);
        let (bra, bdec) = radec(b);
        let s = 0.05;
        let wcs = Wcs {
            ra: bra,
            dec: bdec,
            crpix: [(w - 1.0) / 2.0, (h - 1.0) / 2.0],
            cd: [[-s, 0.0], [0.0, s]],
            sip: None,
            proj: Proj::Arc,
            native_pole: None,
            local_grid: None,
        };
        let mut pairs = vec![];
        let mut q = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = || {
            q ^= q << 13;
            q ^= q >> 7;
            q ^= q << 17;
            (q >> 11) as f64 / (1u64 << 53) as f64
        };
        while pairs.len() < 300 {
            let (x, y) = (rnd() * w, rnd() * h);
            let sky = wcs.pixel_to_sky(x, y); // apparent direction of the pixel
            let a = unit(sky[0], sky[1]);
            if 90.0 - dot(a, zen).acos().to_degrees() < 0.5 {
                continue; // below the horizon
            }
            // the true direction: undo the refraction (fixed point)
            let mut t = a;
            for _ in 0..20 {
                let r = refract(t, zen, 1.15);
                t = crate::sphere::normalize([
                    t[0] + a[0] - r[0],
                    t[1] + a[1] - r[1],
                    t[2] + a[2] - r[2],
                ]);
            }
            let (ra, dec) = radec(t);
            pairs.push([x, y, ra, dec]);
        }
        let r = measure(&wcs, &pairs, w, h).expect("refraction measured");
        let err = crate::sphere::angle(unit(r.zenith[0], r.zenith[1]), zen).to_degrees();
        assert!(err < 1.0, "zenith off by {err} deg");
        assert!((r.scale - 1.15).abs() < 0.1, "scale {}", r.scale);
        assert!(r.rms_refracted_px < 0.2 * r.rms_px);
    }
}
