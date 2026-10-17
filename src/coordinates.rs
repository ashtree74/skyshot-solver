//! Deterministic WCS transforms extracted from the shared geometry layer.
use crate::sphere::{Vec3, dot};
use crate::types::{Proj, Wcs};
use std::f64::consts::PI;

pub fn unit(ra_deg: f64, dec_deg: f64) -> Vec3 {
    let (r, d) = (ra_deg.to_radians(), dec_deg.to_radians());
    let cd = libm::cos(d);
    [cd * libm::cos(r), cd * libm::sin(r), libm::sin(d)]
}

/// Gnomonic offset (degrees, east and north) of (ra, dec) seen from (ra0, dec0); `None` on the
/// far hemisphere.
fn tan_project(ra: f64, dec: f64, ra0: f64, dec0: f64) -> Option<[f64; 2]> {
    let (r, d, r0, d0) = (
        ra.to_radians(),
        dec.to_radians(),
        ra0.to_radians(),
        dec0.to_radians(),
    );
    let dr = r - r0;
    let (sd, cd, sd0, cd0, cdr) = (
        libm::sin(d),
        libm::cos(d),
        libm::sin(d0),
        libm::cos(d0),
        libm::cos(dr),
    );
    let den = sd0 * sd + cd0 * cd * cdr;
    if den <= 0.0 {
        return None;
    }
    Some([
        (cd * libm::sin(dr) / den).to_degrees(),
        ((cd0 * sd - sd0 * cd * cdr) / den).to_degrees(),
    ])
}

fn tan_unproject(x: f64, y: f64, ra0: f64, dec0: f64) -> [f64; 2] {
    let (x, y, r, d) = (
        x.to_radians(),
        y.to_radians(),
        ra0.to_radians(),
        dec0.to_radians(),
    );
    let (sd, cd) = (libm::sin(d), libm::cos(d));
    let den = cd - y * sd;
    [
        (r + libm::atan2(x, den)).to_degrees().rem_euclid(360.0),
        libm::atan2(sd + y * cd, (den * den + x * x).sqrt()).to_degrees(),
    ]
}

/// East and north unit vectors at (ra0, dec0), with the unit vector there (libm).
fn zenithal_basis(ra0: f64, dec0: f64) -> (Vec3, Vec3, Vec3) {
    let (a, d) = (ra0.to_radians(), dec0.to_radians());
    let (sa, ca, sd, cd) = (libm::sin(a), libm::cos(a), libm::sin(d), libm::cos(d));
    (
        [cd * ca, cd * sa, sd],
        [-sa, ca, 0.0],
        [-sd * ca, -sd * sa, cd],
    )
}

/// Zenithal equidistant (ARC) offset (degrees, east and north; its length is the angular
/// distance) of (ra, dec) seen from (ra0, dec0): `geometry::project_arc` with libm. `None`
/// within 1e-6 rad of the antipode.
fn arc_project(ra: f64, dec: f64, ra0: f64, dec0: f64) -> Option<[f64; 2]> {
    let v = unit(ra, dec);
    let (c, e, n) = zenithal_basis(ra0, dec0);
    let (ve, vn, vc) = (dot(e, v), dot(n, v), dot(c, v));
    let s = libm::hypot(ve, vn);
    let rho = libm::atan2(s, vc);
    if rho > PI - 1e-6 {
        return None;
    }
    let k = if s < 1e-12 { 1.0 } else { rho / s };
    Some([(ve * k).to_degrees(), (vn * k).to_degrees()])
}

/// Inverse of [`arc_project`] (`geometry::unproject_arc` with libm).
fn arc_unproject(x: f64, y: f64, ra0: f64, dec0: f64) -> [f64; 2] {
    let (xr, yr) = (x.to_radians(), y.to_radians());
    let rho = libm::hypot(xr, yr);
    let (c, e, n) = zenithal_basis(ra0, dec0);
    let k = if rho < 1e-12 {
        1.0
    } else {
        libm::sin(rho) / rho
    };
    let cr = libm::cos(rho);
    let v: [f64; 3] = std::array::from_fn(|i| c[i] * cr + k * (xr * e[i] + yr * n[i]));
    [
        libm::atan2(v[1], v[0]).to_degrees().rem_euclid(360.0),
        libm::atan2(v[2], libm::hypot(v[0], v[1])).to_degrees(),
    ]
}

/// Sky -> intermediate coordinates (degrees) of the WCS's projection.
fn proj_project(w: &Wcs, ra: f64, dec: f64) -> Option<[f64; 2]> {
    match w.proj {
        Proj::Tan => tan_project(ra, dec, w.ra, w.dec),
        Proj::Arc => arc_project(ra, dec, w.ra, w.dec),
        _ => w.projection().prepare([w.ra, w.dec]).project(ra, dec),
    }
}

/// Pixel (zero-based) to sky, degrees: SIP, CD, then the WCS's projection (TAN or ARC).
pub fn pixel_to_sky(w: &Wcs, x: f64, y: f64) -> [f64; 2] {
    let Some([x, y]) = w.uncorrect_pixel([x, y]) else {
        return [f64::NAN; 2];
    };
    let (mut dx, mut dy) = (x - w.crpix[0], y - w.crpix[1]);
    if let Some(s) = &w.sip {
        let (fa, fb) = s.eval(dx, dy);
        dx += fa;
        dy += fb;
    }
    let (px, py) = (
        w.cd[0][0] * dx + w.cd[0][1] * dy,
        w.cd[1][0] * dx + w.cd[1][1] * dy,
    );
    match w.proj {
        Proj::Tan => tan_unproject(px, py, w.ra, w.dec),
        Proj::Arc => arc_unproject(px, py, w.ra, w.dec),
        _ => w.projection().unproject(px, py, [w.ra, w.dec]),
    }
}

/// Sky to pixel (zero-based): the WCS's projection (TAN or ARC), then the SIP polynomial
/// inverted by Newton to 1e-7 px. `None` on the far hemisphere (TAN) or at the antipode (ARC),
/// for a singular CD or Jacobian, or when Newton does not converge in 20 steps (far outside a
/// distorted frame).
pub fn sky_to_pixel(w: &Wcs, ra: f64, dec: f64) -> Option<[f64; 2]> {
    let p = proj_project(w, ra, dec)?;
    let det = w.cd[0][0] * w.cd[1][1] - w.cd[0][1] * w.cd[1][0];
    if det.abs() < 1e-16 {
        return None;
    }
    let tx = (w.cd[1][1] * p[0] - w.cd[0][1] * p[1]) / det;
    let ty = (-w.cd[1][0] * p[0] + w.cd[0][0] * p[1]) / det;
    let (mut x, mut y) = (tx, ty);
    if let Some(s) = &w.sip {
        let mut done = false;
        for _ in 0..20 {
            let (fa, fb) = s.eval(x, y);
            let (ex, ey) = (x + fa - tx, y + fb - ty);
            let j = s.jacobian(x, y);
            let dj = j[0][0] * j[1][1] - j[0][1] * j[1][0];
            if dj.abs() < 1e-12 {
                return None;
            }
            let ddx = (j[1][1] * ex - j[0][1] * ey) / dj;
            let ddy = (-j[1][0] * ex + j[0][0] * ey) / dj;
            x -= ddx;
            y -= ddy;
            if (ddx * ddx + ddy * ddy).sqrt() < 1e-7 {
                done = true;
                break;
            }
        }
        if !done {
            return None;
        }
    }
    (x.is_finite() && y.is_finite()).then(|| w.correct_pixel([x + w.crpix[0], y + w.crpix[1]]))
}

/// Plate scale at the frame centre (arcsec per pixel, from one-pixel steps through the full
/// WCS, as `api::field::field_summary` measures it).
pub fn scale_arcsec(w: &Wcs, width: u32, height: u32) -> Option<f64> {
    let c = [
        (f64::from(width) - 1.0) / 2.0,
        (f64::from(height) - 1.0) / 2.0,
    ];
    let s0 = pixel_to_sky(w, c[0], c[1]);
    let step = |dx: f64, dy: f64| {
        let s = pixel_to_sky(w, c[0] + dx, c[1] + dy);
        tan_project(s[0], s[1], s0[0], s0[1]).unwrap_or([0.0; 2])
    };
    let (jx, jy) = (step(1.0, 0.0), step(0.0, 1.0));
    let sc = (jx[0] * jy[1] - jy[0] * jx[1]).abs().sqrt() * 3600.0;
    (sc.is_finite() && sc > 0.0).then_some(sc)
}
