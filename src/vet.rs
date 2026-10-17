//! Frame-relative vetting of extracted sources (ARCHITECTURE §2.1, `vet.rs`):
//! frame PSF and trailed-frame detection, hot pixels / cosmic rays, elongation
//! outliers. Thresholds are relative to the frame so that undersampled and
//! trailed frames are not emptied.
use crate::{
    extract::{Cand, Ctx, Det, Trail},
    types::star_flags as flag,
};

/// Frame point-spread summary measured on a sample of bright candidates.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FramePsf {
    /// Median FWHM in native pixels (0 when there were too few stars).
    pub fwhm: f32,
    pub trail: Option<Trail>,
}

fn median(v: &mut [f64]) -> f64 {
    quantile(v, 0.5)
}

fn quantile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let i = ((v.len() as f64 * q) as usize).min(v.len() - 1);
    *v.select_nth_unstable_by(i, f64::total_cmp).1
}

/// Median FWHM and trail model from up to 300 moderately bright candidates
/// (skipping the very brightest, which are often clipped).
pub(crate) fn frame_psf(ctx: &Ctx, cands: &[Cand]) -> FramePsf {
    if cands.len() < 5 {
        return FramePsf::default();
    }
    // Spatially spread sample: the brightest candidate of every tile, then the
    // second brightest, ... so that one streak or one cluster cannot dominate.
    // Candidates failing a provisional ridge test (ctx.fwhm_frame is a default
    // here) are skipped, so that a nebula or a streak cannot define the PSF.
    let key = |i: usize| cands[i].v - cands[i].bg;
    let tiles = ctx.grid.nx * ctx.grid.ny;
    let mut taken = vec![false; cands.len()];
    let mut order: Vec<usize> = vec![];
    for _ in 0..6 {
        let mut best = vec![usize::MAX; tiles];
        for (i, c) in cands.iter().enumerate() {
            if taken[i] {
                continue;
            }
            let t = ctx.grid.tile_index(c.x as usize, c.y as usize);
            if best[t] == usize::MAX || key(i) > key(best[t]) {
                best[t] = i;
            }
        }
        let mut round: Vec<usize> = best.into_iter().filter(|&i| i != usize::MAX).collect();
        if round.is_empty() {
            break;
        }
        round.sort_by(|&a, &b| key(b).total_cmp(&key(a)));
        for i in round {
            taken[i] = true;
            if order.len() < 420 && ctx.is_compact_star(&cands[i]) {
                order.push(i);
            }
        }
        if order.len() >= 420 {
            break;
        }
    }
    order.sort_by(|&a, &b| key(b).total_cmp(&key(a)));
    let skip = order.len() / 20;
    order.truncate(skip + 300);
    // Stars ≥ 10σ above the sky; in very noisy (stretched, 8-bit) frames where
    // fewer than five reach that, every candidate above the detection threshold,
    // so that the frame PSF (and with it the ridge test, deduplication and
    // vetting scales) is still measured.
    let mut samples: Vec<PsfSample> = vec![];
    for min_snr in [10.0, ctx.k.min(10.0)] {
        if samples.len() >= 5 {
            break;
        }
        samples = order[skip..]
            .iter()
            .filter_map(|&i| psf_sample(ctx, &cands[i], min_snr))
            .collect();
    }
    let streak = streak_members(&samples);
    let samples: Vec<&PsfSample> = samples
        .iter()
        .zip(&streak)
        .filter(|(_, on)| !**on)
        .map(|(s, _)| s)
        .collect();
    if samples.len() < 5 {
        return FramePsf::default();
    }
    let mut fwhm: Vec<f64> = samples.iter().map(|s| s.fwhm).collect();
    let mut elong: Vec<f64> = samples
        .iter()
        .filter_map(|s| s.shape)
        .map(|e| e.0)
        .collect();
    let n = elong.len();
    let (mut c2, mut s2, mut n_e) = (0.0f64, 0.0f64, 0usize);
    let mut length = vec![];
    for &(e, pa, len) in samples.iter().filter_map(|s| s.shape.as_ref()) {
        if e > 1.3 {
            let a = (2.0 * pa).to_radians();
            c2 += a.cos();
            s2 += a.sin();
            n_e += 1;
            length.push(len);
        }
    }
    // Stars are the most compact sources in a frame; a lower quantile keeps the
    // PSF estimate robust when nebular knots or blends dominate the sample.
    let fwhm_med = quantile(&mut fwhm, 0.35);
    let elong_med = median(&mut elong);
    let coherence = if n_e > 0 {
        c2.hypot(s2) / n_e as f64
    } else {
        0.0
    };
    let trail =
        (n >= 10 && elong_med > 1.6 && coherence > 0.8 && n_e * 10 >= n * 6).then(|| Trail {
            pa: (0.5 * s2.atan2(c2)).rem_euclid(std::f64::consts::PI) as f32,
            length: median(&mut length) as f32,
        });
    FramePsf {
        fwhm: fwhm_med as f32,
        trail,
    }
}

/// One PSF sample: position, FWHM and, when the large window fits,
/// (elongation, position angle in degrees, trail length).
struct PsfSample {
    x: f64,
    y: f64,
    fwhm: f64,
    shape: Option<(f64, f64, f64)>,
}

fn psf_sample(ctx: &Ctx, cand: &Cand, min_snr: f32) -> Option<PsfSample> {
    let (px, py) = ctx.native_peak(cand)?;
    let (plane, sig) = ctx.plane_bg(px, py);
    let (bg, pv) = (plane.c, ctx.px(px, py));
    if (pv - bg).is_nan() || pv - bg < min_snr * sig {
        return None;
    }
    let thr = (1.5 * sig).max(0.02 * (pv - bg));
    let f = (thr / (pv - bg)) as f64;
    let small = ctx.moments(px as f64, py as f64, 3, plane, thr);
    let small = if small.shape(f).0 > 4.0 {
        ctx.moments(small.x, small.y, 5, plane, thr)
    } else {
        small
    };
    if small.sum <= 0.0 || small.clipped {
        return None;
    }
    let big = ctx.moments(small.x, small.y, 8, plane, thr);
    let shape = (big.sum > 0.0 && !big.clipped).then(|| {
        let (_, e, pa) = big.shape(f);
        let (major, minor) = big.axes();
        (e, pa, (12.0 * (major - minor).max(0.0)).sqrt())
    });
    Some(PsfSample {
        x: small.x,
        y: small.y,
        fwhm: small.shape(f).0,
        shape,
    })
}

/// Samples on a satellite / airplane streak: strongly elongated and, with at
/// least three other such samples, on one line along their common direction.
/// The PSF sample takes up to six peaks per tile, so a streak crossing a sparse
/// field can outnumber its stars: it doubled the frame FWHM and switched the
/// frame to trailed mode. Trailed stars are spread over the frame instead (a
/// few of them may be collinear by chance; the rest still define the trail).
fn streak_members(samples: &[PsfSample]) -> Vec<bool> {
    let long: Vec<(usize, f64, f64, f64)> = samples
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            let (e, pa, _) = s.shape?;
            (e > 1.8).then_some((i, s.x, s.y, pa))
        })
        .collect();
    let mut on = vec![false; samples.len()];
    for &(i, x, y, pa) in &long {
        let (sn, cs) = pa.to_radians().sin_cos();
        let width = (0.5 * samples[i].fwhm).max(2.0);
        let collinear = long
            .iter()
            .filter(|&&(j, xj, yj, pj)| {
                let dpa = (pj - pa).rem_euclid(180.0);
                j != i
                    && dpa.min(180.0 - dpa) < 10.0
                    && ((xj - x) * sn - (yj - y) * cs).abs() < width
            })
            .count();
        on[i] = collinear >= 3;
    }
    on
}

/// Flag hot pixels / cosmic rays (too sharp for this frame), trailed sources
/// and elongation outliers.
pub(crate) fn vet(ctx: &Ctx, dets: &mut [Det]) {
    let clean = |d: &&Det| {
        d.flags & (flag::SATURATED | flag::NEAR_EDGE | flag::RIDGE) == 0 && d.h >= 10.0 * d.sig_n
    };
    let mut ratio: Vec<f64> = dets.iter().filter(clean).map(|d| d.ratio4 as f64).collect();
    let mut fwhm: Vec<f64> = dets.iter().filter(clean).map(|d| d.fwhm).collect();
    let mut elong: Vec<f64> = dets.iter().filter(clean).map(|d| d.elong).collect();
    let enough = ratio.len() >= 5;
    let ratio_med = if enough { median(&mut ratio) } else { 0.5 };
    let fwhm_med = if enough {
        median(&mut fwhm)
    } else {
        ctx.fwhm_frame.max(2.0) as f64
    };
    let elong_med = if enough { median(&mut elong) } else { 1.2 };
    let sharp_limit = (0.45 * ratio_med).min(0.15);
    let broad_limit = (2.5 * fwhm_med).max(fwhm_med + 4.0);
    let fwhm_limit = (0.5 * fwhm_med).min(1.2);
    // Width of a Gaussian with the source's flux and peak (≈ 0.85 FWHM for a
    // star). Light packed into a few pixels — a cosmic ray or hot-pixel
    // cluster, often clipped, which the absolute 1.2 px FWHM limit misses on
    // well-sampled frames — is far narrower than the frame's sources. Faint
    // stars can be much sharper than the median (photographic plates,
    // stretched JPEGs), hence 0.4; a source clipped at the frame's saturation
    // level is at least as wide as an unclipped star, hence 0.6.
    let width = |d: &Det| (d.flux / (1.1331 * d.h as f64)).max(0.0).sqrt();
    let mut widths: Vec<f64> = dets.iter().filter(clean).map(width).collect();
    let width_med = if enough { median(&mut widths) } else { 0.0 };
    let sat_tol = (0.5 * ctx.lsb).max(1e-6 * ctx.sat_level.abs());
    for d in dets.iter_mut() {
        let saturated = d.flags & flag::SATURATED != 0;
        if d.h >= 10.0 * d.sig_n {
            let sharp = (d.ratio4 as f64) < sharp_limit;
            let tiny = d.fwhm < fwhm_limit;
            let peaky = d.fwhm < 1.5 && d.h as f64 > 0.7 * d.flux && fwhm_med > 2.0;
            let clipped = saturated || d.bg + d.h >= ctx.sat_level - sat_tol;
            // (Only where the aperture flux is trustworthy: on structured
            // backgrounds a low SNR flags an underestimated flux.)
            let packed = d.snr >= 20.0 && width(d) < if clipped { 0.6 } else { 0.4 } * width_med;
            if packed || (!saturated && (sharp || tiny || peaky)) {
                d.flags |= flag::HOT_PIXEL;
            }
        }
        // Bright stars broaden (halo, photographic saturation): only sources
        // well below the saturation proxy can be called extended.
        let bright = d.h >= 0.2 * (ctx.sat_level - d.bg);
        if !saturated && !bright && d.core == 0.0 && enough && d.fwhm > broad_limit {
            d.flags |= flag::EXTENDED;
        }
        if let Some(t) = ctx.trail {
            let diff = (d.pa - (t.pa as f64).to_degrees()).rem_euclid(180.0);
            if d.elong > 1.2 && diff.min(180.0 - diff) < 20.0 {
                d.flags |= flag::TRAILED;
            }
        } else if !saturated {
            let floor = if d.snr >= 30.0 { 2.5 } else { 4.0 };
            if d.elong > (2.0 * elong_med).max(floor) {
                d.flags |= flag::ELONGATED;
            }
        }
    }
}
