//! High-pass (difference-of-Gaussians) star detection: a fallback for frames dominated by
//! structured nebulosity, where tile backgrounds cannot follow the sky. The DoG with
//! sigma1 ~ PSF and sigma2 = 3 sigma1 cancels any structure much larger than a star; peaks are
//! found against a robust per-tile noise of the DoG image and measured at native resolution.

use crate::{
    image_io::GrayImage,
    types::{Star, StarField},
};
use rayon::prelude::*;

/// Box radius of [`box_blur3`] for a Gaussian of `sigma`: three passes of a (2r + 1) box.
pub(crate) fn box_radius(sigma: f32) -> usize {
    ((((12.0 * sigma * sigma / 3.0) + 1.0).sqrt() - 1.0) / 2.0)
        .round()
        .max(1.0) as usize
}

/// Planes smaller than this are blurred on the calling thread (forced photometry blurs small
/// patches around each target from inside a parallel loop). Rows and columns are independent,
/// so the result is the same either way.
const PAR_MIN_PX: usize = 1 << 16;

/// Approximate Gaussian blur by three box passes (separable, O(n)) on a w x h plane, edges
/// clamped. Also used on patches by [`crate::measure`] (its DoG confirmation).
pub(crate) fn box_blur3(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let r = box_radius(sigma);
    let n = (2 * r + 1) as f32;
    let par = w * h >= PAR_MIN_PX;
    let row_blur = |(dst, row): (&mut [f32], &[f32])| {
        let mut acc: f32 = row[..=r.min(w - 1)].iter().sum();
        acc += row[0] * r as f32;
        for x in 0..w {
            dst[x] = acc / n;
            let add = row[(x + r + 1).min(w - 1)];
            let sub = row[x.saturating_sub(r)];
            acc += add - sub;
        }
    };
    let col_blur = |x0: usize, x1: usize, src: &[f32], out: &mut [f32]| {
        for x in x0..x1 {
            let mut acc = 0f32;
            for y in 0..=r.min(h - 1) {
                acc += src[y * w + x];
            }
            acc += src[x] * r as f32;
            for y in 0..h {
                out[y * w + x] = acc / n;
                let add = src[(y + r + 1).min(h - 1) * w + x];
                let sub = src[y.saturating_sub(r) * w + x];
                acc += add - sub;
            }
        }
    };
    let mut a = src.to_vec();
    let mut b = vec![0f32; src.len()];
    for _ in 0..3 {
        // horizontal
        if par {
            b.par_chunks_mut(w).zip(a.par_chunks(w)).for_each(row_blur);
        } else {
            b.chunks_mut(w).zip(a.chunks(w)).for_each(row_blur);
        }
        // vertical (transpose-free: process columns in strips)
        let mut out = vec![0f32; src.len()];
        if par {
            let strips = 64usize;
            let chunk = w.div_ceil(strips);
            let ptr = SendPtr(out.as_mut_ptr());
            (0..strips).into_par_iter().for_each(|s| {
                let x0 = s * chunk;
                let x1 = ((s + 1) * chunk).min(w);
                if x0 >= x1 {
                    return;
                }
                // SAFETY: strips write disjoint column ranges of `out`.
                let o = unsafe { std::slice::from_raw_parts_mut(ptr.get(), w * h) };
                col_blur(x0, x1, &b, o);
            });
        } else {
            col_blur(0, w, &b, &mut out);
        }
        a = out;
    }
    a
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    #[inline]
    fn get(self) -> *mut f32 {
        self.0
    }
}

fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let n = v.len() / 2;
    *v.select_nth_unstable_by(n, |a, b| a.total_cmp(b)).1
}

/// Area of the clipped core (pixels at the frame maximum `sat`, 8-connected) touching the 3x3
/// block around a peak; 0 when the peak is not clipped. Capped at 4096 pixels.
fn clipped_area(
    plane: &[f32],
    w: usize,
    h: usize,
    (x, y): (usize, usize),
    sat: f32,
    seen: &mut std::collections::HashSet<usize>,
) -> usize {
    seen.clear();
    let mut stack: Vec<usize> = vec![];
    for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
        for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
            let i = yy * w + xx;
            if plane[i] >= sat && seen.insert(i) {
                stack.push(i);
            }
        }
    }
    let mut n = 0;
    while let Some(i) = stack.pop() {
        n += 1;
        if n >= 4096 {
            break;
        }
        let (cx, cy) = (i % w, i / w);
        for yy in cy.saturating_sub(1)..=(cy + 1).min(h - 1) {
            for xx in cx.saturating_sub(1)..=(cx + 1).min(w - 1) {
                let j = yy * w + xx;
                if plane[j] >= sat && seen.insert(j) {
                    stack.push(j);
                }
            }
        }
    }
    n
}

/// Detect stars with a DoG filter. `fwhm_px` is the expected star FWHM (2.5 is a robust default).
pub fn detect_highpass(img: &GrayImage, fwhm_px: f32, threshold: f32) -> StarField {
    let (w0, h0) = (img.width as usize, img.height as usize);
    // bin 2x2 above ~8 MP for speed
    let bin = if w0 * h0 > 8_000_000 { 2 } else { 1 };
    let (w, h) = (w0 / bin, h0 / bin);
    let mut plane = vec![0f32; w * h];
    plane.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            let mut s = 0f32;
            let mut n = 0f32;
            for dy in 0..bin {
                for dx in 0..bin {
                    let p = img.pixels[(y * bin + dy) * w0 + x * bin + dx];
                    if p.is_finite() {
                        s += p;
                        n += 1.0;
                    }
                }
            }
            *v = if n > 0.0 { s / n } else { 0.0 };
        }
    });
    if w < 16 || h < 16 {
        return StarField {
            width: img.width,
            height: img.height,
            stars: vec![],
            epoch: img.epoch,
        };
    }
    let s1 = (fwhm_px / bin as f32 / 2.355).max(0.7);
    let g1 = box_blur3(&plane, w, h, s1);
    let g2 = box_blur3(&plane, w, h, 3.0 * s1);
    let dog: Vec<f32> = g1.iter().zip(&g2).map(|(a, b)| a - b).collect();
    // robust noise per 64x64 tile of the DoG plane
    let t = 64usize;
    let (tx, ty) = (w.div_ceil(t), h.div_ceil(t));
    let noise: Vec<f32> = (0..tx * ty)
        .into_par_iter()
        .map(|k| {
            let (cx, cy) = (k % tx, k / tx);
            let mut v = vec![];
            for y in (cy * t..((cy + 1) * t).min(h)).step_by(2) {
                for x in (cx * t..((cx + 1) * t).min(w)).step_by(2) {
                    v.push(dog[y * w + x]);
                }
            }
            let m = median(&mut v);
            for x in v.iter_mut() {
                *x = (*x - m).abs();
            }
            (1.4826 * median(&mut v)).max(1e-6)
        })
        .collect();
    // peaks: 3x3 maxima above threshold, brightest 48 per tile
    let peaks: Vec<(usize, usize, f32)> = (1..h - 1)
        .into_par_iter()
        .flat_map_iter(|y| {
            let dog = &dog;
            let noise = &noise;
            (1..w - 1).filter_map(move |x| {
                let v = dog[y * w + x];
                let sig = noise[(y / t) * tx + x / t];
                if v < threshold * sig {
                    return None;
                }
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if (dx != 0 || dy != 0)
                            && dog[(y as i32 + dy) as usize * w + (x as i32 + dx) as usize] > v
                        {
                            return None;
                        }
                    }
                }
                Some((x, y, v))
            })
        })
        .collect();
    let mut by_tile: std::collections::HashMap<usize, Vec<(usize, usize, f32)>> =
        std::collections::HashMap::new();
    for p in peaks {
        by_tile.entry((p.1 / t) * tx + p.0 / t).or_default().push(p);
    }
    let base: Star = serde_json::from_str("{\"x\":0,\"y\":0,\"flux\":1}").expect("default star");
    let sat = plane
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let mut seen = std::collections::HashSet::new();
    let mut stars: Vec<Star> = vec![];
    // clipped-core area of every star (binned pixels)
    let mut clipped: Vec<usize> = vec![];
    for (_, mut v) in by_tile {
        v.sort_by(|a, b| b.2.total_cmp(&a.2));
        v.truncate(48);
        for (x, y, val) in v {
            // quadratic sub-pixel refinement on the DoG plane
            let c = dog[y * w + x];
            let (l, r) = (dog[y * w + x - 1], dog[y * w + x + 1]);
            let (u, d) = (dog[(y - 1) * w + x], dog[(y + 1) * w + x]);
            let ox = {
                let den = l - 2.0 * c + r;
                if den < 0.0 {
                    (0.5 * (l - r) / den).clamp(-0.5, 0.5)
                } else {
                    0.0
                }
            };
            let oy = {
                let den = u - 2.0 * c + d;
                if den < 0.0 {
                    (0.5 * (u - d) / den).clamp(-0.5, 0.5)
                } else {
                    0.0
                }
            };
            let fx = (x as f32 + ox + 0.5) * bin as f32 - 0.5;
            let fy = (y as f32 + oy + 0.5) * bin as f32 - 0.5;
            let sig = noise[(y / t) * tx + x / t];
            // brightness: DoG energy in a 3x3 block (insensitive to smooth backgrounds)
            let mut e = 0f32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    e += dog[(y as i32 + dy) as usize * w + (x as i32 + dx) as usize].max(0.0);
                }
            }
            let snr = (val / sig) as f64;
            clipped.push(clipped_area(&plane, w, h, (x, y), sat, &mut seen));
            stars.push(Star {
                x: fx as f64,
                y: fy as f64,
                flux: e as f64,
                sigma: (0.6 * fwhm_px as f64 / snr.max(1.0)).clamp(0.1, 1.5),
                snr,
                fwhm: fwhm_px as f64,
                elongation: 1.0,
                saturated: false,
                ..base.clone()
            });
        }
    }
    // A clipped star is brighter than every unclipped one, and its DoG energy is no measure
    // of how much brighter: once the flat top is wider than the inner Gaussian the energy
    // falls as the star grows, so a stretched frame whose stars all saturate was ranked at
    // random. Rank clipped stars above the brightest unclipped one by the area of the clipped
    // core: for a Gaussian clipped at S, area = 2 pi s^2 ln(peak / S). The energy breaks ties
    // between equal areas (less than half a pixel of area). One clipped pixel is a hot pixel
    // or a star at the limit and keeps its energy.
    let e_max = stars
        .iter()
        .map(|s| s.flux)
        .fold(0.0, f64::max)
        .max(f64::MIN_POSITIVE);
    let e_sat = stars
        .iter()
        .zip(&clipped)
        .filter(|(_, a)| **a < 2)
        .map(|(s, _)| s.flux)
        .fold(0.0, f64::max)
        .max(e_max * 1e-3);
    let two_pi_s2 = 2.0 * std::f64::consts::PI * (s1 as f64).powi(2);
    for (s, &a) in stars.iter_mut().zip(&clipped) {
        if a >= 2 {
            let area = a as f64 + 0.5 * s.flux / e_max;
            s.flux = e_sat * (area / two_pi_s2).min(40.0).exp();
            s.saturated = true;
        }
    }
    // tiles come out of a hash map: break exact ties by position for a deterministic order
    stars.sort_by(|a, b| {
        b.flux
            .total_cmp(&a.flux)
            .then(a.y.total_cmp(&b.y))
            .then(a.x.total_cmp(&b.x))
    });
    StarField {
        width: img.width,
        height: img.height,
        stars,
        epoch: img.epoch,
    }
}
