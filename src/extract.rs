//! Fast, robust star extractor (ARCHITECTURE §2.1). Replaces v1 `detect` for the
//! solver; `detect.rs` stays as the comparison baseline.
//!
//! Pipeline (all heavy stages run in parallel with rayon):
//! 1. box-bin for detection only (1×1 up to 6.5 MP, 2×2 up to 26 MP, then 4×4);
//! 2. per-tile robust background (median, MAD, posterization-aware noise floor),
//!    bilinear interpolation between tile centres;
//! 3. peaks above `bg + k·σ`, strict 3×3 maximum, ≥ 3 support pixels, per-tile cap,
//!    no global cap; a local-contrast test rejects nebular ridges and streak bumps;
//! 4. measurement at native resolution: Gaussian-windowed centroid (7×7, 11×11 when
//!    FWHM > 4, 5×5 compact), plateau centroid for clipped discs, annulus background
//!    on structured backgrounds, aperture flux, FWHM, elongation, position angle;
//! 5. deblending (secondary maxima, saddle test), duplicate merge, vetting
//!    ([`crate::vet`]: hot pixels, elongation, trailed frames) and streak removal
//!    ([`crate::lines`]).
//!
//! Output stars are sorted by background-subtracted aperture flux, brightest first.
use crate::{
    image_io::{GrayImage, SourceInfo},
    lines,
    types::{Star, StarField, star_flags as flag},
    vet,
};
use rayon::prelude::*;
use std::borrow::Cow;

/// Extraction settings. `Default` is the solver configuration.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ExtractOptions {
    /// Detection threshold above the local background, in units of the
    /// (binned) background noise.
    pub threshold_sigma: f32,
    /// Native-resolution detection with 5×5 measurement windows, for
    /// undersampled stars, dense clusters and structured nebular frames.
    pub compact: bool,
    /// Always measure against a local annulus background (otherwise the annulus
    /// is used only where it differs significantly from the tile model).
    pub local_background: bool,
    /// Keep at most this many of the brightest peaks per 64×64 px tile.
    pub max_per_tile: usize,
    /// Detection bin factor; 0 selects it from the frame size.
    pub bin: usize,
    /// Quantization step of the pixel values (see [`SourceInfo::lsb`]);
    /// 0 detects integer / 8-bit / 16-bit lattices from the data.
    pub lsb: f32,
    /// Keep rejected detections (hot pixels, ridges, streaks) with their flags.
    pub keep_rejected: bool,
    /// Find and remove collinear chains (satellite/airplane trails, bad columns).
    pub remove_lines: bool,
}

impl Default for ExtractOptions {
    fn default() -> Self {
        Self {
            threshold_sigma: 5.0,
            compact: false,
            local_background: false,
            max_per_tile: 64,
            bin: 0,
            lsb: 0.0,
            keep_rejected: false,
            remove_lines: true,
        }
    }
}

impl ExtractOptions {
    /// Default options carrying the quantization step reported by the decoder.
    pub fn for_source(info: &SourceInfo) -> Self {
        Self {
            lsb: info.lsb,
            ..Self::default()
        }
    }
}

/// Native tile size of the background model, in pixels.
const TILE_NATIVE: usize = 64;

/// Stage timer, active when `SKYANCHOR_PROFILE` is set (prints to stderr). Reads no clock
/// otherwise: WebAssembly in a browser has no `Instant`.
struct Profile {
    last: Option<std::time::Instant>,
    stages: Vec<(&'static str, f64)>,
}

impl Profile {
    fn new() -> Self {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let on = *ON.get_or_init(|| std::env::var_os("SKYANCHOR_PROFILE").is_some());
        Self {
            last: on.then(std::time::Instant::now),
            stages: vec![],
        }
    }
    fn on(&self) -> bool {
        self.last.is_some()
    }
    fn mark(&mut self, stage: &'static str) {
        if let Some(last) = self.last {
            let now = std::time::Instant::now();
            self.stages.push((stage, (now - last).as_secs_f64() * 1e3));
            self.last = Some(now);
        }
    }
}

impl Drop for Profile {
    fn drop(&mut self) {
        if self.on() {
            let total: f64 = self.stages.iter().map(|s| s.1).sum();
            let parts: Vec<String> = self
                .stages
                .iter()
                .map(|(n, t)| format!("{n} {t:.2}"))
                .collect();
            eprintln!("extract {total:.2} ms: {}", parts.join(", "));
        }
    }
}

/// Extract stars from a decoded image. See the module documentation.
pub fn extract(image: &GrayImage, opts: &ExtractOptions) -> StarField {
    let mut prof = Profile::new();
    let (w, h) = (image.width as usize, image.height as usize);
    let mut field = StarField {
        width: image.width,
        height: image.height,
        stars: vec![],
        epoch: image.epoch,
    };
    if w < 8 || h < 8 || image.pixels.len() < w * h {
        return field;
    }
    let b = bin_factor(w, h, opts);
    let plane = bin(image, b);
    if plane.w < 4 || plane.h < 4 {
        return field;
    }
    prof.mark("bin");
    let lsb = if opts.lsb > 0.0 {
        opts.lsb
    } else {
        detect_lattice(&image.pixels)
    };
    let Some(grid) = Grid::build(&plane, &image.pixels, w, b, lsb) else {
        return field;
    };
    prof.mark("background");
    let k = opts.threshold_sigma.max(0.5);
    let mut cands = find_peaks(&plane, &grid, k);
    cap_per_tile(&mut cands, &grid, opts.max_per_tile.max(1));
    prof.mark("peaks");

    let mut ctx = Ctx {
        img: &image.pixels,
        w,
        h,
        b,
        plane: &plane,
        grid: &grid,
        lsb,
        k,
        sat_level: cands.iter().map(|c| c.v).fold(f32::NEG_INFINITY, f32::max),
        local_background: opts.local_background,
        fwhm_frame: 0.0,
        trail: None,
    };
    // Saturation proxy: the highest native peak among the brightest binned peaks.
    let mut brightest: Vec<&Cand> = cands.iter().collect();
    let top = brightest.len().min(64);
    if top > 0 && top < brightest.len() {
        brightest.select_nth_unstable_by(top, |a, b| b.v.total_cmp(&a.v));
    }
    ctx.sat_level = brightest[..top]
        .iter()
        .filter_map(|c| ctx.native_peak(c))
        .map(|(x, y)| ctx.px(x, y))
        .fold(ctx.sat_level, f32::max);
    // Frame PSF from candidates that pass a provisional ridge test (so that a
    // nebula or a streak cannot define the PSF), then the final ridge test.
    ctx.fwhm_frame = 3.0;
    let psf = vet::frame_psf(&ctx, &cands);
    ctx.fwhm_frame = psf.fwhm;
    ctx.trail = psf.trail;
    prof.mark("psf");
    if prof.on() {
        eprintln!(
            "bin {b} cands {} fwhm_frame {:.2} trail {:?} sat {:.1}",
            cands.len(),
            ctx.fwhm_frame,
            ctx.trail,
            ctx.sat_level
        );
    }

    // Local contrast (ridge) test on the binned plane, before native measurement.
    let ridge: Vec<bool> = cands.par_iter().map(|c| ctx.is_ridge(c)).collect();
    prof.mark("ridge");
    let deblend = b >= 2 && !opts.compact;
    let mut dets: Vec<Det> = cands
        .par_iter()
        .zip(&ridge)
        .filter(|(_, r)| opts.keep_rejected || !**r)
        .filter_map(|(c, &r)| {
            let (px, py) = ctx.native_peak(c)?;
            let mut d = ctx.measure(px, py, opts.compact, f32::INFINITY, deblend)?;
            if r {
                d.flags |= flag::RIDGE;
            }
            Some(d)
        })
        .collect();
    let line_hints: Vec<[f32; 2]> = cands
        .iter()
        .zip(&ridge)
        .filter(|(_, r)| **r)
        .map(|(c, _)| ctx.to_native(c.x as f32, c.y as f32))
        .collect();

    // Secondary maxima found inside measurement windows become sources of their own.
    if deblend {
        let mut seen: std::collections::HashSet<(u32, u32)> =
            dets.iter().map(|d| (d.px, d.py)).collect();
        let extra: Vec<(u32, u32)> = dets
            .iter()
            .flat_map(|d| d.secondary.iter().flatten().copied())
            .filter(|p| seen.insert(*p))
            .collect();
        let more: Vec<Det> = extra
            .par_iter()
            .filter_map(|&(x, y)| ctx.measure(x as usize, y as usize, true, f32::INFINITY, false))
            .map(|mut d| {
                d.flags |= flag::BLENDED;
                d
            })
            .collect();
        dets.extend(more);
    }
    prof.mark("measure");
    merge_duplicates(&ctx, &mut dets);
    prof.mark("dedupe");
    merge_psf_lobes(&ctx, &mut dets);
    prof.mark("lobes");
    vet::vet(&ctx, &mut dets);
    if opts.remove_lines {
        lines::flag_lines(&ctx, &mut dets, &line_hints);
    }
    prof.mark("vet+lines");
    if !opts.keep_rejected {
        dets.retain(|d| d.flags & flag::REJECT == 0);
    }
    let trail_len = ctx.trail.map_or(0.0, |t| t.length);
    let mut stars: Vec<Star> = dets.iter().map(|d| d.to_star(trail_len)).collect();
    // equal fluxes (8-bit or clipped data) are ordered by position, so the order does not depend
    // on the order of the detections before (which last-bit differences of the maths library
    // can change: the browser build must send the same list as the native one)
    stars.par_sort_unstable_by(|a, b| {
        b.flux
            .total_cmp(&a.flux)
            .then(a.y.total_cmp(&b.y))
            .then(a.x.total_cmp(&b.x))
    });
    prof.mark("output");
    field.stars = stars;
    field
}

/// Bin factor for detection: native up to 6.5 MP, 2×2 up to 26 MP, 4×4 above.
pub fn bin_factor(w: usize, h: usize, opts: &ExtractOptions) -> usize {
    if opts.bin > 0 {
        return opts.bin.min(8);
    }
    if opts.compact {
        return 1;
    }
    match w * h {
        0..=6_500_000 => 1,
        6_500_001..=26_500_000 => 2,
        _ => 4,
    }
}

/// Detect a quantization lattice (integers, k/255, k/65535) from a sample.
///
/// A lattice is accepted only when the sample spans at least three of its
/// levels: values that all round to one or two levels (a dark float frame
/// normalized to [0, 1], data in physical units, an empty 8-bit sky) say
/// nothing about the step, and a false step of 1 would raise the noise floor
/// above every star.
pub(crate) fn detect_lattice(px: &[f32]) -> f32 {
    let step = (px.len() / 4096).max(1);
    let sample: Vec<f32> = px
        .iter()
        .step_by(step)
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if sample.len() < 64 {
        return 0.0;
    }
    for q in [1.0f32, 1.0 / 255.0, 1.0 / 65535.0] {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        let on_lattice = sample.iter().all(|&v| {
            let t = v / q;
            let r = t.round();
            lo = lo.min(r);
            hi = hi.max(r);
            (t - r).abs() < 2e-3 + 1e-6 * t.abs()
        });
        if on_lattice && hi - lo >= 2.0 {
            return q;
        }
    }
    0.0
}

/// A single-channel plane (the binned detection image, or the native one).
pub(crate) struct Plane<'a> {
    pub w: usize,
    pub h: usize,
    pub data: Cow<'a, [f32]>,
}

impl Plane<'_> {
    /// Bilinear sample; NaN outside the plane or next to masked pixels.
    pub fn bilinear(&self, x: f32, y: f32) -> f32 {
        if !(x >= 0.0 && y >= 0.0) {
            return f32::NAN;
        }
        let (x0, y0) = (x as usize, y as usize);
        if x0 + 1 >= self.w || y0 + 1 >= self.h {
            return if x0 < self.w && y0 < self.h && x0 as f32 == x && y0 as f32 == y {
                self.data[y0 * self.w + x0]
            } else {
                f32::NAN
            };
        }
        let (tx, ty) = (x - x0 as f32, y - y0 as f32);
        let i = y0 * self.w + x0;
        let d = &self.data;
        let top = d[i] + (d[i + 1] - d[i]) * tx;
        let bottom = d[i + self.w] + (d[i + self.w + 1] - d[i + self.w]) * tx;
        top + (bottom - top) * ty
    }
}

/// Mean of each finite b×b block (NaN when the whole block is masked).
pub(crate) fn bin(image: &GrayImage, b: usize) -> Plane<'_> {
    let (w, h) = (image.width as usize, image.height as usize);
    if b <= 1 {
        return Plane {
            w,
            h,
            data: Cow::Borrowed(&image.pixels[..w * h]),
        };
    }
    let (bw, bh) = (w / b, h / b);
    let mut out = vec![0f32; bw * bh];
    let px = &image.pixels;
    let rows_per_task = (16384 / bw.max(1)).max(1);
    out.par_chunks_mut(bw * rows_per_task)
        .enumerate()
        .for_each(|(task, chunk)| {
            for (r, orow) in chunk.chunks_mut(bw).enumerate() {
                let by = task * rows_per_task + r;
                let base = by * b * w;
                if b == 2 {
                    let r0 = &px[base..base + 2 * bw];
                    let r1 = &px[base + w..base + w + 2 * bw];
                    for (bx, o) in orow.iter_mut().enumerate() {
                        let s = r0[2 * bx] + r0[2 * bx + 1] + r1[2 * bx] + r1[2 * bx + 1];
                        *o = if s.is_finite() {
                            s * 0.25
                        } else {
                            block_mean(px, w, bx * 2, by * 2, 2)
                        };
                    }
                } else {
                    let inv = 1.0 / (b * b) as f32;
                    for (bx, o) in orow.iter_mut().enumerate() {
                        let mut s = 0.0;
                        for dy in 0..b {
                            let row = &px[base + dy * w + bx * b..base + dy * w + bx * b + b];
                            s += row.iter().sum::<f32>();
                        }
                        *o = if s.is_finite() {
                            s * inv
                        } else {
                            block_mean(px, w, bx * b, by * b, b)
                        };
                    }
                }
            }
        });
    Plane {
        w: bw,
        h: bh,
        data: Cow::Owned(out),
    }
}

fn block_mean(px: &[f32], w: usize, x0: usize, y0: usize, b: usize) -> f32 {
    let (mut s, mut n) = (0.0, 0);
    for y in y0..y0 + b {
        for &v in &px[y * w + x0..y * w + x0 + b] {
            if v.is_finite() {
                s += v;
                n += 1;
            }
        }
    }
    if n == 0 { f32::NAN } else { s / n as f32 }
}

/// Upper-tail standard normal quantile: P(Z > z) = p (Abramowitz–Stegun 26.2.23).
fn z_upper(p: f32) -> f32 {
    let p = (p as f64).clamp(1e-9, 1.0 - 1e-9);
    let (q, sign) = if p > 0.5 { (1.0 - p, -1.0) } else { (p, 1.0) };
    let t = (-2.0 * q.ln()).sqrt();
    let z = t
        - (2.515517 + 0.802853 * t + 0.010328 * t * t)
            / (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t);
    (sign * z) as f32
}

/// Order-preserving integer key of a finite f32 (so selections compare u32).
#[inline]
fn key(v: f32) -> u32 {
    let b = v.to_bits();
    if b >> 31 == 0 { b | 0x8000_0000 } else { !b }
}

#[inline]
fn unkey(k: u32) -> f32 {
    f32::from_bits(if k >> 31 == 1 { k & 0x7fff_ffff } else { !k })
}

/// Background level of a tile sample of keys (reorders it):
/// `(median, censored sigma if degenerate)`. A sample is degenerate (posterized
/// or clipped sky) when half or more of its values are identical, so that its
/// MAD is zero; its noise then comes from the upper tail of a censored Gaussian.
fn tile_level(keys: &mut [u32], lsb: f32) -> TileLevel {
    let n = keys.len();
    if n < 12 {
        return None;
    }
    let mid = n / 2;
    let mk = *keys.select_nth_unstable(mid).1;
    let m = unkey(mk);
    let (mut below, mut equal) = (0usize, 0usize);
    let (mut lo, mut at_lo) = (u32::MAX, 0usize);
    for &k in keys.iter() {
        below += (k < mk) as usize;
        equal += (k == mk) as usize;
        if k < lo {
            (lo, at_lo) = (k, 1);
        } else if k == lo {
            at_lo += 1;
        }
    }
    if 2 * equal < n {
        let mut bg = m;
        if lsb > 0.0 && equal > 1 {
            // Interpolated median of grouped (quantized) data.
            let t = ((n as f32 * 0.5 - below as f32) / equal as f32).clamp(0.0, 1.0);
            bg = m + lsb * (t - 0.5);
        }
        let floor = (4 * at_lo > n)
            .then(|| floor_censored_sigma(keys, lo, at_lo, lsb))
            .flatten();
        return Some((bg, None, floor.unwrap_or(0.0)));
    }
    let above = n - below - equal;
    let sigma = if above >= 8 {
        let p = above as f32 / n as f32;
        // The median of the values above the mode is the 1 - p/2 quantile.
        let mut upper: Vec<u32> = keys.iter().copied().filter(|&k| k > mk).collect();
        let vq = unkey(*upper.select_nth_unstable(above / 2).1);
        (vq - m) / (z_upper(p * 0.5) - z_upper(p)).max(0.3)
    } else {
        0.0
    };
    Some((m, Some(sigma.max(noise_floor(m, lsb))), 0.0))
}

/// Noise of a sky clipped from below (negative values written as 0 after a
/// pedestal subtraction): `at_lo` of the `n` samples sit on the floor `lo`.
/// Past a quarter of the samples the floor reaches within 0.67σ of the median
/// and the MAD collapses (a frame with half its pixels at 0 reads 2.5× too
/// quiet, and the 5σ threshold admits hundreds of noise peaks). The floor is
/// the p0 quantile of the uncensored Gaussian, the median of the values above
/// it the (1 + p0)/2 quantile; both are read on grouped data (the floor bin
/// ends half a step above the floor). Censoring only lowers the MAD, so the
/// caller keeps the larger of the two (a vignetted corner falling to the
/// floor inside one tile is no stationary Gaussian, and its residual MAD is
/// the larger). `None` when the floor is not attached to the sky (zero padding
/// of a registered frame: a gap above the floor).
fn floor_censored_sigma(keys: &mut [u32], lo: u32, at_lo: usize, lsb: f32) -> Option<f32> {
    let n = keys.len();
    let above = n - at_lo;
    if above < 8 {
        return None;
    }
    let c = unkey(lo);
    // Order statistics of the values above the floor (all keys > lo).
    let i_q = at_lo + above / 2;
    let kq = *keys.select_nth_unstable(i_q).1;
    let v1 = unkey(*keys[..i_q].iter().filter(|&&k| k > lo).min().unwrap_or(&kq));
    let mut vq = unkey(kq);
    if lsb > 0.0 {
        let (mut below, mut equal) = (0usize, 0usize);
        for &k in keys.iter() {
            below += (k < kq) as usize;
            equal += (k == kq) as usize;
        }
        let t = ((i_q as f32 + 0.5 - below as f32) / equal as f32).clamp(0.0, 1.0);
        vq += lsb * (t - 0.5);
    }
    if v1 - c > (1.5 * lsb).max(0.25 * (vq - c)) {
        return None;
    }
    // P(X > cb) = 1 - p0 and P(X > vq) = 1 - q for the uncensored Gaussian.
    let (p0, q) = (at_lo as f32 / n as f32, (i_q as f32 + 0.5) / n as f32);
    let cb = c + 0.5 * lsb;
    let den = z_upper(1.0 - q) - z_upper(1.0 - p0);
    (den > 0.3 && vq > cb).then(|| (vq - cb) / den)
}

/// Tile background level, the censored noise estimate of degenerate
/// (posterized) samples, and the lower bound of the noise of floor-censored
/// samples (0 otherwise); `None` for masked tiles.
type TileLevel = Option<(f32, Option<f32>, f32)>;

/// A [`TileLevel`] after masked tiles are filled, with the filled flag.
type TileStats = (f32, Option<f32>, f32, bool);

/// Drop the level of tiles dominated by a clipped disc, so that they are filled
/// from their neighbours like masked tiles. A saturated star wider than about
/// half a tile otherwise *is* the median of its tile: the background model
/// rises to the plateau and the brightest star of the frame disappears.
///
/// A tile qualifies when its median lies within 3 % (of the height above the
/// sky) of its own maximum — the upper half of its samples is a flat top — and
/// that maximum is among the brightest values of the frame. Broad clipped
/// areas (a burnt-out nebula, the Moon: more than max(4, 3 %) of the tiles) are
/// left alone; they are not stars.
fn mask_clipped_tiles(levels: &mut [TileLevel], top: &[f32]) {
    let mut sky: Vec<f32> = levels.iter().flatten().map(|l| l.0).collect();
    if sky.is_empty() {
        return;
    }
    let mid = sky.len() / 2;
    let sky = *sky.select_nth_unstable_by(mid, f32::total_cmp).1;
    let peak = top
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if peak <= sky {
        return;
    }
    let clipped: Vec<usize> = levels
        .iter()
        .zip(top)
        .enumerate()
        .filter(|(_, (l, mx))| {
            l.is_some_and(|(m, _, _)| {
                **mx - sky >= 0.5 * (peak - sky) && **mx - m <= 0.03 * (**mx - sky)
            })
        })
        .map(|(t, _)| t)
        .collect();
    let valid = levels.iter().flatten().count();
    if clipped.is_empty()
        || clipped.len() > (levels.len() * 3 / 100).max(4)
        || 2 * clipped.len() >= valid
    {
        return;
    }
    for t in clipped {
        levels[t] = None;
    }
}

/// Lower bound of the noise: half a quantization step, f32 resolution of the
/// level, and (scale-free) the smallest normal f32, so that frames in physical
/// units (1e-18 erg s⁻¹ cm⁻² Å⁻¹ and below) keep their stars.
pub(crate) fn noise_floor(level: f32, lsb: f32) -> f32 {
    (0.5 * lsb).max(level.abs() * 1e-6).max(f32::MIN_POSITIVE)
}

/// 1.4826 × median |r| of residuals given as keys of |r| (reorders them).
fn mad0(keys: &mut [u32]) -> Option<f32> {
    let n = keys.len();
    (n >= 12).then(|| 1.4826 * unkey(*keys.select_nth_unstable(n / 2).1))
}

/// Euclidean length (plain sqrt; `hypot` guards against overflow we cannot hit).
#[inline]
pub(crate) fn dist(dx: f64, dy: f64) -> f64 {
    (dx * dx + dy * dy).sqrt()
}

/// Visit the sparse checkerboard sample positions of a tile `(x0, y0, x1, y1)`.
#[inline]
fn for_tile_samples(
    (x0, y0, x1, y1): (usize, usize, usize, usize),
    row_step: usize,
    mut f: impl FnMut(usize, usize),
) {
    for (r, y) in (y0..y1).step_by(row_step).enumerate() {
        for x in (x0 + (r & 1)..x1).step_by(2) {
            f(x, y);
        }
    }
}

/// One axis of the background mesh: `n` equal (fractional) tiles over `len` px.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Axis {
    pub n: usize,
    /// Tile size in binned pixels.
    pub s: f32,
    len: usize,
}

impl Axis {
    fn new(len: usize, tile: usize) -> Self {
        let n = len.div_ceil(tile).max(1);
        Self {
            n,
            s: len as f32 / n as f32,
            len,
        }
    }

    /// Pixel range [a, b) of tile `t`.
    pub fn range(&self, t: usize) -> (usize, usize) {
        let edge = |k: usize| ((k as f32 * self.s).round() as usize).min(self.len);
        (
            edge(t),
            if t + 1 == self.n {
                self.len
            } else {
                edge(t + 1)
            },
        )
    }

    #[inline]
    fn index(&self, c: usize) -> usize {
        ((c as f32 / self.s) as usize).min(self.n - 1)
    }

    /// Left node and fractional position between node centres (outside [0, 1]
    /// in the half tiles at the frame edges).
    #[inline]
    fn locate(&self, c: f32) -> (usize, f32) {
        if self.n == 1 {
            return (0, 0.0);
        }
        let f = (c + 0.5) / self.s - 0.5;
        let i = (f.floor().max(0.0) as usize).min(self.n - 2);
        (i, f - i as f32)
    }

    /// Linear interpolation, clamped at the outer node centres.
    #[inline]
    fn linear(&self, c: f32) -> (usize, usize, f32) {
        let (i, t) = self.locate(c);
        if self.n == 1 {
            return (0, 0, 0.0);
        }
        (i, i + 1, t.clamp(0.0, 1.0))
    }

    /// Catmull-Rom weights; missing outer nodes are extrapolated linearly
    /// (p₋₁ = 2p₀ − p₁), so gradients continue to the frame edges.
    #[inline]
    fn cubic(&self, c: f32) -> [(u32, f32); 6] {
        let mut out = [(0u32, 0f32); 6];
        if self.n == 1 {
            out[0] = (0, 1.0);
            return out;
        }
        let (i, t) = self.locate(c);
        let (t2, t3) = (t * t, t * t * t);
        let w = [
            0.5 * (-t3 + 2.0 * t2 - t),
            0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
            0.5 * (-3.0 * t3 + 4.0 * t2 + t),
            0.5 * (t3 - t2),
        ];
        let n = self.n;
        out[1] = (i as u32, w[1]);
        out[2] = (i as u32 + 1, w[2]);
        if i == 0 {
            out[0] = (0, 2.0 * w[0]);
            out[4] = (1, -w[0]);
        } else {
            out[0] = (i as u32 - 1, w[0]);
        }
        if i + 2 >= n {
            out[3] = (n as u32 - 1, 2.0 * w[3]);
            out[5] = (n as u32 - 2, -w[3]);
        } else {
            out[3] = (i as u32 + 2, w[3]);
        }
        out
    }
}

/// Background model: per-tile median (curvature-corrected) interpolated bicubically
/// between tile centres; per-tile noise interpolated bilinearly.
pub(crate) struct Grid {
    pub ax: Axis,
    pub ay: Axis,
    pub nx: usize,
    pub ny: usize,
    pub b: usize,
    pub bg: Vec<f32>,
    /// Noise of the binned plane.
    pub sig_b: Vec<f32>,
    /// Noise of single native pixels.
    pub sig_n: Vec<f32>,
    /// Median native noise of the frame.
    pub typical_sig_n: f32,
    /// Per binned column: cubic and linear interpolation weights along x.
    xcub: Vec<[(u32, f32); 6]>,
    xlin: Vec<(u32, u32, f32)>,
}

impl Grid {
    /// Two passes over the tiles: medians (the background model), then the noise
    /// as the MAD of residuals from the interpolated model, so that gradients
    /// inside a tile (vignetting, sky glow, galaxy halos) do not inflate it; on a
    /// sky clipped at a floor, at least the censored estimate of the tile.
    /// Tiles are sampled on a checkerboard (≤ 1024 samples); the native-pixel
    /// noise is the binned noise times a frame-wide ratio measured on a subset
    /// of tiles.
    pub(crate) fn build(
        plane: &Plane,
        native: &[f32],
        w: usize,
        b: usize,
        lsb: f32,
    ) -> Option<Self> {
        let tile = (TILE_NATIVE / b).max(16);
        let (ax, ay) = (Axis::new(plane.w, tile), Axis::new(plane.h, tile));
        let (nx, ny) = (ax.n, ay.n);
        // Every second row, alternate columns (a quarter of the pixels): 1024
        // samples per 64×64 native tile, 256 per 32×32 binned tile.
        let row_step = 2;
        // A binned pixel averages b² samples: its quantization step is lsb/b².
        let lsb_b = lsb / (b * b) as f32;
        let bounds = |t: usize| {
            let (x0, x1) = ax.range(t % nx);
            let (y0, y1) = ay.range(t / nx);
            (x0, y0, x1, y1)
        };
        // (level, censored sigma) and the largest sample of every tile.
        let (mut first, top): (Vec<TileLevel>, Vec<f32>) = (0..nx * ny)
            .into_par_iter()
            .map_init(Vec::new, |keys: &mut Vec<u32>, t| {
                keys.clear();
                for_tile_samples(bounds(t), row_step, |x, y| {
                    let v = plane.data[y * plane.w + x];
                    if v.is_finite() {
                        keys.push(key(v));
                    }
                });
                let top = keys.iter().max().map_or(f32::NAN, |&k| unkey(k));
                (tile_level(keys, lsb_b), top)
            })
            .unzip();
        if first.iter().all(Option::is_none) {
            return None;
        }
        mask_clipped_tiles(&mut first, &top);
        // Fill masked tiles from their neighbours (they are never re-estimated).
        let mut stats: Vec<Option<TileStats>> = first
            .into_iter()
            .map(|s| s.map(|(m, d, c)| (m, d, c, false)))
            .collect();
        while stats.iter().any(Option::is_none) {
            let prev = stats.clone();
            for t in 0..nx * ny {
                if prev[t].is_some() {
                    continue;
                }
                let (tx, ty) = ((t % nx) as isize, (t / nx) as isize);
                let (mut s, mut n) = (0.0, 0.0);
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let (x, y) = (tx + dx, ty + dy);
                    if x >= 0
                        && y >= 0
                        && (x as usize) < nx
                        && (y as usize) < ny
                        && let Some(v) = prev[y as usize * nx + x as usize]
                    {
                        s += v.0;
                        n += 1.0;
                    }
                }
                if n > 0.0 {
                    stats[t] = Some((s / n, None, 0.0, true));
                }
            }
        }
        let stats: Vec<TileStats> = stats.into_iter().map(Option::unwrap).collect();
        // A tile median sits below the centre value of a concave background
        // (galaxy halo, vignetting) and above it on a convex one, by about
        // (Lx + Ly)/24 where L is the mesh Laplacian. Correcting it lets the
        // bicubic interpolation (exact for quadratics) follow curved skies.
        let medians: Vec<f32> = stats.iter().map(|s| s.0).collect();
        let bg: Vec<f32> = (0..nx * ny)
            .map(|t| {
                let (tx, ty) = (t % nx, t / nx);
                let lap = |a: Option<usize>, b: Option<usize>| match (a, b) {
                    (Some(a), Some(b)) => medians[a] + medians[b] - 2.0 * medians[t],
                    _ => 0.0,
                };
                let lx = lap(
                    tx.checked_sub(1).map(|x| ty * nx + x),
                    (tx + 1 < nx).then(|| ty * nx + tx + 1),
                );
                let ly = lap(
                    ty.checked_sub(1).map(|y| y * nx + tx),
                    (ty + 1 < ny).then(|| (ty + 1) * nx + tx),
                );
                medians[t] - (lx + ly) / 24.0
            })
            .collect();
        let mut grid = Self {
            ax,
            ay,
            nx,
            ny,
            b,
            bg,
            sig_b: vec![0.0; nx * ny],
            sig_n: vec![0.0; nx * ny],
            typical_sig_n: 0.0,
            xcub: (0..plane.w).map(|x| ax.cubic(x as f32)).collect(),
            xlin: (0..plane.w)
                .map(|x| {
                    let (i, j, t) = ax.linear(x as f32);
                    (i as u32, j as u32, t)
                })
                .collect(),
        };
        // Residual MAD of tile t on the binned plane or on native pixels.
        let residual_sigma =
            |t: usize, native_px: bool, (keys, colv): &mut (Vec<u32>, Vec<f32>)| -> Option<f32> {
                let tx = t % nx;
                let (c0, c1) = (tx.saturating_sub(2), (tx + 3).min(nx));
                colv.resize(nx, 0.0);
                keys.clear();
                let mut last_y = usize::MAX;
                for_tile_samples(bounds(t), row_step, |x, y| {
                    if y != last_y {
                        grid.row_cubic_range(y as f32, c0, c1, colv);
                        last_y = y;
                    }
                    let v = if native_px {
                        native[y * b * w + x * b]
                    } else {
                        plane.data[y * plane.w + x]
                    };
                    if v.is_finite() {
                        keys.push(key((v - grid.col_cubic(x, colv)).abs()));
                    }
                });
                mad0(keys)
            };
        let fallback = |t: usize| stats[t].1.unwrap_or(f32::NAN);
        let sig_b: Vec<f32> = (0..nx * ny)
            .into_par_iter()
            .map_init(Default::default, |scratch, t| {
                if stats[t].1.is_some() || stats[t].3 {
                    return fallback(t);
                }
                residual_sigma(t, false, scratch).map_or(f32::NAN, |s| {
                    s.max(stats[t].2).max(noise_floor(stats[t].0, lsb_b))
                })
            })
            .collect();
        // Native/binned noise ratio from up to 64 well-behaved tiles.
        let ratio = if b == 1 {
            1.0
        } else {
            let step = (nx * ny / 64).max(1);
            let mut r: Vec<f32> = (0..nx * ny)
                .step_by(step)
                .collect::<Vec<_>>()
                .into_par_iter()
                .filter(|&t| {
                    stats[t].1.is_none() && stats[t].2 == 0.0 && !stats[t].3 && sig_b[t].is_finite()
                })
                .filter_map(|t| {
                    residual_sigma(t, true, &mut Default::default()).map(|s| s / sig_b[t])
                })
                .filter(|r| r.is_finite() && *r > 0.0)
                .collect();
            if r.is_empty() {
                b as f32
            } else {
                let mid = r.len() / 2;
                *r.select_nth_unstable_by(mid, f32::total_cmp).1
            }
        };
        // Tiles without an estimate (masked, too few samples, wide meshes)
        // take the median noise of the frame.
        let mut finite: Vec<f32> = sig_b.iter().copied().filter(|s| s.is_finite()).collect();
        let typical = if finite.is_empty() {
            noise_floor(0.0, lsb_b)
        } else {
            let mid = finite.len() / 2;
            *finite.select_nth_unstable_by(mid, f32::total_cmp).1
        };
        grid.sig_b = sig_b
            .iter()
            .map(|&s| if s.is_finite() { s } else { typical })
            .collect();
        grid.sig_n = grid
            .sig_b
            .iter()
            .zip(&stats)
            .map(|(&s, st)| (s * ratio).max(noise_floor(st.0, lsb)))
            .collect();
        grid.typical_sig_n = typical * ratio;
        Some(grid)
    }

    /// Background along y at binned row `yb` for tile columns `c0..c1`.
    pub fn row_cubic_range(&self, yb: f32, c0: usize, c1: usize, out: &mut [f32]) {
        let wy = self.ay.cubic(yb);
        for (t, o) in out.iter_mut().enumerate().take(c1).skip(c0) {
            *o = wy
                .iter()
                .map(|&(r, w)| w * self.bg[r as usize * self.nx + t])
                .sum();
        }
    }

    /// Background along y at binned row `yb` for every tile column.
    pub fn row_cubic(&self, yb: f32, out: &mut [f32]) {
        let wy = self.ay.cubic(yb);
        for (t, o) in out.iter_mut().enumerate().take(self.nx) {
            *o = wy
                .iter()
                .map(|&(r, w)| w * self.bg[r as usize * self.nx + t])
                .sum();
        }
    }

    /// Finish the bicubic background along x from [`Grid::row_cubic`] values.
    #[inline]
    pub fn col_cubic(&self, xb: usize, colv: &[f32]) -> f32 {
        self.xcub[xb]
            .iter()
            .map(|&(c, w)| w * colv[c as usize])
            .sum()
    }

    /// Binned-plane noise along y at row `yb` for every tile column.
    fn row_sigma(&self, yb: f32, out: &mut [f32]) {
        let (j0, j1, ty) = self.ay.linear(yb);
        for (t, o) in out.iter_mut().enumerate().take(self.nx) {
            let (a, c) = (self.sig_b[j0 * self.nx + t], self.sig_b[j1 * self.nx + t]);
            *o = a + (c - a) * ty;
        }
    }

    #[inline]
    fn col_linear(&self, xb: usize, colv: &[f32]) -> f32 {
        let (i0, i1, tx) = self.xlin[xb];
        let (a, c) = (colv[i0 as usize], colv[i1 as usize]);
        a + (c - a) * tx
    }

    /// (background, binned noise, native noise) at binned coordinates.
    pub fn at(&self, xb: f32, yb: f32) -> (f32, f32, f32) {
        let (wx, wy) = (self.ax.cubic(xb), self.ay.cubic(yb));
        let mut bg = 0.0;
        for &(r, a) in &wy {
            for &(c, w) in &wx {
                bg += a * w * self.bg[r as usize * self.nx + c as usize];
            }
        }
        let (i0, i1, tx) = self.ax.linear(xb);
        let (j0, j1, ty) = self.ay.linear(yb);
        let lerp = |v: &[f32]| {
            let top = v[j0 * self.nx + i0] + (v[j0 * self.nx + i1] - v[j0 * self.nx + i0]) * tx;
            let bot = v[j1 * self.nx + i0] + (v[j1 * self.nx + i1] - v[j1 * self.nx + i0]) * tx;
            top + (bot - top) * ty
        };
        (bg, lerp(&self.sig_b), lerp(&self.sig_n))
    }

    /// Same as [`Grid::at`] for native pixel coordinates.
    pub fn at_native(&self, x: f32, y: f32) -> (f32, f32, f32) {
        let b = self.b as f32;
        self.at((x + 0.5) / b - 0.5, (y + 0.5) / b - 0.5)
    }

    pub fn tile_index(&self, xb: usize, yb: usize) -> usize {
        self.ay.index(yb) * self.nx + self.ax.index(xb)
    }
}

/// A peak on the binned plane.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cand {
    pub x: u32,
    pub y: u32,
    /// Binned value at the peak.
    pub v: f32,
    pub bg: f32,
    pub sig: f32,
}

/// Peaks above `bg + k·σ`: strict 3×3 maxima (ties broken by index) with at least
/// three of the nine pixels above `bg + 2σ`.
fn find_peaks(plane: &Plane, grid: &Grid, k: f32) -> Vec<Cand> {
    let (w, h) = (plane.w, plane.h);
    let rows_per_task = (32768 / w.max(1)).clamp(1, 64);
    let tasks = (h - 2).div_ceil(rows_per_task);
    let data = &plane.data;
    (0..tasks)
        .into_par_iter()
        .flat_map_iter(|task| {
            let mut out = vec![];
            let mut bg_row = vec![0f32; grid.nx];
            let mut sg_row = vec![0f32; grid.nx];
            let mut bound = vec![0f32; grid.nx.max(2) - 1];
            let y_start = 1 + task * rows_per_task;
            let y_end = (y_start + rows_per_task).min(h - 1);
            for y in y_start..y_end {
                grid.row_cubic(y as f32, &mut bg_row);
                grid.row_sigma(y as f32, &mut sg_row);
                // Lower bound of the threshold in each interval between tile
                // centres: Catmull-Rom (with linearly extrapolated phantom nodes
                // and half-tile extrapolation at the edges) undershoots its four
                // nodes by less than half their range.
                let nx = grid.nx;
                let node = |j: isize| -> f32 {
                    if nx == 1 {
                        bg_row[0]
                    } else if j < 0 {
                        2.0 * bg_row[0] - bg_row[1]
                    } else if j as usize >= nx {
                        2.0 * bg_row[nx - 1] - bg_row[nx - 2]
                    } else {
                        bg_row[j as usize]
                    }
                };
                for (i, b) in bound.iter_mut().enumerate() {
                    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                    for j in i as isize - 1..=i as isize + 2 {
                        let v = node(j);
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                    let sg = sg_row[i].min(sg_row[(i + 1).min(nx - 1)]);
                    *b = lo - 0.5 * (hi - lo) + k * sg;
                }
                let up = &data[(y - 1) * w..y * w];
                let row = &data[y * w..(y + 1) * w];
                let down = &data[(y + 1) * w..(y + 2) * w];
                // Branch-free scan in blocks of 16 (vectorized compare); only
                // blocks with a pixel above the row's lower threshold bound are
                // examined pixel by pixel.
                let mut x0 = 1;
                while x0 < w - 1 {
                    let x1 = (x0 + 16).min(w - 1);
                    let (ia, ib) = (grid.xlin[x0].0 as usize, grid.xlin[x1 - 1].0 as usize);
                    let thr_min =
                        bound[ia.min(bound.len() - 1)].min(bound[ib.min(bound.len() - 1)]);
                    let any = row[x0..x1].iter().fold(false, |a, &v| a | (v > thr_min));
                    let block = x0..x1;
                    x0 = x1;
                    if !any {
                        continue;
                    }
                    for x in block {
                        let v = row[x];
                        if v.is_nan() || v <= thr_min {
                            continue;
                        }
                        let bg = grid.col_cubic(x, &bg_row);
                        let sg = grid.col_linear(x, &sg_row);
                        if v <= bg + k * sg {
                            continue;
                        }
                        // Earlier pixels win ties so that a flat top yields one peak.
                        if up[x - 1] >= v
                            || up[x] >= v
                            || up[x + 1] >= v
                            || row[x - 1] >= v
                            || row[x + 1] > v
                            || down[x - 1] > v
                            || down[x] > v
                            || down[x + 1] > v
                        {
                            continue;
                        }
                        let s = bg + 2.0 * sg;
                        let support = [
                            up[x - 1],
                            up[x],
                            up[x + 1],
                            row[x - 1],
                            v,
                            row[x + 1],
                            down[x - 1],
                            down[x],
                            down[x + 1],
                        ]
                        .iter()
                        .filter(|&&q| q > s)
                        .count();
                        if support >= 3 {
                            out.push(Cand {
                                x: x as u32,
                                y: y as u32,
                                v,
                                bg,
                                sig: sg,
                            });
                        }
                    }
                }
            }
            out
        })
        .collect()
}

/// Keep only the `cap` brightest peaks of every background tile.
fn cap_per_tile(cands: &mut Vec<Cand>, grid: &Grid, cap: usize) {
    let mut count = vec![0u32; grid.nx * grid.ny];
    for c in cands.iter() {
        count[grid.tile_index(c.x as usize, c.y as usize)] += 1;
    }
    if count.iter().all(|&n| n as usize <= cap) {
        return;
    }
    let mut cutoff = vec![f32::NEG_INFINITY; count.len()];
    for (t, &n) in count.iter().enumerate() {
        if n as usize > cap {
            let mut vals: Vec<f32> = cands
                .iter()
                .filter(|c| grid.tile_index(c.x as usize, c.y as usize) == t)
                .map(|c| c.v - c.bg)
                .collect();
            let i = vals.len() - cap;
            cutoff[t] = *vals.select_nth_unstable_by(i, |a, b| a.total_cmp(b)).1;
        }
    }
    cands.retain(|c| c.v - c.bg >= cutoff[grid.tile_index(c.x as usize, c.y as usize)]);
}

/// Trailed-frame model: common direction and length of star trails.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Trail {
    /// Position angle in radians, [0, π).
    pub pa: f32,
    /// Trail length in native pixels.
    pub length: f32,
}

/// Result of the local-contrast probe around a candidate (binned units).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Contrast {
    /// Peak above the background model.
    pub height: f32,
    /// Peak above its surroundings (second highest axis).
    pub contrast: f32,
    /// Lowest axis level above the background model.
    pub floor: f32,
}

/// Shared read-only state of one extraction.
pub(crate) struct Ctx<'a> {
    pub img: &'a [f32],
    pub w: usize,
    pub h: usize,
    pub b: usize,
    pub plane: &'a Plane<'a>,
    pub grid: &'a Grid,
    pub lsb: f32,
    pub k: f32,
    /// Highest peak value of the frame (proxy for the saturation level).
    pub sat_level: f32,
    pub local_background: bool,
    /// Median FWHM of the frame in native pixels (0 when unknown).
    pub fwhm_frame: f32,
    pub trail: Option<Trail>,
}

/// One measured source (native pixel coordinates).
#[derive(Clone, Debug, Default)]
pub(crate) struct Det {
    pub x: f64,
    pub y: f64,
    /// Native peak pixel.
    pub px: u32,
    pub py: u32,
    /// Background-subtracted peak value.
    pub h: f32,
    pub bg: f32,
    /// Local background plane used for the measurement.
    pub plane: Bg,
    pub sig_n: f32,
    pub flux: f64,
    pub snr: f64,
    pub fwhm: f64,
    pub elong: f64,
    /// Position angle in degrees, [0, 180).
    pub pa: f64,
    /// Largest 4-neighbour of the peak relative to the peak (sharpness).
    pub ratio4: f32,
    /// Plateau radius for clipped discs (0 otherwise).
    pub core: f32,
    pub flags: u32,
    /// Separate maxima found inside the window (deblending).
    pub secondary: [Option<(u32, u32)>; 2],
}

impl Det {
    fn to_star(&self, trail_len: f32) -> Star {
        let mut sigma = (self.fwhm / (2.3548 * self.snr.max(1e-3))).max(0.05);
        if self.flags & flag::SATURATED != 0 {
            sigma = sigma.max(0.3);
        }
        if self.flags & flag::BLENDED != 0 {
            sigma = sigma.max(0.15);
        }
        if self.flags & flag::NEAR_EDGE != 0 {
            sigma = sigma.max(0.1);
        }
        if self.flags & flag::TRAILED != 0 {
            sigma = sigma.max(0.1 * trail_len as f64);
        }
        Star {
            x: self.x,
            y: self.y,
            flux: self.flux,
            sigma: sigma.min(2.0),
            snr: self.snr,
            fwhm: self.fwhm,
            elongation: self.elong,
            saturated: self.flags & flag::SATURATED != 0,
            peak: self.h as f64,
            pa: self.pa,
            flags: self.flags,
        }
    }
}

/// Local background plane around a source: `c + gx·(x - x0) + gy·(y - y0)`.
/// Steep gradients (galaxy halos, nebulae) would otherwise pull centroids uphill.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Bg {
    pub c: f32,
    pub gx: f32,
    pub gy: f32,
    pub x0: f32,
    pub y0: f32,
}

impl Bg {
    #[inline]
    pub fn at(&self, x: f32, y: f32) -> f32 {
        self.c + self.gx * (x - self.x0) + self.gy * (y - self.y0)
    }
    /// Same plane with a different level at its origin.
    pub fn with_level(self, c: f32) -> Self {
        Self { c, ..self }
    }
}

/// Second moments of `v - bg` above a threshold in a square window.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Moments {
    pub sum: f64,
    pub x: f64,
    pub y: f64,
    pub xx: f64,
    pub yy: f64,
    pub xy: f64,
    /// Window touched the frame edge or masked pixels.
    pub clipped: bool,
}

/// Second moment of a Gaussian thresholded at fraction `f` of its peak
/// (weights `(I - f·H)+`), relative to the untruncated one.
fn truncation_gain(f: f64) -> f64 {
    if f <= 1e-4 {
        return 1.0;
    }
    let f = f.min(0.8);
    let u = -f.ln();
    let num = 1.0 - f * (1.0 + u) - 0.5 * f * u * u;
    let den = 1.0 - f - f * u;
    (num / den).clamp(0.1, 1.0)
}

impl Moments {
    /// (FWHM, elongation, position angle in degrees). `f` is the moment
    /// threshold as a fraction of the peak; the FWHM is corrected for the
    /// truncation as if the profile were Gaussian.
    pub fn shape(&self, f: f64) -> (f64, f64, f64) {
        let disc = ((self.xx - self.yy).powi(2) + 4.0 * self.xy * self.xy).sqrt();
        let minor = ((self.xx + self.yy - disc) / 2.0).max(0.02);
        let major = ((self.xx + self.yy + disc) / 2.0).max(minor);
        let fwhm = 2.3548 * ((major + minor) / 2.0 / truncation_gain(f)).sqrt();
        let pa = (0.5 * (2.0 * self.xy).atan2(self.xx - self.yy))
            .to_degrees()
            .rem_euclid(180.0);
        (fwhm, (major / minor).sqrt(), pa)
    }
    pub fn axes(&self) -> (f64, f64) {
        let disc = ((self.xx - self.yy).powi(2) + 4.0 * self.xy * self.xy).sqrt();
        let minor = ((self.xx + self.yy - disc) / 2.0).max(0.0);
        let major = ((self.xx + self.yy + disc) / 2.0).max(minor);
        (major, minor)
    }
}

/// Background-subtracted sum over the pixel centres inside a circle of radius `r` around
/// (cx, cy) of a `w` x `h` plane: (flux, pixels, touched the edge or masked pixels). Sums,
/// products and one square root per row only, so native and WebAssembly builds agree bit for
/// bit. The detector's aperture, also used by forced photometry ([`crate::measure`]).
pub(crate) fn aperture(
    img: &[f32],
    w: usize,
    h: usize,
    cx: f64,
    cy: f64,
    r: f64,
    bg: Bg,
) -> (f64, usize, bool) {
    let ri = r.ceil() as isize;
    let iy = cy.round() as isize;
    let (mut flux, mut n, mut clipped) = (0.0f64, 0usize, false);
    let r2 = r * r;
    for y in iy - ri..=iy + ri {
        let dy2 = (y as f64 - cy).powi(2);
        if dy2 > r2 {
            continue;
        }
        // Pixel centres inside the circle on this row: one contiguous span.
        let half = (r2 - dy2).sqrt();
        let (xa, xb) = ((cx - half).ceil() as isize, (cx + half).floor() as isize);
        if y < 0 || y >= h as isize || xa < 0 || xb >= w as isize {
            clipped = true;
        }
        if y < 0 || y >= h as isize {
            continue;
        }
        let (xa, xb) = (xa.max(0) as usize, xb.min(w as isize - 1));
        if xb < xa as isize {
            continue;
        }
        let row = &img[y as usize * w..(y as usize + 1) * w];
        let row_bg = bg.at(0.0, y as f32);
        let (mut sum, mut sx, mut cnt) = (0.0f32, 0.0f32, 0usize);
        for (k, &v) in row[xa..=xb as usize].iter().enumerate() {
            if v.is_finite() {
                sum += v;
                sx += k as f32;
                cnt += 1;
            } else {
                clipped = true;
            }
        }
        // Σ (v − row_bg − gx·x) over the finite pixels of the span.
        let gx_sum = bg.gx * (sx + (cnt * xa) as f32);
        flux += (sum - row_bg * cnt as f32 - gx_sum) as f64;
        n += cnt;
    }
    (flux, n, clipped)
}

impl Ctx<'_> {
    #[inline]
    pub fn px(&self, x: usize, y: usize) -> f32 {
        self.img[y * self.w + x]
    }

    /// Binned coordinates → native pixel coordinates.
    pub fn to_native(&self, xb: f32, yb: f32) -> [f32; 2] {
        let b = self.b as f32;
        [(xb + 0.5) * b - 0.5, (yb + 0.5) * b - 0.5]
    }

    /// Local background plane at a native pixel, from the bicubic model.
    pub fn plane_bg(&self, px: usize, py: usize) -> (Bg, f32) {
        let (x, y) = (px as f32, py as f32);
        let d = (self.grid.ax.s * self.b as f32 * 0.25).max(2.0);
        let (c, _, sig_n) = self.grid.at_native(x, y);
        let gx = (self.grid.at_native(x + d, y).0 - self.grid.at_native(x - d, y).0) / (2.0 * d);
        let gy = (self.grid.at_native(x, y + d).0 - self.grid.at_native(x, y - d).0) / (2.0 * d);
        (
            Bg {
                c,
                gx,
                gy,
                x0: x,
                y0: y,
            },
            sig_n,
        )
    }

    /// Brightest finite native pixel in the binned block (plus a 1 px margin).
    pub fn native_peak(&self, c: &Cand) -> Option<(usize, usize)> {
        let b = self.b;
        let (x, y) = (c.x as usize, c.y as usize);
        if b == 1 {
            return Some((x, y));
        }
        let (x0, y0) = ((x * b).saturating_sub(1), (y * b).saturating_sub(1));
        let (x1, y1) = ((x * b + b).min(self.w - 1), (y * b + b).min(self.h - 1));
        let mut best = (f32::NEG_INFINITY, 0, 0);
        for yy in y0..=y1 {
            for xx in x0..=x1 {
                let v = self.px(xx, yy);
                if v > best.0 {
                    best = (v, xx, yy);
                }
            }
        }
        (best.0 > f32::NEG_INFINITY).then_some((best.1, best.2))
    }

    /// Local-contrast test on the binned plane (see [`Ctx::contrast`]): ridges
    /// (nebular filaments, streaks) and bumps on structured backgrounds fail,
    /// stars (also crowded ones) and broad bright discs pass.
    pub fn is_ridge(&self, c: &Cand) -> bool {
        let Some(l) = self.contrast(c) else {
            return false;
        };
        if l.contrast >= 0.6 * self.k * c.sig && l.contrast >= 0.3 * l.height {
            return false;
        }
        // Broad, bright disc (clipped star, compact galaxy core): not a ridge.
        !(l.floor >= 0.5 * l.height && l.height >= 30.0 * c.sig)
    }

    /// Strict version for the PSF sample: compact, clearly isolated peaks only.
    pub fn is_compact_star(&self, c: &Cand) -> bool {
        self.contrast(c)
            .is_some_and(|l| l.contrast >= self.k * c.sig && l.contrast >= 0.6 * l.height)
    }

    /// How far a peak stands out of its surroundings. Sixteen axes (11.25°
    /// apart) are probed at 1.5 FWHM; along each ray the lowest of three samples
    /// (r, 1.5 r, 2 r) is kept, so that the saddle before a neighbouring star
    /// counts while a ridge stays high; each axis keeps its fainter side, and the
    /// second highest axis sets the level (a ridge at least a star wide raises
    /// two adjacent axes, a chance neighbour pair only one).
    fn contrast(&self, c: &Cand) -> Option<Contrast> {
        let hgt = c.v - c.bg;
        let b = self.b as f32;
        let ff = self.fwhm_frame.max(2.0) / b;
        let d = (1.5 * ff).max(1.5);
        let (theta0, half) = match self.trail {
            Some(t) => (t.pa, 0.5 * t.length / b),
            None => (0.0, 0.0),
        };
        let (x, y) = (c.x as f32, c.y as f32);
        let probe = |co: f32, s: f32, r: f32| {
            let mut low = f32::INFINITY;
            for t in [1.0, 1.5, 2.0] {
                let v = self.plane.bilinear(x + t * r * co, y + t * r * s);
                if v.is_finite() {
                    low = low.min(v);
                } else if t == 1.0 {
                    return f32::NAN;
                }
            }
            low
        };
        let (mut amax, mut amax2, mut amin) = (f32::NEG_INFINITY, f32::NEG_INFINITY, f32::INFINITY);
        for k in 0..16 {
            let phi = k as f32 * std::f32::consts::PI / 16.0;
            let (s, co) = (theta0 + phi).sin_cos();
            // In trailed frames, axes that would stay on the trail sample beyond its end.
            let r = if half > 0.0 && d * phi.sin().abs() < ff {
                d.max((half + ff) / phi.cos().abs().max(0.3))
            } else {
                d
            };
            let (p, q) = (probe(co, s, r), probe(-co, -s, r));
            let a = match (p.is_finite(), q.is_finite()) {
                (true, true) => p.min(q),
                (true, false) => p,
                (false, true) => q,
                _ => continue,
            } - c.bg;
            if a > amax {
                amax2 = amax;
                amax = a;
            } else if a > amax2 {
                amax2 = a;
            }
            amin = amin.min(a);
        }
        (amax2 > f32::NEG_INFINITY).then_some(Contrast {
            height: hgt,
            contrast: hgt - amax2,
            floor: amin,
        })
    }

    /// Thresholded moments of `v - bg` in a (2hw+1)² window centred on (cx, cy).
    pub fn moments(&self, cx: f64, cy: f64, hw: usize, bg: Bg, thr: f32) -> Moments {
        let (ix, iy) = (cx.round() as isize, cy.round() as isize);
        let hw = hw as isize;
        let mut m = Moments::default();
        let (mut s, mut sx, mut sy) = (0.0f64, 0.0f64, 0.0f64);
        let (mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64);
        for yy in iy - hw..=iy + hw {
            if yy < 0 || yy >= self.h as isize {
                m.clipped = true;
                continue;
            }
            let row_bg = bg.at(0.0, yy as f32) + thr;
            for xx in ix - hw..=ix + hw {
                if xx < 0 || xx >= self.w as isize {
                    m.clipped = true;
                    continue;
                }
                let v = self.px(xx as usize, yy as usize);
                if !v.is_finite() {
                    m.clipped = true;
                    continue;
                }
                let f = (v - row_bg - bg.gx * xx as f32).max(0.0) as f64;
                if f > 0.0 {
                    let (dx, dy) = (xx as f64 - cx, yy as f64 - cy);
                    s += f;
                    sx += f * dx;
                    sy += f * dy;
                    sxx += f * dx * dx;
                    syy += f * dy * dy;
                    sxy += f * dx * dy;
                }
            }
        }
        if s > 0.0 {
            let (mx, my) = (sx / s, sy / s);
            m.sum = s;
            m.x = cx + mx;
            m.y = cy + my;
            m.xx = sxx / s - mx * mx;
            m.yy = syy / s - my * my;
            m.xy = sxy / s - mx * my;
        }
        m
    }

    /// Gaussian-windowed centroid (SExtractor XWIN-style fixed point:
    /// Σ w·I·(x - x̂) = 0), window recentred on the nearest pixel each iteration.
    fn xwin(&self, x0: f64, y0: f64, hw: usize, s: f64, bg: Bg) -> Option<(f64, f64)> {
        let (mut x, mut y) = (x0, y0);
        let n = 2 * hw + 1;
        let mut wx = [0f32; 31];
        let mut wy = [0f32; 31];
        let inv = (-0.5 / (s * s)) as f32;
        for _ in 0..10 {
            let (ix, iy) = (
                x.round() as isize - hw as isize,
                y.round() as isize - hw as isize,
            );
            if ix < 0 || iy < 0 || ix as usize + n > self.w || iy as usize + n > self.h {
                return None;
            }
            let (fx, fy) = ((x - ix as f64) as f32, (y - iy as f64) as f32);
            for k in 0..n {
                let (dx, dy) = (k as f32 - fx, k as f32 - fy);
                wx[k] = (dx * dx * inv).exp();
                wy[k] = (dy * dy * inv).exp();
            }
            let (mut sw, mut sx, mut sy) = (0.0f32, 0.0f32, 0.0f32);
            for (j, &wyj) in wy[..n].iter().enumerate() {
                let row = (iy as usize + j) * self.w + ix as usize;
                let row_bg = bg.at(0.0, (iy + j as isize) as f32);
                let (mut rw, mut rx) = (0.0f32, 0.0f32);
                for (i, &v) in self.img[row..row + n].iter().enumerate() {
                    if !v.is_finite() {
                        continue;
                    }
                    let b = row_bg + bg.gx * (ix + i as isize) as f32;
                    let f = wx[i] * (v - b);
                    rw += f;
                    rx += f * (i as f32 - fx);
                }
                sw += wyj * rw;
                sx += wyj * rx;
                sy += wyj * rw * (j as f32 - fy);
            }
            let (sw, sx, sy) = (sw as f64, sx as f64, sy as f64);
            if sw <= 0.0 {
                return None;
            }
            let dx = (2.0 * sx / sw).clamp(-1.0, 1.0);
            let dy = (2.0 * sy / sw).clamp(-1.0, 1.0);
            x += dx;
            y += dy;
            if dx.abs() < 1e-3 && dy.abs() < 1e-3 {
                break;
            }
        }
        (dist(x - x0, y - y0) <= (hw as f64).max(2.0)).then_some((x, y))
    }

    /// Median of finite pixels in the annulus r_in..r_out around (cx, cy),
    /// sampled on a sparse grid (≤ ~80 samples).
    fn annulus(&self, cx: f64, cy: f64, r_in: f64, r_out: f64) -> Option<f32> {
        let step = ((r_out * r_out * 3.2 / 80.0).sqrt().floor() as isize).max(1);
        let ro = r_out.ceil() as isize;
        let (ix, iy) = (cx.round() as isize, cy.round() as isize);
        let mut buf: Vec<f32> = Vec::with_capacity(256);
        let (ri2, ro2) = (r_in * r_in, r_out * r_out);
        let mut yy = -ro;
        while yy <= ro {
            let y = iy + yy;
            if y >= 0 && y < self.h as isize {
                let mut xx = -ro;
                while xx <= ro {
                    let x = ix + xx;
                    let d2 = ((x as f64 - cx).powi(2)) + ((y as f64 - cy).powi(2));
                    if x >= 0 && x < self.w as isize && d2 >= ri2 && d2 <= ro2 {
                        let v = self.px(x as usize, y as usize);
                        if v.is_finite() {
                            buf.push(v);
                        }
                    }
                    xx += step;
                }
            }
            yy += step;
        }
        if buf.len() < 12 {
            return None;
        }
        let mid = buf.len() / 2;
        Some(*buf.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1)
    }

    /// Background-subtracted aperture sum: (flux, pixels, touched edge/mask).
    fn aperture(&self, cx: f64, cy: f64, r: f64, bg: Bg) -> (f64, usize, bool) {
        aperture(self.img, self.w, self.h, cx, cy, r, bg)
    }

    /// Saddle test between two native peaks: true when the profile along the
    /// segment dips significantly below the fainter peak (two sources).
    pub fn separate(&self, a: (f64, f64), b: (f64, f64), bg: f32, sig_n: f32) -> bool {
        let d = dist(a.0 - b.0, a.1 - b.1);
        // Two stars of the frame PSF closer than ~0.9 FWHM merge into a single
        // maximum; a saddle at that range is structure of one (aberrated,
        // doubled or lumpy) image, not a second star.
        if d < 1.5 || d < 0.9 * self.fwhm_frame as f64 {
            return false;
        }
        let va = self.bilinear_native(a.0, a.1);
        let vb = self.bilinear_native(b.0, b.1);
        let lower = va.min(vb);
        if !lower.is_finite() {
            return false;
        }
        let n = d.ceil().max(2.0) as usize;
        let mut low = f32::INFINITY;
        for i in 1..n {
            let t = i as f64 / n as f64;
            let v = self.bilinear_native(a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
            if v.is_finite() {
                low = low.min(v);
            }
        }
        lower - low >= (3.0 * sig_n).max(0.1 * (lower - bg))
    }

    fn bilinear_native(&self, x: f64, y: f64) -> f32 {
        if !(x >= 0.0 && y >= 0.0) {
            return f32::NAN;
        }
        let (x0, y0) = (x as usize, y as usize);
        if x0 + 1 >= self.w || y0 + 1 >= self.h {
            return if x0 < self.w && y0 < self.h {
                self.px(x0, y0)
            } else {
                f32::NAN
            };
        }
        let (tx, ty) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
        let i = y0 * self.w + x0;
        let d = self.img;
        let top = d[i] + (d[i + 1] - d[i]) * tx;
        let bottom = d[i + self.w] + (d[i + self.w + 1] - d[i + self.w]) * tx;
        top + (bottom - top) * ty
    }

    /// Core of a clipped or near-saturated star: pixels ≥ `level` connected to
    /// the peak. Returns (centroid x, y, pixel count, pixels ≥ `top`, number of
    /// strict local maxima inside, moments).
    fn plateau(
        &self,
        px: usize,
        py: usize,
        level: f32,
        top: f32,
    ) -> (f64, f64, usize, usize, usize, Moments) {
        // A small search box first; a larger one only for regions that reach it.
        match self.plateau_in(px, py, level, top, 24) {
            Some(r) => r,
            None => self.plateau_in(px, py, level, top, 128).unwrap_or_else(|| {
                self.plateau_in(px, py, f32::INFINITY, top, 1)
                    .expect("single-pixel region")
            }),
        }
    }

    /// [`Ctx::plateau`] inside a box of half-size `r`; `None` when the region
    /// reaches the box (and the box is smaller than the frame there).
    fn plateau_in(
        &self,
        px: usize,
        py: usize,
        level: f32,
        top: f32,
        r: usize,
    ) -> Option<(f64, f64, usize, usize, usize, Moments)> {
        let (x0, y0) = (px.saturating_sub(r), py.saturating_sub(r));
        let (x1, y1) = ((px + r).min(self.w - 1), (py + r).min(self.h - 1));
        let bw = x1 - x0 + 1;
        let mut seen = vec![false; bw * (y1 - y0 + 1)];
        let mut stack = vec![(px, py)];
        seen[(py - y0) * bw + px - x0] = true;
        let (mut n, mut sx, mut sy) = (0usize, 0.0f64, 0.0f64);
        let (mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64);
        let mut m = Moments::default();
        let mut n_top = 0;
        let mut n_max = 0;
        while let Some((x, y)) = stack.pop() {
            n += 1;
            let v = self.px(x, y);
            n_top += (v >= top) as usize;
            if x > 0 && y > 0 && x + 1 < self.w && y + 1 < self.h {
                let higher = (y - 1..=y + 1)
                    .flat_map(|yy| (x - 1..=x + 1).map(move |xx| (xx, yy)))
                    .any(|(xx, yy)| (xx, yy) != (x, y) && self.px(xx, yy) >= v);
                n_max += !higher as usize;
            }
            let (fx, fy) = (x as f64, y as f64);
            sx += fx;
            sy += fy;
            sxx += fx * fx;
            syy += fy * fy;
            sxy += fx * fy;
            if x == 0 || y == 0 || x + 1 == self.w || y + 1 == self.h {
                m.clipped = true;
            } else if (x == x0 || y == y0 || x == x1 || y == y1) && r > 1 && r < 128 {
                return None;
            }
            let neighbours = [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ];
            for (nx, ny) in neighbours {
                if nx < x0 || ny < y0 || nx > x1 || ny > y1 {
                    continue;
                }
                let i = (ny - y0) * bw + nx - x0;
                if seen[i] {
                    continue;
                }
                seen[i] = true;
                let v = self.px(nx, ny);
                if v >= level || v.is_nan() {
                    if v.is_nan() {
                        m.clipped = true;
                        continue;
                    }
                    stack.push((nx, ny));
                }
            }
        }
        let nf = n as f64;
        let (cx, cy) = (sx / nf, sy / nf);
        m.sum = nf;
        m.x = cx;
        m.y = cy;
        m.xx = sxx / nf - cx * cx + 1.0 / 12.0;
        m.yy = syy / nf - cy * cy + 1.0 / 12.0;
        m.xy = sxy / nf - cx * cy;
        Some((cx, cy, n, n_top, n_max, m))
    }

    /// Measure one source whose native peak pixel is (px, py).
    /// `sw_cap` caps the Gaussian window width (used for blended pairs).
    pub fn measure(
        &self,
        px: usize,
        py: usize,
        compact: bool,
        sw_cap: f32,
        deblend: bool,
    ) -> Option<Det> {
        let pv = self.px(px, py);
        if !pv.is_finite() {
            return None;
        }
        let (plane_bg, sig_n) = self.plane_bg(px, py);
        let bg_tile = plane_bg.c;
        let mut hpk = pv - bg_tile;
        if hpk <= 0.0 {
            return None;
        }
        let ff = self.fwhm_frame as f64;
        let mut det = Det {
            px: px as u32,
            py: py as u32,
            sig_n,
            bg: bg_tile,
            plane: plane_bg,
            ..Default::default()
        };
        // Flat top: five or more of the 3×3 pixels at the peak level.
        let tol = (0.01 * hpk).max(2.0 * sig_n).max(0.5 * self.lsb);
        // Exact clipping: neighbours at the peak value when the peak is the
        // frame's saturation level (independent of the noise).
        let tol_sat = (0.5 * self.lsb).max(1e-6 * self.sat_level.abs());
        let at_sat = pv >= self.sat_level - tol_sat;
        let (mut n_flat, mut n_clip) = (0, 0);
        let mut max4 = f32::NEG_INFINITY;
        for (dx, dy) in [
            (-1isize, -1isize),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ] {
            let (x, y) = (px as isize + dx, py as isize + dy);
            if x < 0 || y < 0 || x >= self.w as isize || y >= self.h as isize {
                continue;
            }
            let v = self.px(x as usize, y as usize);
            if v >= pv - tol {
                n_flat += 1;
            }
            n_clip += (at_sat && v >= pv - tol_sat) as usize;
            if dx == 0 || dy == 0 {
                max4 = max4.max(v);
            }
        }
        det.ratio4 = ((max4 - bg_tile) / hpk).max(0.0);
        // Clipped plateau, or a lumpy top close to the frame's saturation level:
        // one source, measured on its core region (no deblending inside it).
        let bright = hpk >= 50.0 * sig_n;
        // A clipped disc in a noisy (stretched 8-bit) frame is not `bright`
        // against its noise, yet its flat top defeats the windowed centroid,
        // which would stay at the plateau corner where the peak search lands.
        let clipped_disc = !bright && at_sat && n_clip >= 1;
        let flat = bright && n_flat >= 4;
        let near_sat = bright && pv >= bg_tile + 0.85 * (self.sat_level - bg_tile);

        let trailed = self.trail.filter(|_| !compact);
        let (cx, cy, fwhm, elong, pa, r_ap, clipped);
        // The core region (above 0.7 of the peak) of a bright star is used when
        // its top is flat, lumpy, or clearly larger than that of an unclipped
        // star of the frame PSF (≈ 0.4 FWHM² for a Gaussian): clipped discs and
        // photographically saturated stars.
        let core_region = if flat || near_sat || bright || clipped_disc {
            let level = if clipped_disc {
                pv - (0.05 * hpk).max(tol_sat)
            } else if flat && !near_sat {
                pv - (0.05 * hpk).max(tol)
            } else {
                bg_tile + 0.7 * hpk
            };
            let r = self.plateau(px, py, level, pv - tol);
            let lumpy = r.4 >= 2;
            let large = r.2 as f64 >= 0.8 * ff.max(3.0).powi(2) + 6.0;
            let use_core = if clipped_disc {
                // One round clipped disc; an elongated clipped region is a
                // saturated pair (bm_1_heart-nebula: two catalog stars 17 px
                // apart), left to the windowed measurement of each peak.
                r.2 >= 5 && r.5.shape(0.0).1 <= 1.4
            } else {
                flat || lumpy || large
            };
            use_core.then_some(r)
        } else {
            None
        };
        if let Some((x, y, n, n_top, _, m)) = core_region {
            let core = (n as f64 / std::f64::consts::PI).sqrt();
            let (_, e, p) = m.shape(0.0);
            det.core = core as f32;
            if n_top >= 5 {
                det.flags |= flag::SATURATED;
            }
            cx = x;
            cy = y;
            fwhm = (2.0 * core).max(ff);
            elong = e;
            pa = p;
            r_ap = core + 1.5 * ff.max(2.0) + 1.0;
            clipped = m.clipped;
        } else {
            let thr = (1.5 * sig_n).max(0.02 * hpk);
            let f0 = (thr / hpk) as f64;
            let mut hw = if compact { 2 } else { 3 };
            let m0 = self.moments(px as f64, py as f64, hw, plane_bg, thr);
            let (fwhm0, _, _) = m0.shape(f0);
            if !compact && fwhm0 > 4.0 {
                hw = 5;
            }
            if let Some(t) = trailed {
                hw = ((0.5 * t.length) as f64 + ff.max(2.0))
                    .ceil()
                    .clamp(3.0, 15.0) as usize;
            }
            let fref = if ff > 0.0 {
                fwhm0.clamp(0.7 * ff, 1.5 * ff)
            } else {
                fwhm0.max(1.5)
            };
            let r_guess = if compact {
                (1.2 * fref).clamp(2.0, 6.0)
            } else {
                (1.5 * fref).clamp(2.5, 20.0)
            };
            // Annulus background where the tile model may miss local structure:
            // on request, or in tiles whose residual noise shows structure.
            let r_in = r_guess + 1.5;
            let r_out = r_in + ff.max(fref).max(3.0);
            let mut bg = plane_bg;
            let rough = sig_n > 1.5 * self.grid.typical_sig_n;
            if (self.local_background || rough)
                && let Some(ann) = self.annulus(px as f64, py as f64, r_in, r_out)
            {
                let structured = (ann - bg_tile).abs() > sig_n.max(0.02 * hpk);
                if self.local_background || structured {
                    bg = plane_bg.with_level(ann);
                }
            }
            hpk = pv - bg.c;
            if hpk <= 0.0 {
                return None;
            }
            det.bg = bg.c;
            det.plane = bg;
            let s_w = (fref / 2.3548).clamp(0.8, 3.0).min(sw_cap.max(0.6) as f64);
            let (x, y) = if trailed.is_some() {
                let m = self.moments(px as f64, py as f64, hw, bg, thr);
                let m = self.moments(m.x, m.y, hw, bg, thr);
                if m.sum <= 0.0 {
                    return None;
                }
                (m.x, m.y)
            } else {
                let (sx, sy) = if m0.sum > 0.0
                    && (m0.x - px as f64).abs() < 1.0
                    && (m0.y - py as f64).abs() < 1.0
                {
                    (m0.x, m0.y)
                } else {
                    (px as f64, py as f64)
                };
                match self.xwin(sx, sy, hw, s_w, bg) {
                    Some(c) => c,
                    None => {
                        let m = self.moments(px as f64, py as f64, hw, bg, thr);
                        if m.sum <= 0.0 {
                            return None;
                        }
                        (m.x, m.y)
                    }
                }
            };
            if (x - px as f64).hypot(y - py as f64) > 2.5 + hw as f64 * 0.5 {
                return None;
            }
            let m = self.moments(x, y, hw, bg, thr);
            let shape = if m.sum > 0.0 {
                m.shape((thr / hpk) as f64)
            } else {
                (fref, 1.0, 0.0)
            };
            cx = x;
            cy = y;
            fwhm = shape.0;
            elong = shape.1;
            pa = shape.2;
            // The frame PSF sets the aperture, so that a blend cannot enlarge
            // its own aperture (and its rank) through an inflated FWHM.
            let fa = if ff > 0.0 {
                ff.max(0.8 * fref)
            } else {
                fwhm.max(fref)
            };
            r_ap = if compact {
                (1.2 * fa).clamp(2.0, 6.0)
            } else {
                (1.25 * fa).clamp(2.5, 16.0)
            };
            clipped = m.clipped;
            if deblend {
                self.find_secondary(&mut det, px, py, hpk, bg.c, sig_n);
            }
        }
        let (flux, n_ap, ap_clipped) = self.aperture(cx, cy, r_ap, det.plane);
        if flux <= 0.0 || n_ap == 0 {
            return None;
        }
        det.snr = flux / (sig_n as f64 * (n_ap as f64).sqrt()).max(f64::MIN_POSITIVE);
        let (mut cx, mut cy) = (cx, cy);
        if det.core == 0.0 && trailed.is_none() && sw_cap.is_infinite() {
            // The Gaussian-windowed centroid is optimal for faint stars but biased
            // for undersampled bright ones; the thresholded centre of mass is
            // unbiased once the threshold is small against the peak. Blend by SNR.
            let alpha = ((det.snr - 20.0) / 60.0).clamp(0.0, 1.0);
            if alpha > 0.0 {
                let r = if compact {
                    (1.2 * fwhm).clamp(2.0, 5.0)
                } else {
                    (1.25 * fwhm).clamp(2.5, 8.0)
                };
                if let Some((x, y)) = self.com(cx, cy, r, det.plane, 1.5 * sig_n) {
                    cx += alpha * (x - cx);
                    cy += alpha * (y - cy);
                }
            }
        }
        det.x = cx;
        det.y = cy;
        det.h = hpk;
        det.flux = flux;
        det.fwhm = fwhm;
        det.elong = elong;
        det.pa = pa;
        if clipped || ap_clipped {
            det.flags |= flag::NEAR_EDGE;
        }
        Some(det)
    }

    /// Thresholded centre of mass in a circular aperture, re-centred 3 times.
    fn com(&self, x0: f64, y0: f64, r: f64, bg: Bg, thr: f32) -> Option<(f64, f64)> {
        let (mut x, mut y) = (x0, y0);
        let ri = r.ceil() as isize;
        let r2 = r * r;
        for _ in 0..2 {
            let (ix, iy) = (x.round() as isize, y.round() as isize);
            if ix - ri < 0
                || iy - ri < 0
                || ix + ri >= self.w as isize
                || iy + ri >= self.h as isize
            {
                return None;
            }
            let (mut s, mut sx, mut sy) = (0.0f64, 0.0f64, 0.0f64);
            for yy in iy - ri..=iy + ri {
                let dy = yy as f64 - y;
                if dy * dy > r2 {
                    continue;
                }
                let half = (r2 - dy * dy).sqrt();
                let xa = ((x - half).ceil() as isize).max(0) as usize;
                let xb = ((x + half).floor() as isize).min(self.w as isize - 1);
                if xb < xa as isize {
                    continue;
                }
                let row = &self.img[yy as usize * self.w..(yy as usize + 1) * self.w];
                let row_bg = bg.at(0.0, yy as f32) + thr;
                let (fx0, gx) = ((xa as f64 - x) as f32, bg.gx);
                let (mut rs, mut rx) = (0.0f32, 0.0f32);
                for (k, &v) in row[xa..=xb as usize].iter().enumerate() {
                    let f = v - row_bg - gx * (xa + k) as f32;
                    if f > 0.0 {
                        rs += f;
                        rx += f * (fx0 + k as f32);
                    }
                }
                s += rs as f64;
                sx += rx as f64;
                sy += rs as f64 * dy;
            }
            if s <= 0.0 {
                return None;
            }
            x += sx / s;
            y += sy / s;
        }
        (dist(x - x0, y - y0) < 1.5).then_some((x, y))
    }

    /// Other strict native maxima near the peak that are separated by a saddle.
    fn find_secondary(&self, det: &mut Det, px: usize, py: usize, hpk: f32, bg: f32, sig: f32) {
        let r = self.b as isize + 1;
        let min_h = (self.k * sig).max(0.1 * hpk);
        let mut found = 0;
        for dy in -r..=r {
            for dx in -r..=r {
                if dx.abs().max(dy.abs()) < 2 {
                    continue;
                }
                let (x, y) = (px as isize + dx, py as isize + dy);
                if x < 1 || y < 1 || x + 1 >= self.w as isize || y + 1 >= self.h as isize {
                    continue;
                }
                let (x, y) = (x as usize, y as usize);
                let v = self.px(x, y);
                if (v - bg).is_nan() || v - bg < min_h {
                    continue;
                }
                let mut is_max = true;
                'n: for yy in y - 1..=y + 1 {
                    for xx in x - 1..=x + 1 {
                        if (xx, yy) != (x, y) && self.px(xx, yy) >= v {
                            is_max = false;
                            break 'n;
                        }
                    }
                }
                if is_max
                    && self.separate((px as f64, py as f64), (x as f64, y as f64), bg, sig)
                    && found < 2
                {
                    det.secondary[found] = Some((x as u32, y as u32));
                    det.flags |= flag::BLENDED;
                    found += 1;
                }
            }
        }
    }
}

/// Drop duplicate measurements of one object (maxima on a clipped disc, two
/// binned peaks converging on one star) and flag genuinely separate close pairs
/// as blended (re-measured with a compact window).
///
/// Pair relations (saddle tests) are computed in parallel against all brighter
/// neighbours; a short sequential pass in brightness order then keeps a source
/// unless it duplicates a brighter source that was itself kept.
fn merge_duplicates(ctx: &Ctx, dets: &mut Vec<Det>) {
    let n = dets.len();
    if n < 2 {
        return;
    }
    let ff = (ctx.fwhm_frame as f64).max(2.0);
    // Within r_dup two maxima may be one object (the saddle test decides);
    // separate sources within r_blend contaminate each other's windows.
    let r_dup = (1.6 * ff).max(3.0);
    let r_blend = (2.0 * ff).max(4.0);
    let cell = r_blend;
    let (gw, gh) = (
        (ctx.w as f64 / cell) as usize + 1,
        (ctx.h as f64 / cell) as usize + 1,
    );
    let cell_of = |x: f64, y: f64| {
        let cx = ((x / cell).max(0.0) as usize).min(gw - 1);
        let cy = ((y / cell).max(0.0) as usize).min(gh - 1);
        (cx, cy)
    };
    // Clipped discs first, then by peak height: a clipped plateau yields
    // several maxima of (nearly) equal height, and the plateau measurement
    // must survive over a windowed one taken at its rim.
    let mut order: Vec<usize> = (0..n).collect();
    order.par_sort_by(|&a, &b| {
        (dets[b].core > 0.0)
            .cmp(&(dets[a].core > 0.0))
            .then(dets[b].h.total_cmp(&dets[a].h))
            .then(a.cmp(&b))
    });
    let mut rank = vec![0u32; n];
    for (r, &i) in order.iter().enumerate() {
        rank[i] = r as u32;
    }
    let mut head = vec![u32::MAX; gw * gh];
    let mut next = vec![u32::MAX; n];
    for (i, d) in dets.iter().enumerate() {
        let (gx, gy) = cell_of(d.x, d.y);
        next[i] = head[gy * gw + gx];
        head[gy * gw + gx] = i as u32;
    }
    // Clipped discs, and bright broad sources without a core (photographic
    // halos, lumpy tops), in cell lists covering their radius: maxima inside a
    // disc are duplicates; inside a broad source (1.2 FWHM) they get the saddle
    // test even beyond r_dup. Both kinds are few.
    let mut discs: Vec<Vec<u32>> = vec![];
    let mut big: Vec<Vec<u32>> = vec![];
    let bright = |d: &Det| d.h >= 0.1 * (ctx.sat_level - d.bg);
    for (i, d) in dets.iter().enumerate() {
        let (map, r) = if d.core > 0.0 {
            (&mut discs, d.core as f64 + 2.0)
        } else if d.fwhm > 1.6 * ff && bright(d) {
            (&mut big, (1.2 * d.fwhm).min(40.0))
        } else {
            continue;
        };
        if map.is_empty() {
            map.resize(gw * gh, vec![]);
        }
        let (c0x, c0y) = cell_of(d.x - r, d.y - r);
        let (c1x, c1y) = cell_of(d.x + r, d.y + r);
        for yy in c0y..=c1y {
            for xx in c0x..=c1x {
                map[yy * gw + xx].push(i as u32);
            }
        }
    }
    // Relations to brighter sources: (other, duplicate?, distance).
    // Flat list (source, other, duplicate?, distance), grouped by source below
    // (no per-source allocation).
    let mut rels: Vec<(u32, u32, bool, f32)> = (0..n)
        .into_par_iter()
        .fold(Vec::new, |mut acc, i| {
            let d = &dets[i];
            let mut out = |j: u32, dup: bool, dist: f32| acc.push((i as u32, j, dup, dist));
            let (gx, gy) = cell_of(d.x, d.y);
            for &j in discs.get(gy * gw + gx).into_iter().flatten() {
                let s = &dets[j as usize];
                if rank[j as usize] < rank[i]
                    && (dist(s.x - d.x, s.y - d.y) < s.core as f64 + 2.0
                        || (s.px, s.py) == (d.px, d.py))
                {
                    out(j, true, 0.0);
                }
            }
            for &j in big.get(gy * gw + gx).into_iter().flatten() {
                let o = &dets[j as usize];
                let dist = dist(o.x - d.x, o.y - d.y);
                if rank[j as usize] < rank[i] && dist >= r_dup && dist < (1.2 * o.fwhm).min(40.0) {
                    let pa = (o.px as f64, o.py as f64);
                    let pb = (d.px as f64, d.py as f64);
                    if !ctx.separate(pa, pb, d.bg, d.sig_n) {
                        out(j, true, dist as f32);
                    }
                }
            }
            for yy in gy.saturating_sub(1)..=(gy + 1).min(gh - 1) {
                for xx in gx.saturating_sub(1)..=(gx + 1).min(gw - 1) {
                    let mut j = head[yy * gw + xx];
                    while j != u32::MAX {
                        let ju = j as usize;
                        let o = &dets[ju];
                        if rank[ju] < rank[i] {
                            let dist = dist(o.x - d.x, o.y - d.y);
                            if (o.px, o.py) == (d.px, d.py) || dist < 1.0 {
                                out(j, true, dist as f32);
                            } else if dist < r_dup {
                                let pa = (o.px as f64, o.py as f64);
                                let pb = (d.px as f64, d.py as f64);
                                let separate = ctx.separate(pa, pb, d.bg, d.sig_n);
                                out(j, !separate, dist as f32);
                            } else if dist < r_blend {
                                out(j, false, dist as f32);
                            }
                        }
                        j = next[ju];
                    }
                }
            }
            acc
        })
        .reduce(Vec::new, |mut a, mut b| {
            a.append(&mut b);
            a
        });
    rels.par_sort_unstable_by_key(|r| r.0);
    let mut start = vec![0usize; n + 1];
    for r in &rels {
        start[r.0 as usize + 1] += 1;
    }
    for i in 0..n {
        start[i + 1] += start[i];
    }
    let relations = |i: usize| rels[start[i]..start[i + 1]].iter().map(|r| (r.1, r.2, r.3));
    let mut keep = vec![false; n];
    // (source, separation, neighbour peak / own peak)
    let mut blended: Vec<(usize, f64, f32)> = vec![];
    for &i in &order {
        if relations(i).any(|(j, dup, _)| dup && keep[j as usize]) {
            continue;
        }
        keep[i] = true;
        for (j, _, dist) in relations(i).filter(|r| keep[r.0 as usize]) {
            let (hi, hj) = (
                dets[i].h.max(f32::MIN_POSITIVE),
                dets[j as usize].h.max(f32::MIN_POSITIVE),
            );
            blended.push((i, dist as f64, hj / hi));
            blended.push((j as usize, dist as f64, hi / hj));
        }
    }
    // Re-measure with compact windows the sources whose nearest separate
    // neighbour is close (< 1.3 FWHM) and bright enough to bias the centroid.
    blended.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    blended.dedup_by_key(|b| b.0);
    let redo: Vec<(usize, Det)> = blended
        .par_iter()
        .filter(|(i, sep, ratio)| {
            keep[*i] && dets[*i].core == 0.0 && *sep < 1.3 * ff && *ratio >= 0.25
        })
        .filter_map(|&(i, sep, _)| {
            let d = &dets[i];
            let cap = (sep / 3.0).max(0.6) as f32;
            ctx.measure(d.px as usize, d.py as usize, true, cap, false)
                .map(|mut m| {
                    m.flags |= d.flags | flag::BLENDED;
                    (i, m)
                })
        })
        .collect();
    for (i, m) in redo {
        dets[i] = m;
    }
    for &(i, _, _) in &blended {
        dets[i].flags |= flag::BLENDED;
    }
    let mut k = 0;
    dets.retain(|_| {
        k += 1;
        keep[k - 1]
    });
}

/// Merge PSF lobes. Aberrated optics (coma, tilt, astigmatism) or a guiding
/// jump give every bright star a secondary maximum at a similar offset; the
/// saddle test cannot tell such a lobe from a companion star, but the frame
/// can: a pair whose offset vector is shared by nearby pairs is one star. The
/// lobe is folded into its primary (flux-weighted light centroid, summed
/// flux); random companions in crowded fields do not share vectors and stay.
pub(crate) fn merge_psf_lobes(ctx: &Ctx, dets: &mut Vec<Det>) {
    let n = dets.len();
    let ff = ctx.fwhm_frame as f64;
    if n < 20 || ff <= 0.0 {
        return;
    }
    let reach = 2.5 * ff;
    let cell = reach.max(4.0);
    let (gw, gh) = (
        (ctx.w as f64 / cell) as usize + 1,
        (ctx.h as f64 / cell) as usize + 1,
    );
    let cell_of = |x: f64, y: f64| {
        (
            ((x / cell).max(0.0) as usize).min(gw - 1),
            ((y / cell).max(0.0) as usize).min(gh - 1),
        )
    };
    let mut head = vec![u32::MAX; gw * gh];
    let mut next = vec![u32::MAX; n];
    for (i, d) in dets.iter().enumerate() {
        let (cx, cy) = cell_of(d.x, d.y);
        next[i] = head[cy * gw + cx];
        head[cy * gw + cx] = i as u32;
    }
    // Pairs (primary, secondary, offset) for significant secondaries whose
    // nearest brighter neighbour lies within 2.5 FWHM.
    let pairs: Vec<(u32, u32, [f64; 2])> = (0..n)
        .into_par_iter()
        .filter_map(|i| {
            let d = &dets[i];
            if d.flags & flag::SATURATED != 0 || d.h < 8.0 * d.sig_n {
                return None;
            }
            let (cx, cy) = cell_of(d.x, d.y);
            let mut best: Option<(f64, u32)> = None;
            for yy in cy.saturating_sub(1)..=(cy + 1).min(gh - 1) {
                for xx in cx.saturating_sub(1)..=(cx + 1).min(gw - 1) {
                    let mut j = head[yy * gw + xx];
                    while j != u32::MAX {
                        let o = &dets[j as usize];
                        let dist = dist(o.x - d.x, o.y - d.y);
                        if j as usize != i
                            && o.h > d.h
                            && dist < reach
                            && best.is_none_or(|b| dist < b.0)
                        {
                            best = Some((dist, j));
                        }
                        j = next[j as usize];
                    }
                }
            }
            let (_, j) = best?;
            let o = &dets[j as usize];
            (d.h >= 0.1 * o.h).then_some((j, i as u32, [d.x - o.x, d.y - o.y]))
        })
        .collect();
    if pairs.len() < 8 {
        return;
    }
    // The 3000 brightest pairs, bucketed by position (cells of 0.35 of the frame).
    let mut pairs = pairs;
    pairs.sort_by(|a, b| dets[b.0 as usize].flux.total_cmp(&dets[a.0 as usize].flux));
    pairs.truncate(3000);
    let radius = 0.35 * (ctx.w.max(ctx.h) as f64);
    let (bw, bh) = (
        (ctx.w as f64 / radius) as usize + 1,
        (ctx.h as f64 / radius) as usize + 1,
    );
    let bucket_of = |x: f64, y: f64| {
        (
            ((x / radius).max(0.0) as usize).min(bw - 1),
            ((y / radius).max(0.0) as usize).min(bh - 1),
        )
    };
    let mut buckets: Vec<Vec<u32>> = vec![vec![]; bw * bh];
    for (k, &(p, _, _)) in pairs.iter().enumerate() {
        let (bx, by) = bucket_of(dets[p as usize].x, dets[p as usize].y);
        buckets[by * bw + bx].push(k as u32);
    }
    // A pair is a lobe when at least 3 and 30 % of the pairs around it share
    // its offset vector.
    let lobe: Vec<bool> = pairs
        .par_iter()
        .enumerate()
        .map(|(k, &(p, _, v))| {
            let (x, y) = (dets[p as usize].x, dets[p as usize].y);
            let (bx, by) = bucket_of(x, y);
            let (mut near, mut agree) = (0usize, 0usize);
            let tol = (0.35 * v[0].hypot(v[1])).max(2.0);
            for yy in by.saturating_sub(1)..=(by + 1).min(bh - 1) {
                for xx in bx.saturating_sub(1)..=(bx + 1).min(bw - 1) {
                    for &m in &buckets[yy * bw + xx] {
                        let (q, _, u) = pairs[m as usize];
                        let o = &dets[q as usize];
                        if m as usize == k || dist(o.x - x, o.y - y) > radius {
                            continue;
                        }
                        near += 1;
                        agree += ((u[0] - v[0]).hypot(u[1] - v[1]) < tol) as usize;
                    }
                }
            }
            agree >= 3 && agree * 10 >= near * 3
        })
        .collect();
    let mut drop = vec![false; n];
    for (&(p, s, _), &is_lobe) in pairs.iter().zip(&lobe) {
        let (p, s) = (p as usize, s as usize);
        if !is_lobe || drop[p] || drop[s] {
            continue;
        }
        let (fp, fs) = (dets[p].flux.max(0.0), dets[s].flux.max(0.0));
        let total = (fp + fs).max(f64::MIN_POSITIVE);
        let (sx, sy, sf) = (dets[s].x, dets[s].y, dets[s].fwhm);
        let d = &mut dets[p];
        let sep = dist(sx - d.x, sy - d.y);
        d.x = (fp * d.x + fs * sx) / total;
        d.y = (fp * d.y + fs * sy) / total;
        d.flux = fp + fs;
        d.snr *= (total / fp.max(f64::MIN_POSITIVE)).sqrt();
        d.fwhm = d.fwhm.max(sep + 0.5 * (d.fwhm + sf));
        d.flags &= !flag::BLENDED;
        drop[s] = true;
    }
    let mut k = 0;
    dets.retain(|_| {
        k += 1;
        !drop[k - 1]
    });
}
