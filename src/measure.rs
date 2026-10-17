//! Forced photometry at given positions: the evidence layer of the annotation atlas
//! (docs/annotations/DESIGN.md §7), calibrated as the E0 spike recommends
//! (`lab/skyanchor_lab/atlas/spike_evidence.py`, rule R). Client side: no solver code, so the
//! browser worker, `skyshot serve` and the iPhone engine measure with the same code.
//!
//! A [`FrameModel`] is built once per frame: the detector's background grid (no second
//! background model), sky level and noise, the clip level, the frame FWHM (Gaussian fits to
//! bright isolated stars), the photon-transfer gain of linear sources, the noise factors of the
//! difference-of-Gaussians (DoG) aperture, the noise calibration χ (the robust SNR spread at a
//! few hundred random empty apertures per environment) and, with reference stars, a zero point
//! and depth. [`measure`] then costs O(targets):
//!
//! - plain aperture r = 1 FWHM, annulus 2.5–4 FWHM with a robust plane, level at the median of
//!   the clipped residuals; σ² = n·s² + n²·s²/n_ann (+ F/g for linear sources only);
//! - the same position on a DoG patch (aperture 1.2 FWHM, annulus 4–6 FWHM): a confirmation;
//! - a Gaussian-windowed centroid started at the position;
//! - calibrated SNR = SNR / χ(environment), status by rule R ([`Status`]).
//!
//! Memory: no full-frame plane is kept. Building the model reads the frame a few times and
//! holds the binned detection plane while the grid is built (1/b² of the frame for a bin factor
//! b > 1, as the detector does); a measurement works on a patch of (12 FWHM + 6 r + 3)² pixels
//! (r the box radius of the wide DoG blur).
//!
//! Determinism: transcendental functions go through `libm`, reductions run in a fixed order and
//! the random calibration positions come from a fixed seed, so native and WebAssembly builds
//! return the same numbers (fluxes bit for bit: sums, products and square roots only).

use crate::{
    coordinates as engine,
    extract::{self, Bg, Grid},
    highpass,
    image_io::{GrayImage, SourceFormat, SourceInfo},
    types::{Star, Wcs},
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Plain aperture radius, in FWHM (spike estimator A).
pub const R_AP: f64 = 1.0;
/// Inner and outer radius of the plain aperture's background annulus, in FWHM.
pub const R_IN: f64 = 2.5;
pub const R_OUT: f64 = 4.0;
/// DoG confirmation: aperture radius and annulus radii, in FWHM (spike estimator B).
pub const R_AP_DOG: f64 = 1.2;
pub const R_IN_DOG: f64 = 4.0;
pub const R_OUT_DOG: f64 = 6.0;
/// Rule R: `recorded` needs a calibrated SNR of at least this,
pub const SNR_RECORDED: f64 = 6.0;
/// a calibrated DoG SNR of at least this,
pub const SNR_CONFIRM: f64 = 3.0;
/// and a centroid closer than this many FWHM to the position.
pub const OFFSET_MAX: f64 = 0.75;
/// `trace` from this calibrated SNR on.
pub const SNR_TRACE: f64 = 3.0;
/// Clip level: sky + this fraction of (frame maximum − sky); a peak at or above it is saturated.
pub const SAT_FRACTION: f64 = 0.95;
/// Clipped background: an annulus level at this fraction of the way from the sky to the clip
/// level or above,
pub const BRIGHT_LEVEL: f64 = 0.85;
/// or more than this fraction of the annulus pixels at the clip level.
pub const BRIGHT_CLIPPED: f64 = 0.1;
/// Near the edge: less than this fraction of an annulus on the frame and unmasked.
pub const MIN_ANNULUS: f64 = 0.5;
/// No data: annulus noise below max(FLAT_LSB · quantisation step, FLAT_SKY · sky noise)
/// (interface fill, clipped black or white).
pub const FLAT_LSB: f64 = 0.5;
pub const FLAT_SKY: f64 = 0.05;
/// Blended: another reference star within this many FWHM that is brighter than the object and
/// brighter than the limiting magnitude + [`BLEND_DEPTH`],
pub const BLEND_RADIUS: f64 = 1.5;
pub const BLEND_DEPTH: f64 = 1.0;
/// not counting the nearest one within this distance (the object itself, which Gaia often
/// lists as a source).
pub const SELF_ARCSEC: f64 = 1.5;
/// The same in FWHM when the plate scale is unknown.
pub const SELF_FWHM: f64 = 0.25;
/// Environments: background at the position above the sky, in sky-noise units (below
/// ENV_FAINT sky, below ENV_BRIGHT faint galaxy light, else bright).
pub const ENV_FAINT: f64 = 0.5;
pub const ENV_BRIGHT: f64 = 5.0;
/// Noise calibration: random apertures per environment (the spike drew 700; with 700 the
/// recorded count of a frame moves by up to ±15 % (1 sd) with the random positions, with 2000
/// by ±1–5 %),
pub const N_CAL: usize = 2000;
/// χ of an environment needs this many good apertures (else the frame-wide χ),
pub const CAL_MIN_ENV: usize = 40;
/// the frame-wide χ this many (else the frame is uncalibrated: nothing is assessable).
pub const CAL_MIN: usize = 10;
/// A calibration aperture is empty without a detected star, or a reference star brighter than
/// the formula depth + 1, within this many FWHM.
pub const CAL_EMPTY: f64 = 2.0;
/// Candidate positions drawn for the calibration apertures (at most; a small environment
/// takes what it gets).
const N_CAL_CANDIDATES: usize = 200_000;
/// Seed of the calibration positions (fixed: the same frame gives the same model).
const CAL_SEED: u64 = 0x5eed_c0de_0f00_d5ed;
/// Frame FWHM: at most this many Gaussian fits,
const N_PSF: usize = 150;
/// of detections from this SNR on (when no reference stars are given),
const PSF_MIN_SNR: f64 = 30.0;
/// with an amplitude of at least this many sky sigmas (the relaxed value when fewer than
/// PSF_MIN_FITS stars pass; fewer than PSF_MIN_FITS_RELAXED then means an unknown FWHM).
const PSF_AMP: f64 = 30.0;
const PSF_AMP_RELAXED: f64 = 15.0;
const PSF_MIN_FITS: usize = 5;
const PSF_MIN_FITS_RELAXED: usize = 3;
/// Zero point from reference stars of at least this SNR (isolated, unsaturated).
pub const ZP_MIN_SNR: f64 = 20.0;
/// A response is linear when the free slope of instrumental against catalogue magnitudes is
/// within this of 1; magnitudes and the zero-point depth mean something only then.
pub const LINEAR_SLOPE_TOL: f64 = 0.1;
/// Limiting magnitude: a point source at this SNR on the sky.
pub const SNR_LIMIT: f64 = 5.0;
/// Suspect: recorded although a point source of the object's catalogue magnitude reaches less
/// than this SNR in the same environment (reference-star curve): a foreground star or a
/// neighbour may be what is recorded.
pub const SUSPECT_SNR: f64 = 2.0;
/// Accepted frame FWHM, pixels.
pub const MIN_FWHM: f64 = 0.5;
pub const MAX_FWHM: f64 = 64.0;
/// Request limits.
pub const MAX_TARGETS: usize = 200_000;
pub const MAX_REFS: usize = 500_000;
/// Reference stars measured for the zero point and depth: the brightest half of these and an
/// even sample of the fainter ones (all of them count as neighbours).
const MAX_REFS_MEASURED: usize = 30_000;

/// Measurement flags: the detector's [`crate::types::star_flags`] where they apply, then evidence
/// flags.
pub mod flags {
    use crate::types::star_flags;
    /// A peak at the clip level.
    pub const SATURATED: u32 = star_flags::SATURATED;
    /// A brighter source within [`super::BLEND_RADIUS`] FWHM.
    pub const BLENDED: u32 = star_flags::BLENDED;
    /// Aperture off the frame or masked, or too little of the annulus on it.
    pub const NEAR_EDGE: u32 = star_flags::NEAR_EDGE;
    /// Gamma-encoded (or measured non-linear) source: fluxes are not photometry.
    pub const NONLINEAR: u32 = 1 << 16;
    /// No sky data: masked centre, flat annulus (interface fill, clipped black or white).
    pub const NO_DATA: u32 = 1 << 17;
    /// Background at the clip level (blown-out core).
    pub const BRIGHT_BG: u32 = 1 << 18;
    /// SNR and confirmation reached, but the centroid lies elsewhere (a neighbour's light).
    pub const OFFSET: u32 = 1 << 19;
    /// SNR for `recorded` reached, but the DoG did not confirm a compact source.
    pub const UNCONFIRMED: u32 = 1 << 20;
    /// The DoG aperture could not be measured (edge, mask).
    pub const NO_DOG: u32 = 1 << 21;
    /// The frame has no noise calibration.
    pub const UNCALIBRATED: u32 = 1 << 22;
    /// Recorded, but too bright for the catalogue magnitude ([`super::SUSPECT_SNR`]).
    pub const SUSPECT: u32 = 1 << 23;
    /// Flags that make a position not assessable.
    pub const NOT_ASSESSABLE: u32 =
        SATURATED | BLENDED | NEAR_EDGE | NO_DATA | BRIGHT_BG | UNCALIBRATED;
}

/// Outcome at a position (rule R of the E0 spike).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// Calibrated SNR ≥ [`SNR_RECORDED`], DoG confirmation ≥ [`SNR_CONFIRM`], centroid within
    /// [`OFFSET_MAX`] FWHM.
    Recorded,
    /// Calibrated SNR ≥ [`SNR_TRACE`] otherwise.
    Trace,
    /// Below reach (also when the signal is a neighbour's: flag [`flags::OFFSET`]).
    Below,
    /// Edge, no data, clipped background, saturated, blended or uncalibrated (see the flags).
    NotAssessable,
}

/// Background environment of a position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Env {
    Sky,
    Faint,
    Bright,
}

impl Env {
    fn of(level: f64, sky: f64, sigma: f64) -> Option<Env> {
        let ex = (level - sky) / sigma;
        if !ex.is_finite() {
            None
        } else if ex < ENV_FAINT {
            Some(Env::Sky)
        } else if ex < ENV_BRIGHT {
            Some(Env::Faint)
        } else {
            Some(Env::Bright)
        }
    }
    fn index(self) -> usize {
        self as usize
    }
}

/// An error of the measurement layer: a stable code and a message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeasureError {
    /// `bad_request`, `no_frame` (the image cannot be modelled), `no_psf` (the FWHM cannot be
    /// measured: give one), `content_conflict` (a different content mask was applied before).
    pub code: &'static str,
    pub message: String,
}

impl std::fmt::Display for MeasureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.message, self.code)
    }
}

impl std::error::Error for MeasureError {}

fn error(code: &'static str, message: impl Into<String>) -> MeasureError {
    MeasureError {
        code,
        message: message.into(),
    }
}

/// Image content: pixels outside every `include` box (when there is one) or inside an
/// `exclude` box carry no sky (a screenshot's desktop and dialogs). Boxes are `[x0, y0, x1,
/// y1]` in zero-based pixels, end exclusive.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Content {
    pub include: Vec<[f64; 4]>,
    pub exclude: Vec<[f64; 4]>,
}

/// Set the pixels outside `c` to NaN (masked), row by row; returns how many were masked.
pub fn apply_content(img: &mut GrayImage, c: &Content) -> usize {
    let (w, h) = (img.width as usize, img.height as usize);
    let span = |b: &[f64; 4]| {
        let cl = |v: f64, n: usize| v.clamp(0.0, n as f64) as usize;
        (cl(b[0], w), cl(b[1], h), cl(b[2], w), cl(b[3], h))
    };
    let (inc, exc): (Vec<_>, Vec<_>) = (
        c.include.iter().map(span).collect(),
        c.exclude.iter().map(span).collect(),
    );
    let mut keep = vec![false; w];
    let mut n = 0;
    for (y, row) in img.pixels[..w * h].chunks_mut(w).enumerate() {
        keep.fill(inc.is_empty());
        for &(x0, y0, x1, y1) in &inc {
            if (y0..y1).contains(&y) && x0 < x1 {
                keep[x0..x1].fill(true);
            }
        }
        for &(x0, y0, x1, y1) in &exc {
            if (y0..y1).contains(&y) && x0 < x1 {
                keep[x0..x1].fill(false);
            }
        }
        for (p, k) in row.iter_mut().zip(&keep) {
            if !k && !p.is_nan() {
                *p = f32::NAN;
                n += 1;
            }
        }
    }
    n
}

/// Model options (the `frame` object of a request).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FrameOptions {
    /// Frame FWHM in pixels (default: Gaussian fits to bright isolated stars).
    pub fwhm: Option<f64>,
    /// Linear source (default: FITS yes, 8/16-bit images no: gamma-encoded).
    pub linear: Option<bool>,
    /// Image content (masked in place before the model is built).
    pub content: Option<Content>,
}

/// A reference star (photref pack, `match_mags`, D80) in pixels, with its magnitude.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefStar {
    pub x: f64,
    pub y: f64,
    pub mag: f64,
}

/// A position to measure, with the catalogue magnitude of the object when known (for the
/// blend rule; the band of the reference stars is assumed).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Target {
    pub x: f64,
    pub y: f64,
    pub mag: Option<f64>,
}

/// How the frame FWHM was found.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FwhmSource {
    /// Given by the caller.
    Given,
    /// Median of Gaussian fits to reference stars or detections.
    Fit,
    /// The same with the relaxed amplitude cut (few bright stars).
    FitRelaxed,
}

/// Robust zero point m_cat = zp + m_inst (slope fixed at 1, clipped median) and the free slope
/// of m_inst = a + b·m_cat (clipped least squares; b ≠ 1 is a non-linear response).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZeroPoint {
    /// Stars kept by the slope-1 fit.
    pub n: usize,
    pub zp: f64,
    /// 1.4826 MAD of the kept residuals, magnitudes.
    pub scatter: f64,
    pub slope: Option<f64>,
    pub slope_se: Option<f64>,
    pub intercept: Option<f64>,
    /// Catalogue magnitudes of the kept stars.
    pub mag_range: [f64; 2],
    /// |slope − 1| ≤ [`LINEAR_SLOPE_TOL`].
    pub linear: bool,
}

/// χ (noise calibration) per environment and over the frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Chi {
    pub sky: Option<f64>,
    pub faint: Option<f64>,
    pub bright: Option<f64>,
    pub all: Option<f64>,
}

impl Chi {
    fn get(&self, e: Env) -> Option<f64> {
        match e {
            Env::Sky => self.sky,
            Env::Faint => self.faint,
            Env::Bright => self.bright,
        }
        .or(self.all)
    }
}

/// Per environment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct PerEnv<T> {
    pub sky: T,
    pub faint: T,
    pub bright: T,
}

/// What the model knows about the frame (the `model` object of a response).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelSummary {
    pub width: u32,
    pub height: u32,
    pub fwhm: f64,
    pub fwhm_source: FwhmSource,
    /// Gaussian fits behind the FWHM (0 when given).
    pub n_psf: usize,
    /// Bin factor of the background grid (the detector's).
    pub bin: usize,
    pub lsb: f64,
    pub level_sky: f64,
    pub sigma_sky: f64,
    /// Largest finite pixel value and the clip level derived from it.
    pub saturation: f64,
    pub sat_thr: f64,
    /// Linear source (FITS, or as given).
    pub linear: bool,
    /// Fluxes are photometry: a linear source whose zero-point slope (when fitted) is 1.
    pub photometric: bool,
    /// Photon-transfer gain (data units⁻¹ per electron, i.e. e⁻ per unit), linear sources only.
    pub gain: Option<f64>,
    /// Noise calibration of the plain and of the DoG aperture.
    pub chi: Chi,
    pub chi_dog: Chi,
    /// Good calibration apertures per environment.
    pub n_cal: PerEnv<usize>,
    pub zero_point: Option<ZeroPoint>,
    /// SNR-5 point source on the sky through the zero point (calibrated noise); the blend rule's
    /// depth. Magnitudes in the reference stars' band; meaningful as photometry only when
    /// `photometric`.
    pub m_lim: Option<f64>,
    /// Where the median calibrated SNR of isolated reference stars falls through 5, per
    /// environment (no linearity assumed).
    pub m5_empirical: PerEnv<Option<f64>>,
    /// That curve: [bin centre magnitude, median calibrated SNR] per 0.5-mag bin of at least 8
    /// stars, per environment (the depth for gamma-encoded frames, and the `suspect` test).
    pub snr_curve: PerEnv<Vec<[f64; 2]>>,
    pub n_stars: usize,
    pub n_refs: usize,
    pub n_refs_measured: usize,
    /// Radius within which a reference star is the object itself, pixels.
    pub self_px: f64,
}

/// One position's measurement.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Measurement {
    pub x: f64,
    pub y: f64,
    /// Background-subtracted aperture sum (data units; not photometry for non-linear sources).
    pub flux: Option<f64>,
    /// Formula sigma sqrt(n·s² + n²·s²/n_ann [+ F/g]).
    pub sigma: Option<f64>,
    /// flux / sigma.
    pub snr: Option<f64>,
    /// snr / χ of the position's environment: the value the status uses.
    pub snr_cal: Option<f64>,
    /// Calibrated SNR of the DoG aperture (the confirmation).
    pub snr_dog: Option<f64>,
    /// Centroid offset from the position, pixels (None when the centroid is lost).
    pub dx: Option<f64>,
    pub dy: Option<f64>,
    /// Background level at the position (annulus plane).
    pub bg: Option<f64>,
    /// Aperture pixels.
    pub n_pix: usize,
    pub env: Option<Env>,
    pub status: Status,
    pub flags: u32,
}

// ---------------------------------------------------------------------------------------------
// statistics (fixed order, numpy conventions)

/// Median (the mean of the two middle values for an even count, as numpy); reorders `v`.
fn median(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let hi = *v.select_nth_unstable_by(n / 2, f64::total_cmp).1;
    if n % 2 == 1 {
        hi
    } else {
        let lo = v[..n / 2].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (lo + hi) / 2.0
    }
}

/// Population standard deviation.
fn stdev(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt()
}

/// 1.4826 · median |x − median(x)| of at least 10 values.
fn robust_sd(v: &[f64]) -> Option<f64> {
    if v.len() < 10 {
        return None;
    }
    let mut a = v.to_vec();
    let m = median(&mut a);
    let mut d: Vec<f64> = v.iter().map(|x| (x - m).abs()).collect();
    Some(1.4826 * median(&mut d))
}

/// Percentile with linear interpolation (numpy's default).
fn percentile(v: &[f64], q: f64) -> f64 {
    let mut a = v.to_vec();
    a.sort_by(f64::total_cmp);
    if a.is_empty() {
        return f64::NAN;
    }
    let p = q / 100.0 * (a.len() - 1) as f64;
    let i = p.floor() as usize;
    if i + 1 >= a.len() {
        return a[a.len() - 1];
    }
    a[i] + (a[i + 1] - a[i]) * (p - i as f64)
}

/// Median, standard deviation and count after 3σ clipping (MAD-seeded, at least `floor`;
/// stops when nothing changes or fewer than 8 values would remain).
fn clipped_median(v: &[f64], floor: f64) -> (f64, f64, usize) {
    let mut keep = vec![true; v.len()];
    let mut kept: Vec<f64> = v.to_vec();
    for _ in 0..5 {
        let m = median(&mut kept.clone());
        let mut d: Vec<f64> = kept.iter().map(|x| (x - m).abs()).collect();
        let mad = 1.4826 * median(&mut d);
        let thr = 3.0 * mad.max(floor);
        let new: Vec<bool> = v.iter().map(|x| (x - m).abs() < thr).collect();
        if new == keep || new.iter().filter(|&&k| k).count() < 8 {
            break;
        }
        keep = new;
        kept = v
            .iter()
            .zip(&keep)
            .filter(|(_, k)| **k)
            .map(|(x, _)| *x)
            .collect();
    }
    (median(&mut kept.clone()), stdev(&kept), kept.len())
}

/// Median and standard deviation after five rounds of 3σ clipping (MAD-seeded, at least
/// `floor`; removed values stay removed).
fn clipped_stats(v: &[f64], floor: f64) -> (f64, f64) {
    let mut kept: Vec<f64> = v.to_vec();
    for _ in 0..5 {
        if kept.is_empty() {
            break;
        }
        let m = median(&mut kept.clone());
        let mut d: Vec<f64> = kept.iter().map(|x| (x - m).abs()).collect();
        let thr = 3.0 * (1.4826 * median(&mut d)).max(floor);
        kept.retain(|x| (x - m).abs() <= thr);
    }
    (median(&mut kept.clone()), stdev(&kept))
}

/// Theil–Sen slope of y on x with Sen's (1968) 95 % interval, ties corrected, as SciPy's
/// `theilslopes`: (slope, low, high).
fn theil_sen(x: &[f64], y: &[f64]) -> Option<(f64, f64, f64)> {
    let n = x.len();
    let mut s = Vec::with_capacity(n * n.saturating_sub(1) / 2);
    for i in 0..n {
        for j in 0..n {
            let dx = x[i] - x[j];
            if dx > 0.0 {
                s.push((y[i] - y[j]) / dx);
            }
        }
    }
    if s.is_empty() {
        return None;
    }
    let nt = s.len();
    let slope = median(&mut s);
    let reps = |v: &[f64]| {
        let mut a = v.to_vec();
        a.sort_by(f64::total_cmp);
        let mut sum = 0.0;
        let mut i = 0;
        while i < a.len() {
            let mut j = i + 1;
            while j < a.len() && a[j] == a[i] {
                j += 1;
            }
            let k = (j - i) as f64;
            if j - i > 1 {
                sum += k * (k - 1.0) * (2.0 * k + 5.0);
            }
            i = j;
        }
        sum
    };
    let nf = n as f64;
    let sigsq = (nf * (nf - 1.0) * (2.0 * nf + 5.0) - reps(x) - reps(y)) / 18.0;
    // norm.ppf(0.975)
    let z = 1.959_963_984_540_054;
    let sigma = sigsq.max(0.0).sqrt();
    let ru = (((nt as f64 + z * sigma) / 2.0).round_ties_even() as usize).min(nt - 1);
    let rl = (((nt as f64 - z * sigma) / 2.0).round_ties_even() as isize - 1).max(0) as usize;
    let lo = *s.select_nth_unstable_by(rl.min(nt - 1), f64::total_cmp).1;
    let hi = *s.select_nth_unstable_by(ru, f64::total_cmp).1;
    Some((slope, lo, hi))
}

/// Ranks starting at 1, ties averaged.
fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]).then(a.cmp(&b)));
    let mut r = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && v[idx[j]] == v[idx[i]] {
            j += 1;
        }
        let avg = (i + j + 1) as f64 / 2.0;
        for &k in &idx[i..j] {
            r[k] = avg;
        }
        i = j;
    }
    r
}

/// Spearman rank correlation.
fn spearman(x: &[f64], y: &[f64]) -> f64 {
    let (rx, ry) = (ranks(x), ranks(y));
    let n = rx.len() as f64;
    let (mx, my) = (rx.iter().sum::<f64>() / n, ry.iter().sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in rx.iter().zip(&ry) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx) * (a - mx);
        syy += (b - my) * (b - my);
    }
    sxy / (sxx * syy).sqrt()
}

/// Solve a small dense system by Gaussian elimination with partial pivoting.
fn solve<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for c in 0..N {
        let p = (c..N).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        let pivot = a[p][c].abs();
        if pivot.is_nan() || pivot <= 1e-300 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        let top = a[c];
        for r in c + 1..N {
            let f = a[r][c] / top[c];
            for (v, t) in a[r][c..].iter_mut().zip(&top[c..]) {
                *v -= f * t;
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = [0.0; N];
    for c in (0..N).rev() {
        let mut s = b[c];
        for k in c + 1..N {
            s -= a[c][k] * x[k];
        }
        x[c] = s / a[c][c];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Deterministic uniform numbers (splitmix64).
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ---------------------------------------------------------------------------------------------
// points in cells (detections, reference stars)

/// Points with a value (flux or magnitude) in square cells over the frame plus a margin, for
/// radius queries. Points beyond the margin are left out.
struct Points {
    p: Vec<[f64; 3]>,
    x0: f64,
    y0: f64,
    cell: f64,
    nx: usize,
    ny: usize,
    start: Vec<u32>,
    items: Vec<u32>,
}

impl Points {
    fn new(p: Vec<[f64; 3]>, w: usize, h: usize, margin: f64, cell: f64) -> Self {
        let (x0, y0) = (-margin, -margin);
        let nx = (((w as f64 + 2.0 * margin) / cell).ceil() as usize).max(1);
        let ny = (((h as f64 + 2.0 * margin) / cell).ceil() as usize).max(1);
        let cell_of = |q: &[f64; 3]| -> Option<usize> {
            let (cx, cy) = ((q[0] - x0) / cell, (q[1] - y0) / cell);
            (cx >= 0.0 && cy >= 0.0 && (cx as usize) < nx && (cy as usize) < ny)
                .then(|| cy as usize * nx + cx as usize)
        };
        let mut start = vec![0u32; nx * ny + 1];
        for q in &p {
            if let Some(c) = cell_of(q) {
                start[c + 1] += 1;
            }
        }
        for c in 0..nx * ny {
            start[c + 1] += start[c];
        }
        let mut fill = start.clone();
        let mut items = vec![0u32; start[nx * ny] as usize];
        for (i, q) in p.iter().enumerate() {
            if let Some(c) = cell_of(q) {
                items[fill[c] as usize] = i as u32;
                fill[c] += 1;
            }
        }
        Self {
            p,
            x0,
            y0,
            cell,
            nx,
            ny,
            start,
            items,
        }
    }

    /// Visit the points within `r` of (x, y): (index, squared distance), cell by cell.
    fn within(&self, x: f64, y: f64, r: f64, mut f: impl FnMut(usize, f64)) {
        let span = |c: f64, o: f64, n: usize| -> Option<(usize, usize)> {
            let a = ((c - r - o) / self.cell).floor();
            let b = ((c + r - o) / self.cell).floor();
            if !(b >= 0.0 && a < n as f64) {
                return None;
            }
            Some((a.max(0.0) as usize, (b as usize).min(n - 1)))
        };
        let (Some((cx0, cx1)), Some((cy0, cy1))) =
            (span(x, self.x0, self.nx), span(y, self.y0, self.ny))
        else {
            return;
        };
        let r2 = r * r;
        for cy in cy0..=cy1 {
            for cx in cx0..=cx1 {
                let c = cy * self.nx + cx;
                for &i in &self.items[self.start[c] as usize..self.start[c + 1] as usize] {
                    let q = &self.p[i as usize];
                    let d2 = (q[0] - x) * (q[0] - x) + (q[1] - y) * (q[1] - y);
                    if d2 <= r2 {
                        f(i as usize, d2);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// estimators

/// Robust plane c + gx·dx + gy·dy over annulus samples (5 rounds of least squares and 3σ
/// clipping at max(MAD, s/2, floor)), its level moved to the median of the clipped residuals
/// (faint sources skew an annulus upwards; a mean would bias every aperture negative).
struct PlaneFit {
    level: f64,
    gx: f64,
    gy: f64,
    /// Standard deviation of the residuals of the last fit.
    s: f64,
    n_keep: usize,
}

fn plane_fit(dx: &[f64], dy: &[f64], v: &[f64], floor: f64) -> PlaneFit {
    let n = v.len();
    let mut keep = vec![true; n];
    let mut coef = [median(&mut v.to_vec()), 0.0, 0.0];
    let mut s = 0.0;
    for _ in 0..5 {
        if keep.iter().filter(|&&k| k).count() < 8 {
            break;
        }
        let mut a = [[0.0; 3]; 3];
        let mut b = [0.0; 3];
        for i in (0..n).filter(|&i| keep[i]) {
            let row = [1.0, dx[i], dy[i]];
            for r in 0..3 {
                for c in 0..3 {
                    a[r][c] += row[r] * row[c];
                }
                b[r] += row[r] * v[i];
            }
        }
        if let Some(c) = solve(a, b) {
            coef = c;
        }
        let res: Vec<f64> = (0..n)
            .map(|i| v[i] - (coef[0] + coef[1] * dx[i] + coef[2] * dy[i]))
            .collect();
        let rk: Vec<f64> = (0..n).filter(|&i| keep[i]).map(|i| res[i]).collect();
        s = stdev(&rk);
        let m = median(&mut rk.clone());
        let mut d: Vec<f64> = rk.iter().map(|r| (r - m).abs()).collect();
        let mad = 1.4826 * median(&mut d);
        let thr = 3.0 * mad.max(0.5 * s).max(floor);
        let new: Vec<bool> = res.iter().map(|r| r.abs() < thr).collect();
        if new == keep {
            break;
        }
        keep = new;
    }
    let mut rk: Vec<f64> = (0..n)
        .filter(|&i| keep[i])
        .map(|i| v[i] - (coef[0] + coef[1] * dx[i] + coef[2] * dy[i]))
        .collect();
    let shift = if rk.is_empty() { 0.0 } else { median(&mut rk) };
    PlaneFit {
        level: coef[0] + shift,
        gx: coef[1],
        gy: coef[2],
        s,
        n_keep: rk.len(),
    }
}

/// Gaussian-windowed centroid (σ of the frame PSF, factor-2 steps of at most 1 px) started at the
/// position, over (dx, dy, max(v − bg, 0)) of the pixels within [`R_IN`] FWHM. The offset in
/// pixels, or None when there is no positive flux or it wanders beyond `rmax`.
fn centroid(pix: &[[f64; 3]], fwhm: f64, rmax: f64) -> Option<(f64, f64)> {
    let sw = fwhm / 2.3548;
    let sw2 = 2.0 * sw * sw;
    let (mut cx, mut cy) = (0.0f64, 0.0f64);
    for _ in 0..30 {
        let (mut s, mut sx, mut sy) = (0.0, 0.0, 0.0);
        for &[dx, dy, v] in pix {
            let (ex, ey) = (dx - cx, dy - cy);
            let w = libm::exp(-(ex * ex + ey * ey) / sw2) * v;
            s += w;
            sx += w * ex;
            sy += w * ey;
        }
        if s.is_nan() || s <= 0.0 {
            return None;
        }
        let stx = (2.0 * sx / s).clamp(-1.0, 1.0);
        let sty = (2.0 * sy / s).clamp(-1.0, 1.0);
        cx += stx;
        cy += sty;
        if (cx * cx + cy * cy).sqrt() > rmax {
            return None;
        }
        if stx.abs() < 1e-3 && sty.abs() < 1e-3 {
            break;
        }
    }
    Some((cx, cy))
}

/// Plain aperture at a position (spike estimator A).
#[derive(Clone, Copy, Debug, Default)]
struct Plain {
    /// Not measurable: position off the frame, aperture off the frame or masked, or less than
    /// [`MIN_ANNULUS`] of the annulus usable.
    edge: bool,
    /// The centre pixel is masked.
    masked: bool,
    flux: f64,
    sigma: f64,
    /// Background part of sigma (no source Poisson term).
    bgsig: f64,
    level: f64,
    /// Residual noise of the annulus pixels.
    pix: f64,
    n: usize,
    peak: f64,
    /// Fraction of the usable annulus pixels at the clip level.
    sat_ann: f64,
    off: Option<(f64, f64)>,
}

impl Plain {
    fn snr(&self) -> f64 {
        if self.edge {
            f64::NAN
        } else {
            self.flux / self.sigma
        }
    }
}

/// Difference-of-Gaussians blur scales (as `highpass.rs`) and the noise factors of its
/// aperture sum, from the filter's own impulse response and a Gaussian PSF.
#[derive(Clone, Copy, Debug)]
struct Dog {
    s1: f32,
    s2: f32,
    /// Support half-width of the wide blur (3 box radii).
    m: usize,
    /// Noise correlation factor c, Poisson weighting kappa of the aperture sum.
    c: f64,
    kappa: f64,
}

impl Dog {
    fn new(fwhm: f64) -> Self {
        let s1 = ((fwhm / 2.355) as f32).max(0.7);
        let s2 = 3.0 * s1;
        let m = 3 * highpass::box_radius(s2);
        let ra = R_AP_DOG * fwhm;
        let half = m + ra.ceil() as usize + 2;
        let n = 2 * half + 1;
        let dog = |src: &[f32]| -> Vec<f64> {
            let a = highpass::box_blur3(src, n, n, s1);
            let b = highpass::box_blur3(src, n, n, s2);
            a.iter().zip(&b).map(|(p, q)| (p - q) as f64).collect()
        };
        let mut z = vec![0f32; n * n];
        z[half * n + half] = 1.0;
        let k = dog(&z);
        let mut ap = vec![0f32; n * n];
        let mut n_ap = 0usize;
        let sp = fwhm / 2.3548;
        let mut psf = vec![0f64; n * n];
        for j in 0..n {
            for i in 0..n {
                let (dx, dy) = (i as f64 - half as f64, j as f64 - half as f64);
                let r2 = dx * dx + dy * dy;
                if r2 <= ra * ra {
                    ap[j * n + i] = 1.0;
                    n_ap += 1;
                }
                psf[j * n + i] = libm::exp(-r2 / (2.0 * sp * sp));
            }
        }
        let wgt = dog(&ap);
        let norm: f64 = psf.iter().sum();
        let (mut w2, mut k2, mut w2p, mut wp) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..n * n {
            let p = psf[i] / norm;
            w2 += wgt[i] * wgt[i];
            k2 += k[i] * k[i];
            w2p += wgt[i] * wgt[i] * p;
            wp += wgt[i] * p;
        }
        Self {
            s1,
            s2,
            m,
            c: (w2 / (k2 * n_ap as f64)).sqrt(),
            kappa: w2p / wp,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// frame model

/// Everything a measurement needs about a frame, built once ([`FrameModel::new`]).
pub struct FrameModel {
    w: usize,
    h: usize,
    grid: Grid,
    bin: usize,
    lsb: f64,
    fwhm: f64,
    fwhm_source: FwhmSource,
    n_psf: usize,
    level_sky: f64,
    sigma_sky: f64,
    saturation: f64,
    sat_thr: f64,
    linear: bool,
    gain: Option<f64>,
    dog: Dog,
    chi: Chi,
    chi_dog: Chi,
    n_cal: PerEnv<usize>,
    zero_point: Option<ZeroPoint>,
    m_lim: Option<f64>,
    m5_empirical: PerEnv<Option<f64>>,
    /// Calibrated SNR of isolated reference stars per magnitude bin, per environment.
    curve: [Vec<(f64, f64)>; 3],
    self_px: f64,
    /// Detections (x, y, flux).
    stars: Points,
    /// Reference stars (x, y, mag); empty without them.
    refs: Points,
    n_refs_measured: usize,
}

impl FrameModel {
    /// Model a frame. `stars` are the detector's stars (the default attempt; they seed and,
    /// without reference stars, supply the PSF fits, keep calibration apertures empty and
    /// decide blends); `refs` are reference stars in pixels (optional: zero point, depth, PSF
    /// stars, blends); `scale_arcsec` the plate scale (the radius within which a reference star
    /// is the object itself).
    pub fn new(
        img: &GrayImage,
        info: &SourceInfo,
        stars: &[Star],
        refs: &[RefStar],
        scale_arcsec: Option<f64>,
        opts: &FrameOptions,
    ) -> Result<FrameModel, MeasureError> {
        let (w, h) = (img.width as usize, img.height as usize);
        if w < 16 || h < 16 || img.pixels.len() < w * h {
            return Err(error("no_frame", "The image is too small to measure."));
        }
        if let Some(f) = opts.fwhm
            && !(MIN_FWHM..=MAX_FWHM).contains(&f)
        {
            return Err(error(
                "bad_request",
                format!("frame.fwhm must lie in [{MIN_FWHM}, {MAX_FWHM}] px"),
            ));
        }
        let px = &img.pixels[..w * h];
        // background grid of the detector's default attempt
        let eopts = extract::ExtractOptions::for_source(info);
        let bin = extract::bin_factor(w, h, &eopts);
        let lsb = if info.lsb > 0.0 {
            info.lsb
        } else {
            extract::detect_lattice(px)
        };
        let grid = {
            let plane = extract::bin(img, bin);
            if plane.w < 4 || plane.h < 4 {
                return Err(error("no_frame", "The image is too small to measure."));
            }
            Grid::build(&plane, px, w, bin, lsb)
                .ok_or_else(|| error("no_frame", "The image has no usable pixels."))?
        };
        let lsb = lsb as f64;
        // live tiles (at least half their pixels unmasked): sky level and noise from the
        // darkest 30 %
        let nt = grid.nx * grid.ny;
        let bounds = |t: usize| tile_bounds(&grid, t, w, h);
        let live: Vec<bool> = (0..nt)
            .into_par_iter()
            .map(|t| {
                let (x0, y0, x1, y1) = bounds(t);
                let (mut n, mut fin) = (0usize, 0usize);
                for y in (y0..y1).step_by(2) {
                    for x in (x0..x1).step_by(2) {
                        n += 1;
                        fin += px[y * w + x].is_finite() as usize;
                    }
                }
                n > 0 && 2 * fin >= n
            })
            .collect();
        let levels: Vec<f64> = (0..nt)
            .filter(|&t| live[t] && grid.bg[t].is_finite())
            .map(|t| grid.bg[t] as f64)
            .collect();
        if levels.is_empty() {
            return Err(error("no_frame", "The image has no usable pixels."));
        }
        let p30 = percentile(&levels, 30.0);
        let lo: Vec<usize> = (0..nt)
            .filter(|&t| live[t] && (grid.bg[t] as f64) <= p30)
            .collect();
        let level_sky = median(&mut lo.iter().map(|&t| grid.bg[t] as f64).collect::<Vec<_>>());
        let sigma_sky = median(&mut lo.iter().map(|&t| grid.sig_n[t] as f64).collect::<Vec<_>>());
        if !(sigma_sky > 0.0 && level_sky.is_finite()) {
            return Err(error("no_frame", "The image has no measurable sky noise."));
        }
        let saturation = px
            .par_chunks(1 << 16)
            .map(|c| {
                c.iter()
                    .copied()
                    .filter(|v| v.is_finite())
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .reduce(|| f32::NEG_INFINITY, f32::max) as f64;
        let sat_thr = level_sky + SAT_FRACTION * (saturation - level_sky);
        let linear = opts.linear.unwrap_or(info.format == SourceFormat::Fits);

        // detections and reference stars in cells
        let guess = {
            let mut f: Vec<f64> = stars
                .iter()
                .filter(|s| s.snr > PSF_MIN_SNR && !s.saturated && s.fwhm.is_finite())
                .map(|s| s.fwhm)
                .collect();
            let g = median(&mut f);
            if g.is_finite() {
                g.clamp(MIN_FWHM, MAX_FWHM)
            } else {
                3.0
            }
        };
        let reach = |f: f64| (R_OUT + R_AP).max(R_OUT_DOG).max(4.0) * f + 8.0;
        let margin = reach(guess.max(opts.fwhm.unwrap_or(0.0)));
        let cell = (2.0 * guess).max(16.0);
        let det = Points::new(
            stars
                .iter()
                .filter(|s| s.x.is_finite() && s.y.is_finite())
                .map(|s| [s.x, s.y, s.flux])
                .collect(),
            w,
            h,
            margin,
            cell,
        );
        let refs = Points::new(
            refs.iter()
                .filter(|r| r.x.is_finite() && r.y.is_finite() && r.mag.is_finite())
                .map(|r| [r.x, r.y, r.mag])
                .collect(),
            w,
            h,
            margin,
            cell,
        );

        // frame FWHM
        let (fwhm, fwhm_source, n_psf) = match opts.fwhm {
            Some(f) => (f, FwhmSource::Given, 0),
            None => {
                let fits = |g: f64, amp: f64| {
                    psf_fits(img, &det, &refs, g, amp, sigma_sky, sat_thr, N_PSF)
                };
                let first = fits(guess, PSF_AMP);
                let (list, src) = if first.len() >= PSF_MIN_FITS {
                    (first, FwhmSource::Fit)
                } else {
                    let mut f = first.clone();
                    let seed = if f.is_empty() { guess } else { median(&mut f) };
                    (fits(seed, PSF_AMP_RELAXED), FwhmSource::FitRelaxed)
                };
                if list.len() < PSF_MIN_FITS_RELAXED {
                    return Err(error(
                        "no_psf",
                        "Too few bright isolated stars to measure the frame FWHM; give frame.fwhm.",
                    ));
                }
                let n = list.len();
                let f = median(&mut list.clone());
                if !(MIN_FWHM..=MAX_FWHM).contains(&f) {
                    return Err(error(
                        "no_psf",
                        format!(
                            "The fitted frame FWHM ({f:.2} px) is out of range; give frame.fwhm."
                        ),
                    ));
                }
                (f, src, n)
            }
        };
        let gain = if linear {
            ptc_gain(px, w, h, &grid, &live, level_sky, saturation, lsb)
        } else {
            None
        };
        let self_px = match scale_arcsec {
            Some(s) if s > 0.0 && s.is_finite() => SELF_ARCSEC / s,
            _ => SELF_FWHM * fwhm,
        };
        let mut model = FrameModel {
            w,
            h,
            grid,
            bin,
            lsb,
            fwhm,
            fwhm_source,
            n_psf,
            level_sky,
            sigma_sky,
            saturation,
            sat_thr,
            linear,
            gain,
            dog: Dog::new(fwhm),
            chi: Chi::default(),
            chi_dog: Chi::default(),
            n_cal: PerEnv::default(),
            zero_point: None,
            m_lim: None,
            m5_empirical: PerEnv::default(),
            curve: Default::default(),
            self_px,
            stars: det,
            refs,
            n_refs_measured: 0,
        };

        // reference stars: zero point and formula depth (the emptiness cut of the calibration)
        let measured: Vec<usize> = {
            let mut idx: Vec<usize> = (0..model.refs.p.len())
                .filter(|&i| {
                    let q = model.refs.p[i];
                    q[0] >= 0.0 && q[1] >= 0.0 && q[0] <= (w - 1) as f64 && q[1] <= (h - 1) as f64
                })
                .collect();
            idx.sort_by(|&a, &b| {
                model.refs.p[a][2]
                    .total_cmp(&model.refs.p[b][2])
                    .then(a.cmp(&b))
            });
            if idx.len() > MAX_REFS_MEASURED {
                // the brightest half (zero point), an even sample of the rest (the faint end of
                // the SNR curve)
                let half = MAX_REFS_MEASURED / 2;
                let (rest, take) = (idx.len() - half, MAX_REFS_MEASURED - half);
                let sample: Vec<usize> = (0..take).map(|k| idx[half + k * rest / take]).collect();
                idx.truncate(half);
                idx.extend(sample);
            }
            idx
        };
        model.n_refs_measured = measured.len();
        let ref_plain: Vec<Plain> = measured
            .par_iter()
            .map(|&i| model.plain(img, model.refs.p[i][0], model.refs.p[i][1], false))
            .collect();
        // isolated: no reference star brighter than mag + 2.5 within the annulus reach
        let iso: Vec<bool> = measured
            .par_iter()
            .map(|&i| {
                let q = model.refs.p[i];
                let mut ok = true;
                model
                    .refs
                    .within(q[0], q[1], (R_OUT + R_AP) * fwhm, |j, _| {
                        if j != i && model.refs.p[j][2] < q[2] + 2.5 {
                            ok = false;
                        }
                    });
                ok
            })
            .collect();
        // assessable as a reference: measured, unsaturated, not flat, not on a clipped background
        let usable: Vec<bool> = ref_plain
            .iter()
            .map(|a| !a.edge && a.peak < model.sat_thr && !model.is_flat(a) && !model.is_bright(a))
            .collect();
        let (mut zc, mut zi) = (vec![], vec![]);
        for (k, a) in ref_plain.iter().enumerate() {
            if iso[k] && usable[k] && a.snr() >= ZP_MIN_SNR && a.flux > 0.0 {
                zc.push(model.refs.p[measured[k]][2]);
                zi.push(-2.5 * libm::log10(a.flux));
            }
        }
        model.zero_point = fit_zero_point(&zc, &zi);
        let m5_formula = model.zero_point.as_ref().map(|z| model.depth(z.zp, 1.0));

        // noise calibration at random empty apertures, stratified by environment (by the grid
        // background; the measured level decides the environment an aperture counts for)
        let mut rng = SplitMix(CAL_SEED);
        let mut picked: [Vec<[f64; 2]>; 3] = Default::default();
        for _ in 0..N_CAL_CANDIDATES {
            if picked.iter().all(|p| p.len() >= N_CAL) {
                break;
            }
            let (x, y) = (rng.unit() * (w - 1) as f64, rng.unit() * (h - 1) as f64);
            let (ix, iy) = ((x + 0.5) as usize, (y + 0.5) as usize);
            if !px[iy.min(h - 1) * w + ix.min(w - 1)].is_finite() {
                continue;
            }
            let bg = model.grid.at_native(x as f32, y as f32).0 as f64;
            if let Some(e) = Env::of(bg, level_sky, sigma_sky)
                && picked[e.index()].len() < N_CAL
            {
                picked[e.index()].push([x, y]);
            }
        }
        let cal: Vec<[f64; 2]> = picked.concat();
        let cal_m: Vec<(Plain, Option<(f64, f64)>)> = cal
            .par_iter()
            .map(|&[x, y]| {
                let a = model.plain(img, x, y, false);
                let d = if a.edge {
                    None
                } else {
                    model.dog_photometry(img, x, y)
                };
                (a, d)
            })
            .collect();
        let mut snr_a: [Vec<f64>; 3] = Default::default();
        let mut snr_b: [Vec<f64>; 3] = Default::default();
        for (&[x, y], (a, d)) in cal.iter().zip(&cal_m) {
            let mut empty = true;
            model
                .stars
                .within(x, y, CAL_EMPTY * fwhm, |_, _| empty = false);
            if let Some(m5) = m5_formula {
                model.refs.within(x, y, CAL_EMPTY * fwhm, |j, _| {
                    if model.refs.p[j][2] < m5 + 1.0 {
                        empty = false;
                    }
                });
            }
            let snr = a.snr();
            if !(empty && !a.edge && snr.is_finite() && !model.is_flat(a) && !model.is_bright(a)) {
                continue;
            }
            let Some(e) = Env::of(a.level, level_sky, sigma_sky) else {
                continue;
            };
            snr_a[e.index()].push(snr);
            if let Some((f, s)) = d {
                let sb = f / s;
                if sb.is_finite() {
                    snr_b[e.index()].push(sb);
                }
            }
        }
        let chi_of = |v: &[Vec<f64>; 3]| {
            let env = |k: usize| {
                if v[k].len() >= CAL_MIN_ENV {
                    robust_sd(&v[k])
                } else {
                    None
                }
            };
            let all: Vec<f64> = v.concat();
            Chi {
                sky: env(0),
                faint: env(1),
                bright: env(2),
                all: if all.len() >= CAL_MIN {
                    robust_sd(&all)
                } else {
                    None
                },
            }
        };
        model.chi = chi_of(&snr_a);
        model.chi_dog = chi_of(&snr_b);
        model.n_cal = PerEnv {
            sky: snr_a[0].len(),
            faint: snr_a[1].len(),
            bright: snr_a[2].len(),
        };

        // calibrated depth (the blend rule's) and the empirical depth per environment
        if let (Some(z), Some(c)) = (model.zero_point.as_ref(), model.chi.get(Env::Sky)) {
            model.m_lim = Some(model.depth(z.zp, c));
        }
        let mut pts: [Vec<(f64, f64)>; 3] = Default::default();
        for (k, a) in ref_plain.iter().enumerate() {
            if !iso[k] || !usable[k] {
                continue;
            }
            let (Some(e), snr) = (Env::of(a.level, level_sky, sigma_sky), a.snr()) else {
                continue;
            };
            if let Some(c) = model.chi.get(e)
                && snr.is_finite()
            {
                pts[e.index()].push((model.refs.p[measured[k]][2], snr / c));
            }
        }
        model.curve = pts.map(|p| snr_curve(&p));
        model.m5_empirical = PerEnv {
            sky: m5_from_curve(&model.curve[0]),
            faint: m5_from_curve(&model.curve[1]),
            bright: m5_from_curve(&model.curve[2]),
        };
        Ok(model)
    }

    /// The frame FWHM in pixels.
    pub fn fwhm(&self) -> f64 {
        self.fwhm
    }

    /// Fluxes are photometry: a linear source whose zero-point slope (when fitted) is 1.
    pub fn photometric(&self) -> bool {
        self.linear && self.zero_point.as_ref().is_none_or(|z| z.linear)
    }

    pub fn summary(&self) -> ModelSummary {
        ModelSummary {
            width: self.w as u32,
            height: self.h as u32,
            fwhm: self.fwhm,
            fwhm_source: self.fwhm_source,
            n_psf: self.n_psf,
            bin: self.bin,
            lsb: self.lsb,
            level_sky: self.level_sky,
            sigma_sky: self.sigma_sky,
            saturation: self.saturation,
            sat_thr: self.sat_thr,
            linear: self.linear,
            photometric: self.photometric(),
            gain: self.gain,
            chi: self.chi,
            chi_dog: self.chi_dog,
            n_cal: self.n_cal,
            zero_point: self.zero_point.clone(),
            m_lim: self.m_lim,
            m5_empirical: self.m5_empirical,
            snr_curve: {
                let c = |k: usize| self.curve[k].iter().map(|&(m, s)| [m, s]).collect();
                PerEnv {
                    sky: c(0),
                    faint: c(1),
                    bright: c(2),
                }
            },
            n_stars: self.stars.p.len(),
            n_refs: self.refs.p.len(),
            n_refs_measured: self.n_refs_measured,
            self_px: self.self_px,
        }
    }

    /// No noise in the annulus beyond quantisation: interface fill, clipped black or white.
    /// (Compressed JPEG highlights have little noise in luma units but still carry signal, so the
    /// limit is absolute, not relative to the local noise.)
    fn is_flat(&self, a: &Plain) -> bool {
        a.pix.is_nan() || a.pix < (FLAT_LSB * self.lsb).max(FLAT_SKY * self.sigma_sky)
    }

    /// Background at the clip level (blown-out core).
    fn is_bright(&self, a: &Plain) -> bool {
        a.level >= self.level_sky + BRIGHT_LEVEL * (self.sat_thr - self.level_sky)
            || a.sat_ann > BRIGHT_CLIPPED
    }

    /// Magnitude of an SNR-5 point source on the sky through zero point `zp`, with the sky noise
    /// scaled by `chi` (and the source Poisson term of linear sources).
    fn depth(&self, zp: f64, chi: f64) -> f64 {
        let f = self.fwhm;
        let n_a = std::f64::consts::PI * (R_AP * f) * (R_AP * f);
        let n_ann = std::f64::consts::PI * (R_OUT * R_OUT - R_IN * R_IN) * f * f;
        let bg2 = self.sigma_sky * self.sigma_sky * (n_a + n_a * n_a / n_ann);
        let k = SNR_LIMIT * SNR_LIMIT;
        let a = self.gain.map_or(0.0, |g| k / g);
        let f5 = (a + (a * a + 4.0 * k * bg2 * chi * chi).sqrt()) / 2.0;
        zp - 2.5 * libm::log10(f5)
    }

    /// The plain aperture (and, with `centroid`, the windowed centroid) at (x, y).
    fn plain(&self, img: &GrayImage, x: f64, y: f64, with_centroid: bool) -> Plain {
        let (w, h, f) = (self.w, self.h, self.fwhm);
        let px = &img.pixels;
        let mut out = Plain {
            edge: true,
            flux: f64::NAN,
            sigma: f64::NAN,
            bgsig: f64::NAN,
            level: f64::NAN,
            pix: f64::NAN,
            peak: f64::NAN,
            sat_ann: f64::NAN,
            ..Plain::default()
        };
        let (ix, iy) = ((x + 0.5).floor(), (y + 0.5).floor());
        if !(ix >= 0.0 && iy >= 0.0 && ix < w as f64 && iy < h as f64) {
            return out;
        }
        out.masked = !px[iy as usize * w + ix as usize].is_finite();
        let (ra, ri, ro) = (R_AP * f, R_IN * f, R_OUT * f);
        let (ra2, ri2, ro2) = (ra * ra, ri * ri, ro * ro);
        let (mut adx, mut ady, mut av) = (vec![], vec![], vec![]);
        let mut cen: Vec<[f64; 3]> = vec![];
        let (mut total, mut ap_bad, mut peak) = (0usize, false, f64::NEG_INFINITY);
        let (y0, y1) = ((y - ro).floor() as isize, (y + ro).ceil() as isize);
        let (x0, x1) = ((x - ro).floor() as isize, (x + ro).ceil() as isize);
        for py in y0..=y1 {
            let dy = py as f64 - y;
            for qx in x0..=x1 {
                let dx = qx as f64 - x;
                let d2 = dx * dx + dy * dy;
                if d2 > ro2 {
                    continue;
                }
                let inside = qx >= 0 && py >= 0 && (qx as usize) < w && (py as usize) < h;
                let v = if inside {
                    px[py as usize * w + qx as usize] as f64
                } else {
                    f64::NAN
                };
                if d2 >= ri2 {
                    total += 1;
                    if v.is_finite() {
                        adx.push(dx);
                        ady.push(dy);
                        av.push(v);
                    }
                }
                if d2 <= ra2 {
                    if v.is_finite() {
                        peak = peak.max(v);
                    } else {
                        ap_bad = true;
                    }
                }
                if with_centroid && d2 <= ri2 {
                    cen.push([dx, dy, v]);
                }
            }
        }
        if ap_bad || (av.len() as f64) < MIN_ANNULUS * total as f64 {
            return out;
        }
        let fit = plane_fit(&adx, &ady, &av, self.lsb);
        let bg = Bg {
            c: fit.level as f32,
            gx: fit.gx as f32,
            gy: fit.gy as f32,
            x0: x as f32,
            y0: y as f32,
        };
        let (flux, n, clipped) = extract::aperture(px, w, h, x, y, ra, bg);
        if clipped || n == 0 {
            return out;
        }
        let nf = n as f64;
        let s2 = fit.s * fit.s;
        let bgv = nf * s2 + nf * nf * s2 / fit.n_keep.max(1) as f64;
        let var = bgv + self.gain.map_or(0.0, |g| flux.max(0.0) / g);
        out.edge = false;
        out.flux = flux;
        out.sigma = if var > 0.0 { var.sqrt() } else { f64::NAN };
        out.bgsig = bgv.sqrt();
        out.level = fit.level;
        out.pix = fit.s;
        out.n = n;
        out.peak = peak;
        out.sat_ann = av.iter().filter(|&&v| v >= self.sat_thr).count() as f64 / av.len() as f64;
        if with_centroid {
            for c in cen.iter_mut() {
                let b = fit.level + fit.gx * c[0] + fit.gy * c[1];
                c[2] = if c[2].is_finite() {
                    (c[2] - b).max(0.0)
                } else {
                    0.0
                };
            }
            out.off = centroid(&cen, f, 1.5 * R_IN * f);
        }
        out
    }

    /// DoG aperture at (x, y) on a patch around it: (flux, sigma), or None when the aperture
    /// or half the annulus is off the frame or masked. Masked pixels are filled with the grid
    /// background for the blur, and every DoG pixel whose support reaches one is masked.
    fn dog_photometry(&self, img: &GrayImage, x: f64, y: f64) -> Option<(f64, f64)> {
        let (w, h, f) = (self.w as isize, self.h as isize, self.fwhm);
        let ro = R_OUT_DOG * f;
        let (rx0, rx1) = ((x - ro).floor() as isize, (x + ro).ceil() as isize);
        let (ry0, ry1) = ((y - ro).floor() as isize, (y + ro).ceil() as isize);
        let m = self.dog.m as isize;
        let (px0, px1) = ((rx0 - m).max(0), (rx1 + m).min(w - 1));
        let (py0, py1) = ((ry0 - m).max(0), (ry1 + m).min(h - 1));
        if px0 > px1 || py0 > py1 {
            return None;
        }
        let (pw, ph) = ((px1 - px0 + 1) as usize, (py1 - py0 + 1) as usize);
        let mut patch = vec![0f32; pw * ph];
        // prefix sums of masked pixels (only built when there is one)
        let mut nan = vec![];
        for j in 0..ph {
            let row = (py0 as usize + j) * self.w + px0 as usize;
            for i in 0..pw {
                let v = img.pixels[row + i];
                patch[j * pw + i] = if v.is_finite() {
                    v
                } else {
                    if nan.is_empty() {
                        nan = vec![0u32; (pw + 1) * (ph + 1)];
                    }
                    nan[(j + 1) * (pw + 1) + i + 1] = 1;
                    self.grid
                        .at_native((px0 as usize + i) as f32, (py0 as usize + j) as f32)
                        .0
                }
            }
        }
        if !nan.is_empty() {
            for j in 1..=ph {
                for i in 1..=pw {
                    let k = j * (pw + 1) + i;
                    nan[k] += nan[k - 1] + nan[k - pw - 1] - nan[k - pw - 2];
                }
            }
        }
        let masked = |i: usize, j: usize| -> bool {
            if nan.is_empty() {
                return false;
            }
            let mu = self.dog.m;
            let (a0, a1) = (i.saturating_sub(mu), (i + mu).min(pw - 1) + 1);
            let (b0, b1) = (j.saturating_sub(mu), (j + mu).min(ph - 1) + 1);
            let at = |a: usize, b: usize| nan[b * (pw + 1) + a];
            at(a1, b1) + at(a0, b0) - at(a0, b1) - at(a1, b0) > 0
        };
        let g1 = highpass::box_blur3(&patch, pw, ph, self.dog.s1);
        let g2 = highpass::box_blur3(&patch, pw, ph, self.dog.s2);
        let (ra2, ri2, ro2) = (
            (R_AP_DOG * f) * (R_AP_DOG * f),
            (R_IN_DOG * f) * (R_IN_DOG * f),
            ro * ro,
        );
        let (mut ap, mut ann) = (vec![], vec![]);
        let mut total = 0usize;
        for qy in ry0..=ry1 {
            let dy = qy as f64 - y;
            for qx in rx0..=rx1 {
                let dx = qx as f64 - x;
                let d2 = dx * dx + dy * dy;
                let (in_ap, in_ann) = (d2 <= ra2, d2 >= ri2 && d2 <= ro2);
                if !in_ap && !in_ann {
                    continue;
                }
                let v = if qx >= 0 && qy >= 0 && qx < w && qy < h {
                    let (i, j) = ((qx - px0) as usize, (qy - py0) as usize);
                    if masked(i, j) {
                        f64::NAN
                    } else {
                        (g1[j * pw + i] - g2[j * pw + i]) as f64
                    }
                } else {
                    f64::NAN
                };
                if in_ap {
                    if !v.is_finite() {
                        return None;
                    }
                    ap.push(v);
                }
                if in_ann {
                    total += 1;
                    if v.is_finite() {
                        ann.push(v);
                    }
                }
            }
        }
        if ap.is_empty() || (ann.len() as f64) < MIN_ANNULUS * total as f64 {
            return None;
        }
        let (lvl, sd, n_ann) = clipped_median(&ann, self.lsb * 1e-3);
        let flux: f64 = ap.iter().map(|v| v - lvl).sum();
        let nb = ap.len() as f64;
        let c = self.dog.c;
        let bgv =
            c * c * sd * sd * (nb + nb * nb * std::f64::consts::FRAC_PI_2 / n_ann.max(1) as f64);
        let var = bgv
            + self
                .gain
                .map_or(0.0, |g| self.dog.kappa * flux.max(0.0) / g);
        (var > 0.0).then(|| (flux, var.sqrt()))
    }

    /// Whether a target is blended: with reference stars and a depth, another reference star
    /// within [`BLEND_RADIUS`] FWHM (not the object itself) brighter than the object and than
    /// the depth + [`BLEND_DEPTH`]; else a detected star there brighter than the target's own
    /// aperture flux.
    fn blended(&self, t: &Target, flux: f64) -> bool {
        let r = BLEND_RADIUS * self.fwhm;
        if let (false, Some(m5)) = (self.refs.p.is_empty(), self.m_lim) {
            let mut near: Vec<(f64, usize)> = vec![];
            self.refs.within(t.x, t.y, r, |j, d2| near.push((d2, j)));
            near.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let skip = usize::from(near.first().is_some_and(|n| n.0.sqrt() <= self.self_px));
            let lim = t.mag.unwrap_or(m5).min(m5 + BLEND_DEPTH);
            return near[skip..].iter().any(|&(_, j)| self.refs.p[j][2] < lim);
        }
        let mut b = false;
        self.stars.within(t.x, t.y, r, |j, d2| {
            if d2.sqrt() > self.self_px && self.stars.p[j][2] > flux.max(0.0) {
                b = true;
            }
        });
        b
    }

    /// Measure one position.
    fn measure_one(&self, img: &GrayImage, t: &Target) -> Measurement {
        let a = self.plain(img, t.x, t.y, true);
        let d = if a.edge {
            None
        } else {
            self.dog_photometry(img, t.x, t.y)
        };
        let env = (!a.edge)
            .then(|| Env::of(a.level, self.level_sky, self.sigma_sky))
            .flatten();
        let chi = env.and_then(|e| self.chi.get(e));
        let chi_dog = env.and_then(|e| self.chi_dog.get(e));
        let snr = a.snr();
        let snr_cal = chi.map(|c| snr / c);
        let snr_dog = match (d, chi_dog) {
            (Some((f, s)), Some(c)) => Some(f / s / c),
            _ => None,
        };
        let mut fl = 0u32;
        if !self.photometric() {
            fl |= flags::NONLINEAR;
        }
        let mut status = Status::Below;
        let rec_snr = snr_cal.is_some_and(|s| s >= SNR_RECORDED);
        let confirmed = snr_dog.is_some_and(|s| s >= SNR_CONFIRM);
        if snr_cal.is_some_and(|s| s >= SNR_TRACE) {
            status = Status::Trace;
        }
        if rec_snr && !confirmed {
            fl |= flags::UNCONFIRMED;
        }
        if rec_snr && confirmed {
            let near = a
                .off
                .is_some_and(|(dx, dy)| (dx * dx + dy * dy).sqrt() / self.fwhm < OFFSET_MAX);
            if near {
                status = Status::Recorded;
            } else {
                status = Status::Below;
                fl |= flags::OFFSET;
            }
        }
        if !a.edge && d.is_none() {
            fl |= flags::NO_DOG;
        }
        // not assessable
        if a.edge {
            fl |= flags::NEAR_EDGE;
        } else {
            if self.is_flat(&a) || !snr.is_finite() {
                fl |= flags::NO_DATA;
            }
            if self.is_bright(&a) {
                fl |= flags::BRIGHT_BG;
            }
            if a.peak >= self.sat_thr {
                fl |= flags::SATURATED;
            }
            if chi.is_none() {
                fl |= flags::UNCALIBRATED;
            }
        }
        if a.masked {
            fl |= flags::NO_DATA;
        }
        if self.blended(t, if a.edge { 0.0 } else { a.flux }) {
            fl |= flags::BLENDED;
        }
        if fl & flags::NOT_ASSESSABLE != 0 {
            status = Status::NotAssessable;
        }
        if status == Status::Recorded
            && let (Some(e), Some(m)) = (env, t.mag)
            && predicted_snr(&self.curve[e.index()], m).is_some_and(|p| p < SUSPECT_SNR)
        {
            fl |= flags::SUSPECT;
        }
        let fin = |v: f64| v.is_finite().then_some(v);
        Measurement {
            x: t.x,
            y: t.y,
            flux: (!a.edge).then_some(a.flux).and_then(fin),
            sigma: (!a.edge).then_some(a.sigma).and_then(fin),
            snr: fin(snr),
            snr_cal: snr_cal.and_then(fin),
            snr_dog: snr_dog.and_then(fin),
            dx: a.off.map(|o| o.0),
            dy: a.off.map(|o| o.1),
            bg: (!a.edge).then_some(a.level).and_then(fin),
            n_pix: a.n,
            env,
            status,
            flags: fl,
        }
    }
}

/// Native pixel bounds (x0, y0, x1, y1) of grid tile `t`.
fn tile_bounds(grid: &Grid, t: usize, w: usize, h: usize) -> (usize, usize, usize, usize) {
    let b = grid.b;
    let (x0, x1) = grid.ax.range(t % grid.nx);
    let (y0, y1) = grid.ay.range(t / grid.nx);
    (
        (x0 * b).min(w),
        (y0 * b).min(h),
        (x1 * b).min(w),
        (y1 * b).min(h),
    )
}

/// Photon-transfer gain of a linear source: the variance of horizontal pixel differences
/// (σ_diff²) grows with the tile level as level / g, galaxy light included. Theil–Sen slope
/// over at most 1500 tiles below 80 % of the clip range; a gain only when the slope and the
/// lower end of its 95 % interval are positive and the rank correlation exceeds 0.3.
#[allow(clippy::too_many_arguments)]
fn ptc_gain(
    px: &[f32],
    w: usize,
    h: usize,
    grid: &Grid,
    live: &[bool],
    level_sky: f64,
    saturation: f64,
    lsb: f64,
) -> Option<f64> {
    let nt = grid.nx * grid.ny;
    let pts: Vec<Option<(f64, f64)>> = (0..nt)
        .into_par_iter()
        .map(|t| {
            let level = grid.bg[t] as f64;
            if !live[t] || level.is_nan() || level >= level_sky + 0.8 * (saturation - level_sky) {
                return None;
            }
            let (x0, y0, x1, y1) = tile_bounds(grid, t, w, h);
            let mut d = Vec::with_capacity(1100);
            for (r, y) in (y0..y1).step_by(2).enumerate() {
                for x in (x0 + (r & 1)..x1.min(w - 1)).step_by(2) {
                    let (a, b) = (px[y * w + x], px[y * w + x + 1]);
                    if a.is_finite() && b.is_finite() {
                        d.push((b - a) as f64);
                    }
                }
            }
            if d.len() < 64 {
                return None;
            }
            let s = clipped_stats(&d, lsb).1 / std::f64::consts::SQRT_2;
            s.is_finite().then_some((level - level_sky, s * s))
        })
        .collect();
    let mut pts: Vec<(f64, f64)> = pts.into_iter().flatten().collect();
    if pts.len() < 20 {
        return None;
    }
    if pts.len() > 1500 {
        let n = pts.len();
        pts = (0..1500).map(|k| pts[k * n / 1500]).collect();
    }
    let (x, y): (Vec<f64>, Vec<f64>) = pts.into_iter().unzip();
    let (slope, lo, _) = theil_sen(&x, &y)?;
    let rho = spearman(&x, &y);
    (slope > 0.0 && lo > 0.0 && rho > 0.3).then(|| 1.0 / slope)
}

/// Circular Gaussian b + a·exp(−r²/2s²) fitted to an n × n cutout by Levenberg–Marquardt with
/// bounds (a ≥ 0, centre within 3 px of the cutout centre, s in [0.3, 6·s0 + 3]): (FWHM, a, x,
/// y), the centre relative to the cutout centre.
fn gauss_fit(cut: &[f64], n: usize, s0: f64) -> Option<(f64, f64, f64, f64)> {
    let c = (n - 1) as f64 / 2.0;
    let mut border = Vec::with_capacity(4 * n);
    for i in 0..n {
        border.push(cut[i]);
        border.push(cut[(n - 1) * n + i]);
    }
    for i in 0..n {
        border.push(cut[i * n]);
        border.push(cut[i * n + n - 1]);
    }
    let b0 = median(&mut border);
    let a0 = cut.iter().copied().fold(f64::NEG_INFINITY, f64::max) - b0;
    if a0.is_nan() || a0 <= 0.0 {
        return None;
    }
    let lo = [f64::NEG_INFINITY, 0.0, c - 3.0, c - 3.0, 0.3];
    let hi = [
        f64::INFINITY,
        f64::INFINITY,
        c + 3.0,
        c + 3.0,
        6.0 * s0 + 3.0,
    ];
    let clamp = |p: [f64; 5]| -> [f64; 5] { std::array::from_fn(|k| p[k].clamp(lo[k], hi[k])) };
    let cost = |p: &[f64; 5]| -> f64 {
        let mut s = 0.0;
        for j in 0..n {
            for i in 0..n {
                let (dx, dy) = (i as f64 - p[2], j as f64 - p[3]);
                let e = libm::exp(-(dx * dx + dy * dy) / (2.0 * p[4] * p[4]));
                let r = p[0] + p[1] * e - cut[j * n + i];
                s += r * r;
            }
        }
        s
    };
    let mut p = clamp([b0, a0, c, c, s0]);
    let mut cur = cost(&p);
    let mut lambda = 1e-3;
    for _ in 0..200 {
        let mut jtj = [[0.0; 5]; 5];
        let mut jtr = [0.0; 5];
        for j in 0..n {
            for i in 0..n {
                let (dx, dy) = (i as f64 - p[2], j as f64 - p[3]);
                let s2 = p[4] * p[4];
                let e = libm::exp(-(dx * dx + dy * dy) / (2.0 * s2));
                let r = p[0] + p[1] * e - cut[j * n + i];
                let ae = p[1] * e;
                let g = [
                    1.0,
                    e,
                    ae * dx / s2,
                    ae * dy / s2,
                    ae * (dx * dx + dy * dy) / (s2 * p[4]),
                ];
                for a in 0..5 {
                    for b in 0..5 {
                        jtj[a][b] += g[a] * g[b];
                    }
                    jtr[a] += g[a] * r;
                }
            }
        }
        let mut improved = false;
        while lambda < 1e12 {
            let mut m = jtj;
            for (k, row) in m.iter_mut().enumerate() {
                row[k] += lambda * jtj[k][k].max(1e-12);
            }
            let Some(d) = solve(m, jtr.map(|v| -v)) else {
                lambda *= 10.0;
                continue;
            };
            let q = clamp(std::array::from_fn(|k| p[k] + d[k]));
            let next = cost(&q);
            if next < cur {
                let done = cur - next <= 1e-12 * cur.max(1e-300);
                p = q;
                cur = next;
                lambda = (lambda / 10.0).max(1e-12);
                improved = !done;
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    p.iter()
        .all(|v| v.is_finite())
        .then(|| (2.3548 * p[4], p[1], p[2] - c, p[3] - c))
}

/// Gaussian FWHMs of bright isolated unsaturated stars: reference stars by magnitude when
/// there are any (no other within 4 guesses brighter than mag + 3), else detections by flux
/// (none within 4 guesses brighter than flux / 15.85), at most `max` fits whose amplitude
/// exceeds `amp` sky sigmas and whose centre lies within one guess of the star.
#[allow(clippy::too_many_arguments)]
fn psf_fits(
    img: &GrayImage,
    det: &Points,
    refs: &Points,
    guess: f64,
    amp: f64,
    sigma_sky: f64,
    sat_thr: f64,
    max: usize,
) -> Vec<f64> {
    let (w, h) = (img.width as usize, img.height as usize);
    let by_mag = !refs.p.is_empty();
    let src = if by_mag { refs } else { det };
    let mut order: Vec<usize> = (0..src.p.len()).collect();
    if by_mag {
        order.sort_by(|&a, &b| src.p[a][2].total_cmp(&src.p[b][2]).then(a.cmp(&b)));
    } else {
        order.sort_by(|&a, &b| src.p[b][2].total_cmp(&src.p[a][2]).then(a.cmp(&b)));
    }
    let half = (2.5 * guess).ceil() as usize + 2;
    let n = 2 * half + 1;
    let s0 = guess / 2.3548;
    let mut out = vec![];
    // fit in batches, in order, until `max` are accepted (parallel inside a batch)
    let fit_one = |k: usize| -> Option<f64> {
        let q = src.p[k];
        let (ix, iy) = (q[0].round(), q[1].round());
        let hf = half as f64;
        if !(ix >= hf && iy >= hf && ix + hf < w as f64 && iy + hf < h as f64) {
            return None;
        }
        let mut isolated = true;
        src.within(q[0], q[1], 4.0 * guess, |j, _| {
            let brighter = if by_mag {
                src.p[j][2] < q[2] + 3.0
            } else {
                src.p[j][2] > q[2] / 15.85
            };
            if j != k && brighter {
                isolated = false;
            }
        });
        if !isolated {
            return None;
        }
        let (ix, iy) = (ix as usize, iy as usize);
        let mut cut = Vec::with_capacity(n * n);
        for y in iy - half..=iy + half {
            for x in ix - half..=ix + half {
                let v = img.pixels[y * w + x];
                if !v.is_finite() || v as f64 >= sat_thr {
                    return None;
                }
                cut.push(v as f64);
            }
        }
        let (fw, a, ox, oy) = gauss_fit(&cut, n, s0)?;
        let off = (ox + ix as f64 - q[0]).hypot(oy + iy as f64 - q[1]);
        (a > amp * sigma_sky && off < guess).then_some(fw)
    };
    for batch in order.chunks(256) {
        let got: Vec<Option<f64>> = batch.par_iter().map(|&k| fit_one(k)).collect();
        for f in got.into_iter().flatten() {
            if out.len() < max {
                out.push(f);
            }
        }
        if out.len() >= max {
            break;
        }
    }
    out
}

/// Median SNR per 0.5-mag bin (at least 8 stars, bins from 8.0 to 21.5) of (magnitude, SNR)
/// pairs: (bin centre, median), brightest first.
fn snr_curve(pts: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut rows: Vec<(f64, f64)> = vec![];
    let mut b0 = 8.0;
    while b0 < 21.5 {
        let mut s: Vec<f64> = pts
            .iter()
            .filter(|p| p.0 >= b0 && p.0 < b0 + 0.5)
            .map(|p| p.1)
            .collect();
        if s.len() >= 8 {
            rows.push((b0 + 0.25, median(&mut s)));
        }
        b0 += 0.5;
    }
    rows
}

/// Where a [`snr_curve`] falls through [`SNR_LIMIT`], interpolated in log SNR.
fn m5_from_curve(rows: &[(f64, f64)]) -> Option<f64> {
    rows.windows(2).find_map(|r| {
        let ((m0, s0), (m1, s1)) = (r[0], r[1]);
        if !(s0 >= SNR_LIMIT && SNR_LIMIT > s1) {
            return None;
        }
        let t = if s1 > 0.0 {
            (libm::log(s0) - libm::log(SNR_LIMIT)) / (libm::log(s0) - libm::log(s1))
        } else {
            (s0 - SNR_LIMIT) / (s0 - s1)
        };
        Some(m0 + t * (m1 - m0))
    })
}

/// SNR a point source of magnitude `mag` reaches according to a [`snr_curve`]: interpolated in
/// log SNR, held at the bright end and extrapolated along the last segment past the faint end
/// (as the spike's `predicted_snr`); None with fewer than two positive bins.
fn predicted_snr(rows: &[(f64, f64)], mag: f64) -> Option<f64> {
    let r: Vec<(f64, f64)> = rows
        .iter()
        .filter(|r| r.1 > 0.0)
        .map(|r| (r.0, libm::log10(r.1)))
        .collect();
    if r.len() < 2 || !mag.is_finite() {
        return None;
    }
    let n = r.len();
    let y = if mag <= r[0].0 {
        r[0].1
    } else if mag >= r[n - 1].0 {
        let slope = (r[n - 1].1 - r[n - 2].1) / (r[n - 1].0 - r[n - 2].0);
        r[n - 1].1 + slope * (mag - r[n - 1].0)
    } else {
        let k = r.windows(2).position(|w| mag < w[1].0).unwrap_or(n - 2);
        let (a, b) = (r[k], r[k + 1]);
        a.1 + (b.1 - a.1) * (mag - a.0) / (b.0 - a.0)
    };
    Some(libm::pow(10.0, y))
}

/// Least-squares line y = a + b·x: (b, a).
fn line_fit(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
    let n = x.len() as f64;
    if x.len() < 2 {
        return None;
    }
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx) * (a - mx);
    }
    (sxx > 0.0).then(|| (sxy / sxx, my - sxy / sxx * mx))
}

/// Robust zero point of catalogue against instrumental magnitudes (−2.5 log10 flux) of
/// reference stars measured with the same estimator: the slope-1 offset (clipped median, at
/// most 8 rounds of 3σ clipping with a 0.05 mag floor) and the free slope (clipped least
/// squares). `None` with fewer than 5 finite pairs.
pub fn fit_zero_point(cat: &[f64], inst: &[f64]) -> Option<ZeroPoint> {
    let ok: Vec<bool> = cat
        .iter()
        .zip(inst)
        .map(|(c, i)| c.is_finite() && i.is_finite())
        .collect();
    if ok.iter().filter(|&&k| k).count() < 5 {
        return None;
    }
    let d: Vec<f64> = cat.iter().zip(inst).map(|(c, i)| c - i).collect();
    let pick = |v: &[f64], m: &[bool]| -> Vec<f64> {
        v.iter()
            .zip(m)
            .filter(|(_, k)| **k)
            .map(|(x, _)| *x)
            .collect()
    };
    let mut keep = ok.clone();
    for _ in 0..8 {
        let mut dk = pick(&d, &keep);
        let med = median(&mut dk.clone());
        dk.iter_mut().for_each(|x| *x = (*x - med).abs());
        let mad = 1.4826 * median(&mut dk);
        let new: Vec<bool> = d
            .iter()
            .zip(&ok)
            .map(|(x, o)| *o && (x - med).abs() < (3.0 * mad).max(0.05))
            .collect();
        if new == keep || !new.contains(&true) {
            break;
        }
        keep = new;
    }
    let mut dk = pick(&d, &keep);
    let zp = median(&mut dk.clone());
    dk.iter_mut().for_each(|x| *x = (*x - zp).abs());
    let scatter = 1.4826 * median(&mut dk);
    let ck = pick(cat, &keep);
    let mag_range = [
        ck.iter().copied().fold(f64::INFINITY, f64::min),
        ck.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    ];
    // free slope: m_inst = a + b·m_cat
    let mut keep2 = ok.clone();
    let mut fit = None;
    for _ in 0..8 {
        let Some((b, a)) = line_fit(&pick(cat, &keep2), &pick(inst, &keep2)) else {
            break;
        };
        fit = Some((b, a));
        let res: Vec<f64> = cat.iter().zip(inst).map(|(c, i)| i - (a + b * c)).collect();
        let mut rk = pick(&res, &keep2);
        let m = median(&mut rk.clone());
        rk.iter_mut().for_each(|x| *x = (*x - m).abs());
        let mad = 1.4826 * median(&mut rk);
        let new: Vec<bool> = res
            .iter()
            .zip(&ok)
            .map(|(r, o)| *o && r.abs() < (3.0 * mad).max(0.05))
            .collect();
        if new == keep2 || new.iter().filter(|&&k| k).count() < 5 {
            break;
        }
        keep2 = new;
    }
    let se = fit.and_then(|(b, a)| {
        let (xs, ys) = (pick(cat, &keep2), pick(inst, &keep2));
        let n = xs.len() as f64;
        let mx = xs.iter().sum::<f64>() / n;
        let rss: f64 = xs
            .iter()
            .zip(&ys)
            .map(|(x, y)| (y - (a + b * x)) * (y - (a + b * x)))
            .sum();
        let sxx: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
        let se = (rss / (n - 2.0).max(1.0) / sxx).sqrt();
        se.is_finite().then_some(se)
    });
    Some(ZeroPoint {
        n: ck.len(),
        zp,
        scatter,
        slope: fit.map(|f| f.0),
        slope_se: se,
        intercept: fit.map(|f| f.1),
        mag_range,
        linear: fit.is_some_and(|f| (f.0 - 1.0).abs() <= LINEAR_SLOPE_TOL),
    })
}

/// Forced photometry at `targets` (O(targets); parallel natively, the result in input order).
pub fn measure(img: &GrayImage, model: &FrameModel, targets: &[Target]) -> Vec<Measurement> {
    targets
        .par_iter()
        .map(|t| model.measure_one(img, t))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// request / response: what `sk_measure` and `skyshot measure --rows` exchange

/// A position in a request: pixels (`x`, `y`, zero-based) or sky (`ra`, `dec`, degrees, with
/// the request's `wcs`); `mag` is the catalogue magnitude (targets: optional, for the blend
/// rule; reference stars: required).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Point {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ra: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dec: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mag: Option<f64>,
}

/// A measurement request (`sk_measure`, `skyshot measure`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Request {
    /// WCS of the frame (CRPIX zero-based): needed for positions given in ra/dec, and gives the
    /// plate scale.
    pub wcs: Option<Wcs>,
    pub targets: Vec<Point>,
    /// Reference stars with magnitudes (zero point, depth, PSF stars, blends).
    pub refs: Vec<Point>,
    pub frame: FrameOptions,
}

/// Row columns of a [`Response`].
pub const COLUMNS: [&str; 14] = [
    "x", "y", "flux", "sigma", "snr", "snr_cal", "snr_dog", "dx", "dy", "bg", "n_pix", "env",
    "status", "flags",
];

/// One row per target, in request order (see [`COLUMNS`]; unknown values are `null`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Row(
    pub Option<f64>,
    pub Option<f64>,
    pub Option<f64>,
    pub Option<f64>,
    pub Option<f64>,
    pub Option<f64>,
    pub Option<f64>,
    pub Option<f64>,
    pub Option<f64>,
    pub Option<f64>,
    pub usize,
    pub Option<Env>,
    pub Status,
    pub u32,
);

impl From<&Measurement> for Row {
    fn from(m: &Measurement) -> Self {
        Row(
            Some(m.x),
            Some(m.y),
            m.flux,
            m.sigma,
            m.snr,
            m.snr_cal,
            m.snr_dog,
            m.dx,
            m.dy,
            m.bg,
            m.n_pix,
            m.env,
            m.status,
            m.flags,
        )
    }
}

/// Status counts of a response.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Counts {
    pub recorded: usize,
    pub trace: usize,
    pub below: usize,
    pub not_assessable: usize,
}

/// A measurement response.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Response {
    pub model: ModelSummary,
    pub counts: Counts,
    pub columns: [&'static str; 14],
    pub rows: Vec<Row>,
}

/// State kept with an open frame between requests: the content mask applied to it and the
/// model of the last request (rebuilt when the frame options, the reference stars or the WCS
/// change).
#[derive(Default)]
pub struct Session {
    content: Option<Content>,
    model: Option<(Vec<u8>, FrameModel)>,
}

fn check_point(p: &Point, what: &str, need_mag: bool, wcs: bool) -> Result<(), MeasureError> {
    let bad = |m: &str| Err(error("bad_request", format!("{what}: {m}")));
    match (p.x, p.y, p.ra, p.dec) {
        (Some(x), Some(y), None, None) => {
            if !(x.abs() < 1e9 && y.abs() < 1e9) {
                return bad("x and y must be finite pixel positions");
            }
        }
        (None, None, Some(ra), Some(dec)) => {
            if !wcs {
                return bad("ra/dec positions need the request's wcs");
            }
            if !(ra.is_finite() && (-90.0..=90.0).contains(&dec)) {
                return bad("ra must be finite and dec within [-90, 90]");
            }
        }
        _ => return bad("give either x and y or ra and dec"),
    }
    match p.mag {
        Some(m) if !m.is_finite() || m.abs() > 100.0 => bad("mag out of range"),
        None if need_mag => bad("reference stars need mag"),
        _ => Ok(()),
    }
}

/// Pixel position of a request point (None when it does not project).
fn pixel(p: &Point, wcs: Option<&Wcs>) -> Option<[f64; 2]> {
    match (p.x, p.y, p.ra, p.dec, wcs) {
        (Some(x), Some(y), ..) => Some([x, y]),
        (_, _, Some(ra), Some(dec), Some(w)) => engine::sky_to_pixel(w, ra, dec),
        _ => None,
    }
}

/// Serve a request on an open frame: apply the content mask (once per frame), build or reuse
/// the model (`stars` runs the detector's default attempt when a model has to be built), and
/// measure. Never panics on any request.
pub fn run(
    img: &mut GrayImage,
    info: &SourceInfo,
    req: &Request,
    session: &mut Session,
    stars: impl FnOnce(&GrayImage) -> Vec<Star>,
) -> Result<Response, MeasureError> {
    if req.targets.len() > MAX_TARGETS {
        return Err(error(
            "bad_request",
            format!("at most {MAX_TARGETS} targets per request"),
        ));
    }
    if req.refs.len() > MAX_REFS {
        return Err(error(
            "bad_request",
            format!("at most {MAX_REFS} reference stars per request"),
        ));
    }
    let has_wcs = req.wcs.is_some();
    for (i, p) in req.targets.iter().enumerate() {
        check_point(p, &format!("targets[{i}]"), false, has_wcs)?;
    }
    for (i, p) in req.refs.iter().enumerate() {
        check_point(p, &format!("refs[{i}]"), true, has_wcs)?;
    }
    if let Some(w) = &req.wcs {
        let det = w.cd[0][0] * w.cd[1][1] - w.cd[0][1] * w.cd[1][0];
        let finite = [w.ra, w.dec, w.crpix[0], w.crpix[1], det]
            .iter()
            .all(|v| v.is_finite());
        if !finite || det == 0.0 || w.sip.as_ref().is_some_and(|s| s.order > 9) {
            return Err(error("bad_request", "wcs: not a usable WCS"));
        }
        if let Some(s) = &w.sip {
            let terms = crate::types::Sip::terms(s.order).len();
            if s.a.len() != terms || s.b.len() != terms {
                return Err(error(
                    "bad_request",
                    "wcs: SIP coefficients do not match the order",
                ));
            }
        }
    }
    if let Some(c) = &req.frame.content {
        let ok = c
            .include
            .iter()
            .chain(&c.exclude)
            .all(|b| b.iter().all(|v| v.is_finite()));
        if !ok || c.include.len() + c.exclude.len() > 1024 {
            return Err(error(
                "bad_request",
                "frame.content: at most 1024 finite boxes",
            ));
        }
        match &session.content {
            Some(applied) if applied == c => {}
            Some(_) => {
                return Err(error(
                    "content_conflict",
                    "A different content mask was applied to this frame; open it again.",
                ));
            }
            None => {
                apply_content(img, c);
                session.content = Some(c.clone());
                session.model = None;
            }
        }
    }
    let wcs = req.wcs.as_ref();
    let key = serde_json::to_vec(&(&req.frame, &req.refs, wcs))
        .map_err(|e| error("bad_request", e.to_string()))?;
    if session.model.as_ref().is_none_or(|(k, _)| *k != key) {
        session.model = None;
        let refs: Vec<RefStar> = req
            .refs
            .iter()
            .filter_map(|p| {
                let [x, y] = pixel(p, wcs)?;
                Some(RefStar {
                    x,
                    y,
                    mag: p.mag.unwrap_or(f64::NAN),
                })
            })
            .collect();
        let scale = wcs.and_then(|w| engine::scale_arcsec(w, img.width, img.height));
        let found = stars(img);
        let model = FrameModel::new(img, info, &found, &refs, scale, &req.frame)?;
        session.model = Some((key, model));
    }
    let Some((_, model)) = session.model.as_ref() else {
        return Err(error("no_frame", "no model"));
    };
    let targets: Vec<Option<Target>> = req
        .targets
        .iter()
        .map(|p| pixel(p, wcs).map(|[x, y]| Target { x, y, mag: p.mag }))
        .collect();
    let valid: Vec<Target> = targets.iter().flatten().copied().collect();
    let mut measured = measure(img, model, &valid).into_iter();
    let mut counts = Counts::default();
    let rows: Vec<Row> = targets
        .iter()
        .map(|t| match t {
            Some(_) => {
                let m = measured.next().expect("one measurement per valid target");
                match m.status {
                    Status::Recorded => counts.recorded += 1,
                    Status::Trace => counts.trace += 1,
                    Status::Below => counts.below += 1,
                    Status::NotAssessable => counts.not_assessable += 1,
                }
                Row::from(&m)
            }
            None => {
                counts.not_assessable += 1;
                let mut fl = flags::NEAR_EDGE;
                if !model.photometric() {
                    fl |= flags::NONLINEAR;
                }
                Row(
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                    None,
                    Status::NotAssessable,
                    fl,
                )
            }
        })
        .collect();
    Ok(Response {
        model: model.summary(),
        counts,
        columns: COLUMNS,
        rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_matches_numpy() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
        assert!(median(&mut []).is_nan());
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0, 5.0], 30.0), 2.2);
    }

    #[test]
    fn theil_sen_recovers_a_line() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| 2.0 * v + 1.0 + if i % 7 == 0 { 30.0 } else { 0.0 })
            .collect();
        let (s, lo, hi) = theil_sen(&x, &y).unwrap();
        assert!((s - 2.0).abs() < 1e-9, "{s}");
        assert!(lo <= s && s <= hi);
        let line: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
        let down: Vec<f64> = x.iter().map(|v| -v).collect();
        assert!((spearman(&x, &line) - 1.0).abs() < 1e-12);
        assert!((spearman(&x, &down) + 1.0).abs() < 1e-12);
        assert!(spearman(&x, &y) > 0.8);
    }

    #[test]
    fn snr_curve_depth_and_prediction() {
        let rows = [(10.25, 100.0), (10.75, 50.0), (11.25, 10.0), (11.75, 2.0)];
        // falls through 5 between 11.25 (10) and 11.75 (2), log-interpolated
        let m5 = m5_from_curve(&rows).unwrap();
        let want = 11.25 + 0.5 * 2f64.ln() / 5f64.ln();
        assert!((m5 - want).abs() < 1e-9, "{m5}");
        assert!((predicted_snr(&rows, 10.25).unwrap() - 100.0).abs() < 1e-9);
        assert!((predicted_snr(&rows, 9.0).unwrap() - 100.0).abs() < 1e-9);
        let mid = predicted_snr(&rows, 10.5).unwrap();
        assert!((mid - (100.0f64 * 50.0).sqrt()).abs() < 1e-6, "{mid}");
        // past the faint end along the last segment: / 5 per 0.5 mag
        assert!((predicted_snr(&rows, 12.25).unwrap() - 0.4).abs() < 1e-9);
        assert!(predicted_snr(&rows[..1], 10.0).is_none());
        let pts: Vec<(f64, f64)> = (0..40)
            .map(|i| (12.0 + 0.01 * i as f64, i as f64))
            .collect();
        assert_eq!(snr_curve(&pts), vec![(12.25, 19.5)]);
    }

    #[test]
    fn ranks_average_ties() {
        assert_eq!(ranks(&[10.0, 20.0, 20.0, 5.0]), vec![2.0, 3.5, 3.5, 1.0]);
    }

    #[test]
    fn zero_point_slope_and_offset() {
        let cat: Vec<f64> = (0..40).map(|i| 10.0 + 0.1 * i as f64).collect();
        let inst: Vec<f64> = cat.iter().map(|m| m - 24.0).collect();
        let z = fit_zero_point(&cat, &inst).unwrap();
        assert!((z.zp - 24.0).abs() < 1e-9);
        assert!((z.slope.unwrap() - 1.0).abs() < 1e-9 && z.linear);
        let bent: Vec<f64> = cat.iter().map(|m| 0.6 * m - 20.0).collect();
        assert!(!fit_zero_point(&cat, &bent).unwrap().linear);
        assert!(fit_zero_point(&cat[..4], &inst[..4]).is_none());
    }

    #[test]
    fn gaussian_fit_recovers_fwhm() {
        let n = 21;
        let (s, x0, y0) = (1.7, 10.3, 9.6);
        let cut: Vec<f64> = (0..n * n)
            .map(|k| {
                let (dx, dy) = ((k % n) as f64 - x0, (k / n) as f64 - y0);
                5.0 + 100.0 * (-(dx * dx + dy * dy) / (2.0 * s * s)).exp()
            })
            .collect();
        let (fw, a, ox, oy) = gauss_fit(&cut, n, 1.0).unwrap();
        assert!((fw - 2.3548 * s).abs() < 1e-4, "{fw}");
        assert!((a - 100.0).abs() < 1e-3);
        assert!((ox - 0.3).abs() < 1e-5 && (oy + 0.4).abs() < 1e-5);
    }

    #[test]
    fn points_find_neighbours() {
        let p = Points::new(
            vec![
                [1.0, 1.0, 0.0],
                [50.0, 50.0, 1.0],
                [-10.0, 3.0, 2.0],
                [9e9, 0.0, 3.0],
            ],
            100,
            100,
            20.0,
            16.0,
        );
        let mut got = vec![];
        p.within(0.0, 0.0, 12.0, |i, _| got.push(i));
        got.sort();
        assert_eq!(got, vec![0, 2]);
        let mut none = vec![];
        p.within(-500.0, -500.0, 10.0, |i, _| none.push(i));
        assert!(none.is_empty());
    }
}
