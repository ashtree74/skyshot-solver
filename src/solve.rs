//! Blind solver: dominance rungs -> fingerprints -> index probes -> consistency filters ->
//! similarity hypotheses -> Hough voting over (centre cell, log scale, rotation, parity) with
//! distinct-anchor counting -> two-stage verification (depth-matched through dominance levels).

use crate::{
    fingerprint::{KeyVariant, image_gate, image_keys},
    index::Index,
    learned::{ElemCache, Raw},
    select::{Grid2, dominance_px, edge_px, rung_order, rung_radius},
    skymask::SkyMask,
    sphere::{Vec3, angle, gnomonic, radec, tangent_basis, unit},
    types::{Star, StarField, Wcs},
};
use serde::Serialize;
use std::{
    cell::RefCell,
    collections::HashMap,
    f64::consts::{PI, TAU},
    sync::Arc,
    time::Instant,
};

mod refract;
mod wide;
pub use crate::types::Refraction;

/// `SA2_DEBUG` traces on stderr (builds with the `solver-debug` feature only).
#[inline]
pub(crate) fn debug_on() -> bool {
    #[cfg(feature = "solver-debug")]
    {
        std::env::var_os("SA2_DEBUG").is_some()
    }
    #[cfg(not(feature = "solver-debug"))]
    {
        false
    }
}

/// `SA2_UNWEIGHTED`: unweighted refinement fits (an experiment switch; `solver-debug` only).
#[inline]
fn unweighted_on() -> bool {
    #[cfg(feature = "solver-debug")]
    {
        std::env::var_os("SA2_UNWEIGHTED").is_some()
    }
    #[cfg(not(feature = "solver-debug"))]
    {
        false
    }
}

#[derive(Clone, Debug)]
pub struct SolveConfig {
    /// A preliminary constrained search, followed by blind solving if it fails.
    pub hint: Option<crate::hints::SolveHint>,
    pub mode: crate::api::SolveMode,
    pub deep_index: Option<DeepIndex>,
    pub min_fov_deg: f64,
    pub max_fov_deg: f64,
    pub deadline_ms: f64,
    pub max_stars: usize,
    /// target pool size of the first rung
    pub target_pool: usize,
    pub min_pool: usize,
    pub max_pool: usize,
    /// distinct anchors in a bin before stage-1 verification
    pub trigger_anchors: usize,
    /// one-sided tolerance on ln(s * r / R_m). A configuration is processed only at the first
    /// rung where it appears, which may be coarser than the level of its true record; 0.35
    /// rejected ~23 % of true-hit anchors (independent review of the learned-coder study) and
    /// 1.0 (e^1 = 2.7x) keeps them while still discarding gross scale mismatches.
    pub rung_tol: f64,
    /// tolerance (ln) of the anchor dominance check
    pub dom_tol: f64,
    /// tolerance (ln) of the neighbour dominance lower bound s * min(d_members) >= R_m
    pub member_tol: f64,
    pub n_refs: usize,
    pub n_rest: usize,
    /// also use non-dominant anchors (exhaustive pass)
    pub exhaustive: bool,
    pub min_log_odds: f64,
    pub keep_matches: bool,
    /// learned index only: probes per configuration (0 = matched budget, i.e. as many as key v1's
    /// `image_keys` returns for the same configuration). Default 8: in the end-to-end A/B it kept
    /// every solve of the matched budget (one more real frame than key v1) at lower cost.
    pub learned_k: usize,
    /// Ambiguity gate (nats). 0 (default) keeps the historical behaviour: the first verified
    /// hypothesis is returned at once. When > 0 the search continues after the first acceptance
    /// (the remaining anchors of the current rung, then vote bins with unverified evidence,
    /// opposite parity first) for at most `continuation_factor` x the time already spent and
    /// never past `deadline_ms`. A second verified hypothesis that is geometrically inconsistent
    /// with the first (other parity, centre more than two vote cells away, scale more than two
    /// scale bins away or rotation more than two rotation bins away) and whose log-odds come
    /// within this margin of the best makes the result `ambiguous` with no WCS.
    pub min_margin_nats: f64,
    /// Catalogue depth cap (BP magnitude) for verification, refinement and model choice: stars
    /// fainter than this are not expected in the frame (phone lenses). None keeps every indexed
    /// star. Hypothesis generation (keys, votes) is not affected.
    pub mag_limit: Option<f32>,
    /// Time budget of the ambiguity continuation as a fraction of the time to the first
    /// acceptance (0.5 = up to +50 %). Only used when `min_margin_nats` > 0.
    pub continuation_factor: f64,
    /// Parts of the frame where stars are hidden (ground, buildings, the outside of a fisheye's
    /// image circle). Detections and catalogue stars there are ignored and the open area is the
    /// chance area of the tests.
    pub mask: Option<Arc<SkyMask>>,
    /// Window search (`wide`): when the whole-frame search ends unsolved before the deadline,
    /// solve windows of the frame (a fisheye or ultra-wide lens is far closer to the similarity
    /// model inside a window, and a window can leave the foreground out), then extend the
    /// window's solution to the whole frame.
    pub windows: bool,
    /// Verifications already spent on this frame by earlier searches (the windows before this
    /// one); the acceptance thresholds count them as trials.
    pub extra_trials: usize,
    /// Allow a zenithal equidistant (ARC) projection for fields too wide for TAN. Off for API
    /// clients that did not ask for it (they would read the WCS as TAN).
    pub allow_arc: bool,
    /// Local residual registration, explicitly enabled by the local photo panel. API
    /// clients keep the base WCS contract unless they implement the additional grid.
    pub allow_local_grid: bool,
    /// Allow STG and cylindrical panorama models. API clients explicitly opt in.
    pub allow_panorama: bool,
    /// Image pipelines only ([`crate::pipeline`]): estimate a sky mask from the pixels when
    /// `mask` is not given ([`crate::skymask::estimate`]).
    pub auto_mask: bool,
    /// A companion index with coarser levels (the `phone` profile, R_7 = 7.76'): fields of tens
    /// of degrees with only bright stars detected have dominance radii of several degrees, beyond
    /// the coarsest level of the main index. Searched after the main index, for the whole frame
    /// and for each window; the solution is extended on the main index.
    pub wide_index: Option<WideIndex>,
}

/// A shared handle to the companion index of [`SolveConfig::wide_index`].
#[derive(Clone)]
pub struct WideIndex(pub Arc<Index>);

#[derive(Clone)]
pub struct DeepIndex(pub Arc<Index>);

impl std::fmt::Debug for DeepIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "DeepIndex({})", self.0.build_id())
    }
}

impl DeepIndex {
    pub fn companion(main: &std::path::Path) -> anyhow::Result<Option<Self>> {
        crate::index::companion_deep_dir(main)
            .map(|p| {
                let idx = Index::open(&p)?;
                anyhow::ensure!(
                    idx.manifest.version == crate::index::FORMAT_VERSION_DEEP,
                    "{} is not a DEEP companion index",
                    p.display()
                );
                Ok(Self(Arc::new(idx)))
            })
            .transpose()
    }
}

impl std::fmt::Debug for WideIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WideIndex(r7 = {:.2}')", self.0.r7.to_degrees() * 60.0)
    }
}

/// The field-width bounds (deg) of searches on the companion index: what the `phone` profile's
/// levels cover.
pub const WIDE_INDEX_FOV: (f64, f64) = (10.0, 100.0);

impl Default for SolveConfig {
    fn default() -> Self {
        Self {
            hint: None,
            mode: crate::api::SolveMode::Auto,
            deep_index: None,
            min_fov_deg: 0.15,
            allow_local_grid: false,
            max_fov_deg: 75.0,
            deadline_ms: 2000.0,
            max_stars: 6000,
            target_pool: 150,
            min_pool: 10,
            max_pool: 800,
            trigger_anchors: 2,
            rung_tol: 1.0,
            dom_tol: 0.2,
            member_tol: 0.3,
            n_refs: 3,
            n_rest: 5,
            exhaustive: true,
            min_log_odds: 30.0,
            keep_matches: false,
            learned_k: 8,
            min_margin_nats: 0.0,
            mag_limit: None,
            continuation_factor: 0.5,
            mask: None,
            windows: true,
            extra_trials: 0,
            allow_arc: true,
            allow_panorama: true,
            auto_mask: true,
            wide_index: None,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SolveStats {
    /// A position/scale hint was tried before the blind search.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub hint_used: bool,
    /// The hinted search failed and the original blind mode was restored.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub hint_fallback: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub deep_search: bool,
    pub n_stars: usize,
    pub rungs_tried: usize,
    pub anchors: usize,
    pub keys: usize,
    pub probes: usize,
    pub records: usize,
    pub hits: usize,
    pub bins: usize,
    pub verifications: usize,
    pub cheap_rejects: usize,
    pub stage2: usize,
    pub exhaustive_pass: bool,
    /// learned index: configurations that also enumerated the opposite-parity branch
    pub branch2: usize,
    /// time spent generating probe keys (key v1 variants or the learned coder)
    pub keygen_ms: f64,
    pub select_ms: f64,
    pub probe_ms: f64,
    pub verify_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct MatchOut {
    pub x: f64,
    pub y: f64,
    pub ra: f64,
    pub dec: f64,
    pub residual_px: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Solution {
    pub status: String,
    pub width: u32,
    pub height: u32,
    pub wcs: Option<Wcs>,
    pub ra: f64,
    pub dec: f64,
    pub scale_arcsec: f64,
    pub rotation_deg: f64,
    pub parity: i32,
    pub fov_width_deg: f64,
    pub n_matches: usize,
    pub n_catalog: usize,
    pub rms_px: f64,
    pub log_odds: f64,
    pub cluster_anchors: usize,
    pub verify_level: u32,
    pub stats: SolveStats,
    pub matches: Vec<MatchOut>,
    /// Log-odds lead (nats) of the reported hypothesis over the best geometrically inconsistent
    /// verified alternative. NaN when the ambiguity continuation did not run
    /// (`min_margin_nats` = 0, or nothing verified), +inf when it ran and found no alternative,
    /// the measured gap otherwise (below `min_margin_nats` the status is `ambiguous`).
    pub margin_nats: f64,
    /// Distinct verified hypotheses (geometrically consistent re-verifications count once);
    /// 1 for an ordinary solve, 0 when unsolved, >= 2 when ambiguous.
    pub n_candidates: u32,
    /// The window (x0, y0, width, height in pixels) whose blind search identified the field,
    /// when the whole-frame search did not; the WCS was then extended to the whole frame.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window: Option<[f64; 4]>,
    /// Atmospheric refraction measured from the residuals of a wide landscape (zenith and
    /// strength); see [`Refraction`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refraction: Option<Refraction>,
    /// Pixel extent (x0, y0, x1, y1) of the open sky under the mask: where the WCS is fitted.
    /// `fov_width_deg` and the field summary measure the field there (the WCS of a circular
    /// fisheye extrapolated into its black corners would add a meaningless 200 deg).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sky_extent: Option<[f64; 4]>,
}

/// A verified hypothesis kept during the ambiguity continuation.
struct Candidate {
    sol: Solution,
    /// seed hit of the verified bin (hypothesis-space centre, scale, rotation, parity)
    seed: Hit,
}

#[derive(Clone, Copy, Debug)]
struct Hit {
    img_anchor: u32,
    cat_anchor: u32,
    center: Vec3,
    s: f64,
    phi: f64,
    parity: i32,
    next: u32,
}

#[derive(Clone, Debug, Default)]
struct Bin {
    anchors: [u32; 8],
    n_anchors: u8,
    head: u32,
    hits: u32,
    verified_at: u8,
    /// parity of the bin's hypotheses (+1 / -1), set with the first hit
    parity: i8,
}

const NONE: u32 = u32::MAX;

#[inline]
fn hash_u64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[derive(Default)]
struct FastHasher(u64);
impl std::hash::Hasher for FastHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = hash_u64(self.0 ^ b as u64);
        }
    }
    fn write_u64(&mut self, i: u64) {
        self.0 = hash_u64(i);
    }
}
type FastMap<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<FastHasher>>;

struct Ctx<'a> {
    idx: &'a Index,
    catalog_cache: RefCell<crate::index::ConeCache>,
    inlier_grid: RefCell<Option<(usize, Grid2)>>,
    cfg: &'a SolveConfig,
    hint_cone: Option<(Vec3, f64)>,
    pts: Vec<[f64; 2]>,
    sig: Vec<f64>,
    d: Vec<f64>,
    edge: Vec<f64>,
    w: f64,
    h: f64,
    /// Area (px^2) where stars can be seen: the frame minus the mask's blocked cells. The chance
    /// expectations of the tests use it instead of w * h.
    open_area: f64,
    stats: SolveStats,
    hits: Vec<Hit>,
    bins: FastMap<u64, Bin>,
    t0: Instant,
    pa_sin: Vec<f64>,
    pa_cos: Vec<f64>,
    n_refs: usize,
    n_rest: usize,
    /// learned coder: element embeddings of the current anchor
    elem_cache: ElemCache,
    /// ambiguity continuation (`min_margin_nats` > 0): distinct verified hypotheses, in order of
    /// acceptance
    cands: Vec<Candidate>,
    /// elapsed ms at the first acceptance; Some while the continuation runs
    accept_ms: Option<f64>,
}

/// Vote-grid parameters of a scale `s` (rad/px) on a frame of longest side `wmax` px: the
/// log-scale bin `ls` (0.25 wide, offset 64) and the gnomonic cube subdivision exponent `nb`
/// (face-centre cell width 2 / 2^nb rad, about a quarter of the binned field width).
#[inline]
fn vote_grid(s: f64, wmax: f64) -> (u64, u32) {
    let s_arcsec = s * (180.0 / PI * 3600.0);
    let ls = ((s_arcsec.ln() * 4.0).floor() as i64 + 64).clamp(0, 255) as u64;
    let fov = ((ls as f64 - 64.0 + 0.5) * 0.25).exp() / 3600.0 * (PI / 180.0) * wmax;
    let nb = ((8.0 / fov).log2().round().clamp(0.0, 15.0)) as u32;
    (ls, nb)
}

/// Angular width (rad) of one vote cell at scale `s` (rad/px), see [`vote_grid`].
fn vote_cell_rad(s: f64, wmax: f64) -> f64 {
    2.0 / (1u64 << vote_grid(s, wmax).1) as f64
}

/// Geometric consistency of two verified hypotheses (the ambiguity criterion): same parity,
/// centres within two vote cells, scales within two scale bins (0.5 in ln), rotations within
/// two rotation bins (20 deg). Anything else is a different place on the sky.
fn same_hypothesis(a: &Solution, b: &Solution, wmax: f64) -> bool {
    if a.parity != b.parity {
        return false;
    }
    let (sa, sb) = (a.scale_arcsec, b.scale_arcsec);
    if !(sa > 0.0 && sb > 0.0 && sa.is_finite() && sb.is_finite()) {
        return false;
    }
    if (sa / sb).ln().abs() > 0.5 {
        return false;
    }
    let drot = (a.rotation_deg - b.rotation_deg).rem_euclid(360.0);
    if drot.min(360.0 - drot) > 20.0 {
        return false;
    }
    let s = sa.min(sb).to_radians() / 3600.0;
    angle(unit(a.ra, a.dec), unit(b.ra, b.dec)) <= 2.0 * vote_cell_rad(s, wmax)
}

/// A result without geometry (status `unsolved` or `ambiguous`): NaN angles, zero counts.
fn blank_solution(status: &str, width: u32, height: u32) -> Solution {
    Solution {
        status: status.into(),
        width,
        height,
        wcs: None,
        ra: f64::NAN,
        dec: f64::NAN,
        scale_arcsec: f64::NAN,
        rotation_deg: f64::NAN,
        parity: 0,
        fov_width_deg: f64::NAN,
        n_matches: 0,
        n_catalog: 0,
        rms_px: f64::NAN,
        log_odds: 0.0,
        cluster_anchors: 0,
        verify_level: 0,
        stats: SolveStats::default(),
        matches: vec![],
        margin_nats: f64::NAN,
        n_candidates: 0,
        window: None,
        refraction: None,
        sky_extent: None,
    }
}

/// Similarity fit with known parity: w = a z + b over complex numbers.
fn fit_similarity(z: &[(f64, f64)], w: &[(f64, f64)]) -> Option<((f64, f64), (f64, f64))> {
    let n = z.len() as f64;
    if z.len() < 2 {
        return None;
    }
    let zm = (
        z.iter().map(|p| p.0).sum::<f64>() / n,
        z.iter().map(|p| p.1).sum::<f64>() / n,
    );
    let wm = (
        w.iter().map(|p| p.0).sum::<f64>() / n,
        w.iter().map(|p| p.1).sum::<f64>() / n,
    );
    let (mut num_re, mut num_im, mut den) = (0.0, 0.0, 0.0);
    for (zi, wi) in z.iter().zip(w) {
        let (zx, zy) = (zi.0 - zm.0, zi.1 - zm.1);
        let (wx, wy) = (wi.0 - wm.0, wi.1 - wm.1);
        // (w)(z*) = (wx + i wy)(zx - i zy)
        num_re += wx * zx + wy * zy;
        num_im += wy * zx - wx * zy;
        den += zx * zx + zy * zy;
    }
    if den <= 0.0 {
        return None;
    }
    let a = (num_re / den, num_im / den);
    let b = (
        wm.0 - (a.0 * zm.0 - a.1 * zm.1),
        wm.1 - (a.0 * zm.1 + a.1 * zm.0),
    );
    Some((a, b))
}

/// Build a TAN WCS (degrees) from a similarity a (complex, radians per pixel), parity and the
/// pixel that maps to the tangent point `center`.
fn wcs_from_similarity(center: Vec3, a: (f64, f64), p: f64, crpix: [f64; 2]) -> Wcs {
    let (ra, dec) = radec(center);
    // xi = Re(a) dx - Im(a) p dy ; eta = Im(a) dx + Re(a) p dy
    let k = 180.0 / PI;
    Wcs {
        ra,
        dec,
        crpix,
        cd: [[a.0 * k, -a.1 * p * k], [a.1 * k, a.0 * p * k]],
        sip: None,
        proj: crate::types::Proj::Tan,
        native_pole: None,
        local_grid: None,
    }
}

impl<'a> Ctx<'a> {
    fn elapsed_ms(&self) -> f64 {
        self.t0.elapsed().as_secs_f64() * 1e3
    }

    /// The refraction of a solved wide landscape (a landscape mask and a field of 30 deg or
    /// more), from the solution's inlier pairs; see [`refract`].
    fn measure_refraction(&self, wcs: &Wcs, pairs: &[[f64; 4]]) -> Option<Refraction> {
        // window searches (`extra_trials` from the frame's earlier searches, no windows of
        // their own) leave it to the extension over the whole frame
        if (!self.cfg.windows && self.cfg.extra_trials > 0)
            || !self.cfg.mask.as_deref().is_some_and(SkyMask::is_landscape)
            || wcs.width_deg(self.w as u32, self.h as u32) < 30.0
        {
            return None;
        }
        refract::measure(wcs, pairs, self.w, self.h)
    }

    /// Pixel extent (x0, y0, x1, y1) of the open sky under the mask, if there is one.
    fn sky_extent(&self) -> Option<[f64; 4]> {
        self.cfg
            .mask
            .as_deref()
            .map(|m| m.open_bbox(self.w, self.h))
    }

    /// Angular width (deg) of the field under `wcs` along the middle row of the open sky (of the
    /// frame without a mask).
    fn fov_width(&self, wcs: &Wcs) -> f64 {
        match self.sky_extent() {
            Some(e) => {
                let y = (e[1] + e[3] - 1.0) / 2.0;
                wcs.arc_deg([e[0] - 0.5, y], [e[2] - 0.5, y])
            }
            None => wcs.width_deg(self.w as u32, self.h as u32),
        }
    }

    /// ln of the verifications so far on this frame, including earlier searches (windows).
    #[inline]
    fn trials_ln(&self) -> f64 {
        ((self.stats.verifications + self.cfg.extra_trials).max(1) as f64).ln()
    }

    /// Whether a catalogue star projected to `px` can be seen: inside the frame and not masked.
    #[inline]
    fn visible(&self, px: [f64; 2]) -> bool {
        px[0] >= 0.0
            && px[1] >= 0.0
            && px[0] <= self.w - 1.0
            && px[1] <= self.h - 1.0
            && self
                .cfg
                .mask
                .as_deref()
                .is_none_or(|m| !m.is_blocked(px[0], px[1]))
    }

    /// Angular radius (rad) of the frame, plus `margin` px, around `c` = the sky position of
    /// the reference point of `wcs`: the largest angle to points sampled along the border.
    fn frame_radius(&self, wcs: &Wcs, c: Vec3, margin: f64) -> f64 {
        let (x0, y0, x1, y1) = (-margin, -margin, self.w + margin, self.h + margin);
        let mut radius: f64 = 0.0;
        for k in 0..=16 {
            let t = k as f64 / 16.0;
            for q in [
                [x0 + t * (x1 - x0), y0],
                [x0 + t * (x1 - x0), y1],
                [x0, y0 + t * (y1 - y0)],
                [x1, y0 + t * (y1 - y0)],
            ] {
                let sky = wcs.pixel_to_sky(q[0], q[1]);
                radius = radius.max(angle(c, unit(sky[0], sky[1])));
            }
        }
        radius
    }

    /// Largest frame radius (rad) the projection can cover: TAN diverges at 90 deg.
    fn max_frame_radius(&self, wcs: &Wcs, tan_limit: f64) -> f64 {
        match wcs.proj {
            crate::types::Proj::Tan => tan_limit,
            _ => 2.9,
        }
    }

    /// Frame centre and rotation (at the centre) implied by one hit. `dir` is the unit vector
    /// A->R in pixels. Uses complex rotations (no trigonometry except one atan2).
    #[inline]
    fn hypothesis(
        &self,
        a_px: [f64; 2],
        dir: (f64, f64),
        av: Vec3,
        pa_code: u32,
        s: f64,
        p: f64,
    ) -> Option<(Vec3, f64)> {
        let (sp, cp) = (self.pa_sin[pa_code as usize], self.pa_cos[pa_code as usize]);
        // e^{i theta_sky} = sp + i cp ; e^{-i p psi} = conj(dir) for p > 0, dir for p < 0
        let (ex, ey) = if p > 0.0 {
            (dir.0, -dir.1)
        } else {
            (dir.0, dir.1)
        };
        let rc = sp * ex - cp * ey;
        let rs = sp * ey + cp * ex;
        let vx = (self.w - 1.0) / 2.0 - a_px[0];
        let vy = p * ((self.h - 1.0) / 2.0 - a_px[1]);
        let xi = s * (rc * vx - rs * vy);
        let eta = s * (rs * vx + rc * vy);
        let rr2 = xi * xi + eta * eta;
        let k = if rr2 < 0.04 {
            1.0 + rr2 / 3.0
        } else {
            let rr = rr2.sqrt();
            if rr > 1.4 {
                return None;
            }
            rr.tan() / rr
        };
        let (east, north) = tangent_basis(av);
        let c = crate::sphere::normalize([
            av[0] + k * (xi * east[0] + eta * north[0]),
            av[1] + k * (xi * east[1] + eta * north[1]),
            av[2] + k * (xi * east[2] + eta * north[2]),
        ]);
        // image +x direction at A, transported to the centre's tangent plane
        let u = [
            rc * east[0] + rs * north[0],
            rc * east[1] + rs * north[1],
            rc * east[2] + rs * north[2],
        ];
        let (ce, cn) = tangent_basis(c);
        let phi_c = crate::sphere::dot(u, cn).atan2(crate::sphere::dot(u, ce));
        Some((c, phi_c))
    }

    fn bin_key(&self, center: Vec3, s: f64, phi: f64, p: f64) -> u64 {
        // gnomonic cube grid, face-centre cell width 2/2^nb rad ~ 0.25 FOV
        let (ls, nb) = vote_grid(s, self.w.max(self.h));
        let n = 1u64 << nb;
        let (ax, ay, az) = (center[0].abs(), center[1].abs(), center[2].abs());
        let (face, u, v) = if ax >= ay && ax >= az {
            (
                if center[0] > 0.0 { 0 } else { 1 },
                center[1] / ax,
                center[2] / ax,
            )
        } else if ay >= az {
            (
                if center[1] > 0.0 { 2 } else { 3 },
                center[0] / ay,
                center[2] / ay,
            )
        } else {
            (
                if center[2] > 0.0 { 4 } else { 5 },
                center[0] / az,
                center[1] / az,
            )
        };
        let i = (((u + 1.0) * 0.5 * n as f64) as u64).min(n - 1);
        let j = (((v + 1.0) * 0.5 * n as f64) as u64).min(n - 1);
        let cell = (face as u64) << 32 | j << 16 | i;
        let pb = (phi.rem_euclid(TAU) / (10f64).to_radians()) as u64 % 36;
        let par = (p > 0.0) as u64;
        cell ^ ((nb as u64) << 35) ^ (pb << 40) ^ (ls << 46) ^ (par << 54)
    }

    fn add_hit(&mut self, hit: Hit) -> Option<u64> {
        let key = self.bin_key(hit.center, hit.s, hit.phi, hit.parity as f64);
        let hi = self.hits.len() as u32;
        let bin = self.bins.entry(key).or_default();
        let mut h = hit;
        if bin.hits == 0 {
            bin.parity = hit.parity as i8;
        }
        h.next = if bin.hits == 0 { NONE } else { bin.head };
        bin.head = hi;
        bin.hits += 1;
        let present = bin.anchors[..bin.n_anchors as usize].contains(&hit.img_anchor);
        if !present && (bin.n_anchors as usize) < bin.anchors.len() {
            bin.anchors[bin.n_anchors as usize] = hit.img_anchor;
            bin.n_anchors += 1;
        }
        let n = bin.n_anchors as usize;
        let trigger = !present
            && n >= self.cfg.trigger_anchors
            && (bin.verified_at == 0 || n >= bin.verified_at as usize + 2);
        self.hits.push(h);
        if trigger {
            bin.verified_at = n as u8;
            Some(key)
        } else {
            None
        }
    }

    /// Process all keys of one anchor at rung r; returns a verified solution if found.
    fn process_anchor(
        &mut self,
        a: u32,
        nb: &[(f64, u32)],
        r: f64,
        variants: &mut Vec<KeyVariant>,
        seen: &mut std::collections::HashSet<u64, std::hash::BuildHasherDefault<FastHasher>>,
    ) -> Option<Solution> {
        let n_refs = self.n_refs.min(nb.len());
        let min_fov = self.cfg.min_fov_deg.to_radians();
        let max_fov = self.cfg.max_fov_deg.to_radians();
        let wmax = self.w.max(self.h);
        let a_px = self.pts[a as usize];
        let d_a = self.d[a as usize];
        let exact_a = d_a < self.edge[a as usize];
        self.elem_cache.clear();
        for ri in 0..n_refs {
            let ref_id = nb[ri].1;
            let rest: Vec<u32> = nb
                .iter()
                .map(|x| x.1)
                .filter(|&j| j != ref_id)
                .take(self.n_rest)
                .collect();
            if rest.len() < 3 {
                continue;
            }
            for i in 0..rest.len() {
                for j in i + 1..rest.len() {
                    for k in j + 1..rest.len() {
                        let sub = [rest[i], rest[j], rest[k]];
                        let mut ids = [ref_id, sub[0], sub[1], sub[2]];
                        ids[1..].sort_unstable();
                        let sig =
                            hash_u64(((a as u64) << 40) ^ ((ids[0] as u64) << 20) ^ ids[1] as u64)
                                ^ hash_u64(((ids[2] as u64) << 32) | ids[3] as u64);
                        if !seen.insert(sig) {
                            continue;
                        }
                        let sigma = [a, ref_id, sub[0], sub[1], sub[2]]
                            .iter()
                            .map(|&q| self.sig[q as usize])
                            .fold(0.0, f64::max);
                        let r_px = self.pts[ref_id as usize];
                        let min_d = [ref_id, sub[0], sub[1], sub[2]]
                            .iter()
                            .map(|&q| self.d[q as usize])
                            .fold(f64::INFINITY, f64::min);
                        let tk = Instant::now();
                        let nbr_px = sub.map(|q| self.pts[q as usize]);
                        // a learned index with a fixed probe count needs only key v1's gate and
                        // |AR|; the matched budget needs the number of key v1 variants
                        let gate_only = self.idx.coder.is_some() && self.cfg.learned_k > 0;
                        let ar = if gate_only {
                            image_gate(a_px, r_px, nbr_px, sigma)
                        } else {
                            image_keys(a_px, r_px, nbr_px, sigma, variants)
                        };
                        let Some(ar_px) = ar else {
                            self.stats.keygen_ms += tk.elapsed().as_secs_f64() * 1e3;
                            continue;
                        };
                        if let Some(coder) = &self.idx.coder {
                            // learned key: same configuration and gate (key v1 sees no possible
                            // record when image_keys returns None), K probes instead of the
                            // hand-crafted variants
                            let k = if self.cfg.learned_k > 0 {
                                self.cfg.learned_k
                            } else {
                                variants.len()
                            };
                            let rel = |q: u32| {
                                let p = self.pts[q as usize];
                                [p[0] - a_px[0], p[1] - a_px[1]]
                            };
                            let raw = Raw {
                                xy: [rel(ref_id), rel(sub[0]), rel(sub[1]), rel(sub[2])],
                                d: [
                                    d_a,
                                    self.d[ref_id as usize],
                                    self.d[sub[0] as usize],
                                    self.d[sub[1] as usize],
                                    self.d[sub[2] as usize],
                                ],
                                sigma,
                                edge: self.edge[a as usize],
                            };
                            match coder.image_variants(&raw, k, &mut self.elem_cache, variants) {
                                Some(b2) => self.stats.branch2 += b2 as usize,
                                None => {
                                    self.stats.keygen_ms += tk.elapsed().as_secs_f64() * 1e3;
                                    continue;
                                }
                            }
                        }
                        self.stats.keygen_ms += tk.elapsed().as_secs_f64() * 1e3;
                        self.stats.keys += 1;
                        let dir = ((r_px[0] - a_px[0]) / ar_px, (r_px[1] - a_px[1]) / ar_px);
                        let lo_member = (-self.cfg.member_tol).exp();
                        // index loop: `variants` is reused, `self` is borrowed mutably inside
                        #[allow(clippy::needless_range_loop)]
                        for vi in 0..variants.len() {
                            let v = variants[vi];
                            self.stats.probes += 1;
                            for rec in self.idx.probe(v.key) {
                                self.stats.records += 1;
                                let m = rec.level();
                                let rm = self.idx.level_radius(m);
                                let s = rec.ar_rad(self.idx.r7) / ar_px;
                                let fov = s * wmax;
                                if fov < min_fov || fov > max_fov {
                                    continue;
                                }
                                // rung consistency (one-sided): image pool threshold must not exceed R_m
                                if (s * r / rm).ln() > self.cfg.rung_tol {
                                    continue;
                                }
                                // pool members must be dominant at the record's level
                                if s * min_d < rm * lo_member {
                                    continue;
                                }
                                // anchor dominance consistency
                                let dc = rec.dcode();
                                let lo = rm * (dc as f64 / 2.0).exp2();
                                let hi = if dc >= 7 {
                                    f64::INFINITY
                                } else {
                                    rm * ((dc + 1) as f64 / 2.0).exp2()
                                };
                                let sd = s * d_a;
                                let tol = self.cfg.dom_tol.exp();
                                if sd > hi * tol {
                                    continue;
                                }
                                if exact_a && sd < lo / tol {
                                    continue;
                                }
                                let p = if v.flip == rec.flip() { 1.0 } else { -1.0 };
                                let av = self.idx.star(rec.anchor()).v;
                                let Some((center, phi)) =
                                    self.hypothesis(a_px, dir, av, rec.pa_code(), s, p)
                                else {
                                    continue;
                                };
                                if self.hint_cone.is_some_and(|(v, cos_radius)| {
                                    crate::sphere::dot(v, center) < cos_radius
                                }) {
                                    continue;
                                }
                                self.stats.hits += 1;
                                let hit = Hit {
                                    img_anchor: a,
                                    cat_anchor: rec.anchor(),
                                    center,
                                    s,
                                    phi,
                                    parity: p as i32,
                                    next: NONE,
                                };
                                if let Some(bk) = self.add_hit(hit) {
                                    let tv = Instant::now();
                                    let verified = self.verify_bin(bk);
                                    self.stats.verify_ms += tv.elapsed().as_secs_f64() * 1e3;
                                    if let Some((sol, seed)) = verified {
                                        if self.cfg.min_margin_nats <= 0.0 {
                                            return Some(sol);
                                        }
                                        // ambiguity continuation: keep searching this rung
                                        self.accept(sol, seed);
                                        if self.continuation_over() {
                                            return None;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Verify one vote bin: cheap consistency gate, similarity fit, then `verify_wcs`. Returns
    /// the solution with the seed hit of the fitted subset (hypothesis-space geometry).
    fn verify_bin(&mut self, key: u64) -> Option<(Solution, Hit)> {
        let bin = self.bins.get(&key)?.clone();
        // gather the bin's hits and keep the largest tightly consistent subset (cheap gate before
        // any catalogue access): centres, scales and rotations must agree far better than the bin
        let mut all: Vec<Hit> = vec![];
        let mut h = bin.head;
        while h != NONE {
            let hit = self.hits[h as usize];
            all.push(hit);
            h = hit.next;
        }
        let first = *all.first()?;
        // continuation: a bin of an already verified hypothesis would only re-verify it
        if self.continuing() && self.matches_candidate(&first) {
            return None;
        }
        self.stats.verifications += 1;
        let fov_deg = first.s.to_degrees() * self.w.max(self.h);
        let tol_c = (0.03 + 0.002 * fov_deg) * all[0].s * self.w.max(self.h);
        let tol_s = 0.04 + 0.002 * fov_deg;
        let tol_p = (3.0 + 0.1 * fov_deg).to_radians();
        let consistent = |a: &Hit, b: &Hit| -> bool {
            let dphi = (a.phi - b.phi + PI).rem_euclid(TAU) - PI;
            (a.s / b.s).ln().abs() < tol_s
                && dphi.abs() < tol_p
                && angle(a.center, b.center) < tol_c
        };
        let mut best: (usize, usize) = (0, 0); // (distinct anchors, seed index)
        for (i, seed) in all.iter().enumerate() {
            let mut img_seen: Vec<u32> = vec![];
            let mut cat_seen: Vec<u32> = vec![];
            for o in &all {
                if consistent(seed, o)
                    && !img_seen.contains(&o.img_anchor)
                    && !cat_seen.contains(&o.cat_anchor)
                {
                    img_seen.push(o.img_anchor);
                    cat_seen.push(o.cat_anchor);
                }
            }
            if img_seen.len() > best.0 {
                best = (img_seen.len(), i);
            }
        }
        if best.0 < 2 {
            self.stats.cheap_rejects += 1;
            return None;
        }
        let seed = all[best.1];
        let mut pairs: Vec<(u32, u32)> = vec![];
        for o in &all {
            if consistent(&seed, o)
                && !pairs
                    .iter()
                    .any(|&(ia, ca)| ia == o.img_anchor || ca == o.cat_anchor)
            {
                pairs.push((o.img_anchor, o.cat_anchor));
            }
        }
        let hsum = Some(seed);
        let base = hsum?;
        let p = base.parity as f64;
        // similarity fit in the tangent plane at the hypothesis centre
        let (ce, cn) = tangent_basis(base.center);
        let cpx = [(self.w - 1.0) / 2.0, (self.h - 1.0) / 2.0];
        let mut z = vec![];
        let mut wv = vec![];
        for &(ia, ca) in &pairs {
            let q = gnomonic(self.idx.star(ca).v, base.center, ce, cn)?;
            let px = self.pts[ia as usize];
            z.push((px[0] - cpx[0], p * (px[1] - cpx[1])));
            wv.push((q[0], q[1]));
        }
        let (a, b) = fit_similarity(&z, &wv)?;
        let s = a.0.hypot(a.1);
        if !(s.is_finite() && s > 0.0) {
            return None;
        }
        // pixel that maps to (0,0): z0 = -b / a
        let den = a.0 * a.0 + a.1 * a.1;
        let z0 = (
            -(b.0 * a.0 + b.1 * a.1) / den,
            -(b.1 * a.0 - b.0 * a.1) / den,
        );
        let crpix = [cpx[0] + z0.0, cpx[1] + p * z0.1];
        let wcs = wcs_from_similarity(base.center, a, p, crpix);
        self.verify_wcs(wcs, bin.n_anchors as usize)
            .map(|sol| (sol, base))
    }

    // ---- ambiguity continuation (`min_margin_nats` > 0) ----------------------------------

    /// A hypothesis has been accepted and the search is in its bounded continuation.
    #[inline]
    fn continuing(&self) -> bool {
        self.accept_ms.is_some()
    }

    /// The continuation must stop: the ambiguity is already established, its time budget
    /// (`continuation_factor` x time to the first acceptance) is spent, or the deadline passed.
    fn continuation_over(&self) -> bool {
        let Some(t_acc) = self.accept_ms else {
            return false;
        };
        if self.margin().1 < self.cfg.min_margin_nats {
            return true;
        }
        let cap = (t_acc * (1.0 + self.cfg.continuation_factor.max(0.0))).min(self.cfg.deadline_ms);
        self.elapsed_ms() > cap
    }

    /// Hypothesis-space check of a bin's hit against the accepted candidates' seed hits: same
    /// parity, centre within two vote cells, scale within two bins, rotation within two bins.
    fn matches_candidate(&self, hit: &Hit) -> bool {
        let wmax = self.w.max(self.h);
        self.cands.iter().any(|c| {
            let s = &c.seed;
            let dphi = (hit.phi - s.phi + PI).rem_euclid(TAU) - PI;
            hit.parity == s.parity
                && (hit.s / s.s).ln().abs() <= 0.5
                && dphi.abs() <= (20f64).to_radians()
                && angle(hit.center, s.center) <= 2.0 * vote_cell_rad(s.s.min(hit.s), wmax)
        })
    }

    /// Record a verified hypothesis. The first acceptance starts the continuation clock; a
    /// hypothesis consistent with an earlier candidate is the same place on the sky and is
    /// dropped (the first verification of a hypothesis is the one reported, so a clean field
    /// yields exactly the solution the default configuration would).
    fn accept(&mut self, sol: Solution, seed: Hit) {
        if self.accept_ms.is_none() {
            self.accept_ms = Some(self.elapsed_ms());
        }
        let wmax = self.w.max(self.h);
        if self
            .cands
            .iter()
            .any(|c| same_hypothesis(&c.sol, &sol, wmax))
        {
            return;
        }
        if debug_on() {
            eprintln!(
                "  candidate {}: centre=({:.3},{:.3}) scale={:.3}\" parity={} log_odds={:.1} at {:.1} ms",
                self.cands.len() + 1,
                sol.ra,
                sol.dec,
                sol.scale_arcsec,
                sol.parity,
                sol.log_odds,
                self.elapsed_ms()
            );
        }
        self.cands.push(Candidate { sol, seed });
    }

    /// (index of the best candidate, its log-odds lead over the runner-up; +inf with one
    /// candidate, NaN with none). Ties keep the earlier acceptance.
    fn margin(&self) -> (usize, f64) {
        let mut best = 0usize;
        let mut second = f64::NEG_INFINITY;
        for (i, c) in self.cands.iter().enumerate() {
            if i == 0 {
                continue;
            }
            let lo = c.sol.log_odds;
            if lo > self.cands[best].sol.log_odds {
                second = self.cands[best].sol.log_odds;
                best = i;
            } else if lo > second {
                second = lo;
            }
        }
        match self.cands.len() {
            0 => (0, f64::NAN),
            1 => (0, f64::INFINITY),
            _ => (best, self.cands[best].sol.log_odds - second),
        }
    }

    /// After the current rung: verify vote bins that gathered distinct anchors since their last
    /// verification (or never reached the trigger while a competitor did), opposite parity
    /// first, then by evidence. Bounded by `continuation_over`. Deterministic order.
    fn sweep_pending_bins(&mut self) {
        let Some(best_parity) = self.cands.first().map(|c| c.sol.parity as i8) else {
            return;
        };
        let trigger = self.cfg.trigger_anchors.max(1) as u8;
        let mut pending: Vec<(bool, u8, u64)> = self
            .bins
            .iter()
            .filter(|(_, b)| b.n_anchors >= trigger && b.n_anchors > b.verified_at)
            .map(|(k, b)| (b.parity == best_parity, b.n_anchors, *k))
            .collect();
        // opposite parity first (false < true), then more anchors, then key for determinism
        pending.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)));
        if debug_on() {
            let hits_opp = self
                .hits
                .iter()
                .filter(|h| h.parity as i8 != best_parity)
                .count();
            let multi = |same: bool| {
                self.bins
                    .values()
                    .filter(|b| (b.parity == best_parity) == same && b.n_anchors >= 2)
                    .count()
            };
            eprintln!(
                "  continuation: {} hits ({hits_opp} opposite parity), {} bins ({} / {} with >= 2 anchors, same / opposite parity), {} pending at {:.1} ms",
                self.hits.len(),
                self.bins.len(),
                multi(true),
                multi(false),
                pending.len(),
                self.elapsed_ms()
            );
        }
        for (_, n, key) in pending {
            if self.continuation_over() {
                break;
            }
            if let Some(bin) = self.bins.get_mut(&key) {
                bin.verified_at = n;
            }
            let tv = Instant::now();
            let verified = self.verify_bin(key);
            self.stats.verify_ms += tv.elapsed().as_secs_f64() * 1e3;
            if let Some((sol, seed)) = verified {
                self.accept(sol, seed);
            }
        }
    }

    /// Final result of the continuation: the best candidate with its margin, or `ambiguous`
    /// (no WCS) when the runner-up is within `min_margin_nats`.
    fn conclude(&mut self) -> Option<Solution> {
        if self.cands.is_empty() {
            return None;
        }
        let (best, margin) = self.margin();
        let n = self.cands.len() as u32;
        if margin < self.cfg.min_margin_nats {
            let b = &self.cands[best].sol;
            if debug_on() {
                eprintln!(
                    "  ambiguous: {n} candidates, best log_odds {:.1}, margin {margin:.1} < {:.1} nats",
                    b.log_odds, self.cfg.min_margin_nats
                );
            }
            let mut s = blank_solution("ambiguous", self.w as u32, self.h as u32);
            s.log_odds = b.log_odds;
            s.margin_nats = margin;
            s.n_candidates = n;
            return Some(s);
        }
        let mut s = self.cands.swap_remove(best).sol;
        s.margin_nats = margin;
        s.n_candidates = n;
        Some(s)
    }

    /// Choose the dominance level for depth-matched verification and run both stages.
    fn verify_wcs(&mut self, wcs0: Wcs, cluster: usize) -> Option<Solution> {
        self.verify_wcs_within(wcs0, cluster, 0.0)
    }

    /// [`Self::verify_wcs`] with a stage-1 matching radius of at least `min_r1` px (tracking:
    /// the error allowed to a predicted WCS).
    fn verify_wcs_within(&mut self, wcs0: Wcs, cluster: usize, min_r1: f64) -> Option<Solution> {
        self.catalog_cache.get_mut().clear();
        *self.inlier_grid.get_mut() = None;
        let s = wcs0.scale_arcsec().to_radians() / 3600.0;
        let area = s * s * self.open_area;
        // finest level with at most ~250 expected stars in the frame (8 needs deep stars)
        let mut mv = if self.idx.has_deep() { 8u32 } else { 7u32 };
        while mv > 1 && area / (PI * self.idx.level_radius(mv).powi(2)) > 250.0 {
            mv -= 1;
        }
        let rm = self.idx.level_radius(mv);
        let thr_px = 0.8 * rm / s;
        let img: Vec<u32> = (0..self.pts.len() as u32)
            .filter(|&i| self.d[i as usize] >= thr_px)
            .collect();
        let cat = self.catalog_in_frame(&wcs0, rm)?;
        if cat.len() < 5 || img.len() < 5 {
            return None;
        }
        // stage 1: binomial test with a generous radius
        let r1 = (0.004 * self.w.max(self.h)).max(4.0).max(min_r1);
        let g = Grid2::build(&self.pts, &img, self.w, self.h, 2.0);
        let (k1, _) = self.count_matches(&g, &cat, r1, &wcs0);
        let p1 = (img.len() as f64 * PI * r1 * r1 / self.open_area).min(0.999);
        let lt = log_binom_tail(cat.len(), k1, p1);
        let dbg = debug_on();
        if dbg {
            let c = wcs0.pixel_to_sky((self.w - 1.0) / 2.0, (self.h - 1.0) / 2.0);
            eprintln!(
                "  stage1: centre=({:.3},{:.3}) scale={:.3}\" mv={mv} cat={} img={} k1={k1} r1={r1:.1} lnP={lt:.1} cluster={cluster}",
                c[0],
                c[1],
                wcs0.scale_arcsec(),
                cat.len(),
                img.len()
            );
        }
        let curved_patch = self.cfg.allow_panorama
            && !self.cfg.windows
            && s.to_degrees() * self.w.max(self.h) > 8.0;
        // Screening uses a similarity seed. Curvature is fitted before the final
        // evidence test, which retains the full correction for every tried model.
        let screen_threshold = -14.0 - if curved_patch { 0.0 } else { self.trials_ln() };
        if lt > screen_threshold || k1 < 5 {
            return None;
        }
        self.stats.stage2 += 1;
        let t_id = Instant::now();
        // Keep ordinary linear identification first. A panorama seed can need
        // curvature already inside its small window, before projection selection.
        let crpix = [(self.w - 1.0) / 2.0, (self.h - 1.0) / 2.0];
        let mut identified = None;
        for order in 1..=if curved_patch { 2 } else { 1 } {
            if order == 2 {
                if self.elapsed_ms() >= self.cfg.deadline_ms {
                    break;
                }
                self.stats.verifications += 1;
            }
            let mut wcs = wcs0.clone();
            let mut radius = r1;
            let mut fitted = true;
            for _ in 0..3 {
                let (_, m) = self.count_matches(&g, &cat, radius, &wcs);
                if m.len() < if order == 2 { 24 } else { 6 } {
                    fitted = false;
                    break;
                }
                let pairs = self.pairs(&m);
                let c = wcs.pixel_to_sky(crpix[0], crpix[1]);
                let candidate = if order == 1 {
                    crate::wcsfit::robust_fit(&pairs, crpix, c, 1)
                } else {
                    crate::wcsfit::select_order(&pairs, crpix, c, 2)
                        .filter(|(w, _, _)| w.sip.as_ref().is_some_and(|sip| sip.order == 2))
                        .map(|(w, keep, _)| (w, keep))
                };
                let Some((nw, keep)) = candidate else {
                    fitted = false;
                    break;
                };
                let r = crate::wcsfit::residuals(&nw, &pairs);
                let inl: Vec<f64> = r
                    .iter()
                    .zip(&keep)
                    .filter(|(_, k)| **k)
                    .map(|(x, _)| *x)
                    .collect();
                let rms = crate::wcsfit::rms(&inl);
                wcs = nw;
                radius = (4.0 * rms).clamp(3.0, radius);
            }
            if !fitted {
                continue;
            }
            // evidence: depth-matched Poisson tail at the final identification radius
            let (k, matched) = self.count_matches(&g, &cat, radius, &wcs);
            let lambda = img.len() as f64 * cat.len() as f64 * PI * radius.powi(2) / self.open_area;
            let ll = poisson_log_odds(k, lambda);
            let (mut xmin, mut xmax, mut ymin, mut ymax) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
            for &(i, _) in &matched {
                let q = self.pts[i as usize];
                xmin = xmin.min(q[0]);
                xmax = xmax.max(q[0]);
                ymin = ymin.min(q[1]);
                ymax = ymax.max(q[1]);
            }
            let spread_ok = (xmax - xmin) >= 0.2 * self.w && (ymax - ymin) >= 0.2 * self.h;
            let threshold = self.cfg.min_log_odds + self.trials_ln();
            if dbg {
                eprintln!(
                    "  stage2: k={k} lambda={lambda:.2} ll={ll:.1} thr={threshold:.1} spread_ok={spread_ok} radius={radius:.2}"
                );
            }
            if ll >= threshold && k >= if order == 2 { 24 } else { 8 } && spread_ok {
                identified = Some((wcs, radius, matched, ll, order == 2));
                break;
            }
        }
        let (wcs, radius, matched, ll, curved_recovery) = identified?;
        // refinement: deeper level, robust TAN+SIP with order chosen on held-out stars
        let id_pairs = self.pairs(&matched);
        let id_rms = crate::wcsfit::rms(&crate::wcsfit::residuals(&wcs, &id_pairs));
        let t_ref = Instant::now();
        let (rwcs, rpairs, rrms, rorder) = self.refine(wcs.clone(), radius);
        if dbg {
            eprintln!(
                "  timing: identification {:.2} ms, refine {:.2} ms",
                (t_ref - t_id).as_secs_f64() * 1e3,
                t_ref.elapsed().as_secs_f64() * 1e3
            );
        }
        let t_choice = Instant::now();
        if dbg {
            eprintln!(
                "  refine: n={} rms={rrms:.3} sip_order={rorder} (identification rms {id_rms:.3}, n={})",
                rpairs.len(),
                id_pairs.len()
            );
        }
        // refinement may only improve on the identification fit; never overturn the identification.
        // Judge both models on the same deep catalogue: unique matches within a fixed tolerance.
        let (wcs, pairs, fit_rms) = if rrms.is_finite() && rpairs.len() >= 8 {
            let n_ref = self.inliers_at(&rwcs, 2.5);
            let n_id = self.inliers_at(&wcs, 2.5);
            if dbg {
                eprintln!(
                    "  model choice: refined {n_ref} vs identification {n_id} inliers at 2.5 px"
                );
            }
            if dbg {
                eprintln!(
                    "  timing: model choice {:.2} ms",
                    t_choice.elapsed().as_secs_f64() * 1e3
                );
            }
            if n_ref as f64 >= 0.9 * n_id as f64 {
                (rwcs, rpairs, rrms)
            } else {
                (wcs, id_pairs, id_rms)
            }
        } else {
            (wcs, id_pairs, id_rms)
        };
        // Identification is not a precision guarantee: a high-scatter refinement
        // can improve on an even worse linear seed while still being unusable.
        if !fit_rms.is_finite() || fit_rms > 4.0 {
            return None;
        }
        if curved_recovery && !wide::mapping_holds(self, &wcs) {
            return None;
        }
        let refraction = self.measure_refraction(&wcs, &pairs);
        let center = wcs.pixel_to_sky(crpix[0], crpix[1]);
        let det = wcs.determinant();
        Some(Solution {
            status: "solved".into(),
            width: self.w as u32,
            height: self.h as u32,
            ra: center[0],
            dec: center[1],
            scale_arcsec: wcs.scale_arcsec(),
            rotation_deg: wcs.rotation_deg(),
            parity: if det > 0.0 { 1 } else { -1 },
            fov_width_deg: self.fov_width(&wcs),
            n_matches: pairs.len(),
            n_catalog: cat.len(),
            rms_px: fit_rms,
            log_odds: ll,
            cluster_anchors: cluster,
            verify_level: mv,
            stats: SolveStats::default(),
            matches: if self.cfg.keep_matches {
                pairs
                    .iter()
                    .map(|p| {
                        let q = wcs.sky_to_pixel(p[2], p[3]).unwrap_or([f64::NAN; 2]);
                        MatchOut {
                            x: p[0],
                            y: p[1],
                            ra: p[2],
                            dec: p[3],
                            residual_px: (q[0] - p[0]).hypot(q[1] - p[1]),
                        }
                    })
                    .collect()
            } else {
                vec![]
            },
            wcs: Some(wcs),
            margin_nats: f64::NAN,
            n_candidates: 1,
            window: None,
            refraction,
            sky_extent: self.sky_extent(),
        })
    }

    /// Fit the tangent point and CD of `prior` to `pairs`, keeping its distortion terms (one lens
    /// across video frames): the pixel positions are undistorted with the prior's SIP, fitted with
    /// TAN, and the SIP is put back. Returns the model and the rms of its inliers (px).
    fn refit_linear(&self, prior: &Wcs, pairs: &[[f64; 4]]) -> Option<(Wcs, f64, Vec<[f64; 4]>)> {
        let crpix = prior.crpix;
        let und: Vec<[f64; 4]> = pairs
            .iter()
            .map(|p| {
                let (u, v) = (p[0] - crpix[0], p[1] - crpix[1]);
                let (du, dv) = prior.sip.as_ref().map_or((0.0, 0.0), |s| s.eval(u, v));
                [crpix[0] + u + du, crpix[1] + v + dv, p[2], p[3]]
            })
            .collect();
        let c = prior.pixel_to_sky(crpix[0], crpix[1]);
        // the prior's projection too (a fisheye stays ARC)
        let (mut w, keep) =
            crate::wcsfit::robust_fit_wmp(&und, &[], crpix, c, 1, 0, prior.projection())?;
        w.sip = prior.sip.clone();
        let inl: Vec<[f64; 4]> = pairs
            .iter()
            .zip(&keep)
            .filter(|(_, k)| **k)
            .map(|(p, _)| *p)
            .collect();
        let rms = crate::wcsfit::rms(&crate::wcsfit::residuals(&w, &inl));
        rms.is_finite().then_some((w, rms, inl))
    }

    /// The pixel shift between a prediction and the image: every catalogue star votes for the
    /// offsets to the detections within `radius`; the peak of the (3 x 3 smoothed, 3 px) vote
    /// histogram must stand well above the uniform background of chance pairs.
    fn offset_vote(
        &self,
        prior: &Wcs,
        cat: &[[f64; 4]],
        g: &Grid2,
        radius: f64,
    ) -> Option<[f64; 2]> {
        const BIN: f64 = 3.0;
        let key = |bx: i64, by: i64| ((bx + (1 << 20)) as u64) << 32 | (by + (1 << 20)) as u64;
        let mut votes: FastMap<u64, (u32, f64, f64)> = FastMap::default();
        let mut nn = vec![];
        let mut total = 0u32;
        for c in cat {
            let Some(p) = prior.sky_to_pixel_fast(c[0], c[1]) else {
                continue;
            };
            g.knn(&self.pts, p, 8, u32::MAX, &mut nn);
            for &(d2, i) in &nn {
                if d2 > radius * radius {
                    break;
                }
                let q = self.pts[i as usize];
                let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
                let e = votes
                    .entry(key((dx / BIN).floor() as i64, (dy / BIN).floor() as i64))
                    .or_insert((0, 0.0, 0.0));
                e.0 += 1;
                e.1 += dx;
                e.2 += dy;
                total += 1;
            }
        }
        let mut best = (0u32, 0.0, 0.0);
        for (&k, _) in votes.iter() {
            let (bx, by) = (
                (k >> 32) as i64 - (1 << 20),
                (k & 0xffff_ffff) as i64 - (1 << 20),
            );
            let mut sum = (0u32, 0.0, 0.0);
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if let Some(v) = votes.get(&key(bx + ox, by + oy)) {
                        sum = (sum.0 + v.0, sum.1 + v.1, sum.2 + v.2);
                    }
                }
            }
            if sum.0 > best.0 {
                best = sum;
            }
        }
        // chance pairs spread over the disc of the radius: expected count in a 3 x 3 window
        let background = total as f64 * 9.0 * BIN * BIN / (PI * radius * radius);
        (best.0 >= 8 && best.0 as f64 >= 5.0 * background.max(1.0))
            .then(|| [best.1 / best.0 as f64, best.2 / best.0 as f64])
    }

    /// Tracking with the distortion held: stage 1 with the prediction, linear re-fits with the
    /// prediction's SIP, the stage-2 evidence test of a blind solve, then one linear re-fit on a
    /// deeper catalogue level for precision. No distortion refinement (a few ms on any field).
    fn verify_tracked(&mut self, prior: &Wcs, search_px: f64) -> Option<Solution> {
        let s = prior.scale_arcsec().to_radians() / 3600.0;
        let area = s * s * self.open_area;
        let mut mv = if self.idx.has_deep() { 8u32 } else { 7u32 };
        while mv > 1 && area / (PI * self.idx.level_radius(mv).powi(2)) > 250.0 {
            mv -= 1;
        }
        let rm = self.idx.level_radius(mv);
        let thr_px = 0.8 * rm / s;
        let img: Vec<u32> = (0..self.pts.len() as u32)
            .filter(|&i| self.d[i as usize] >= thr_px)
            .collect();
        let cat = self.catalog_in_frame(prior, rm)?;
        if cat.len() < 5 || img.len() < 5 {
            return None;
        }
        let g = Grid2::build(&self.pts, &img, self.w, self.h, 2.0);
        // a loose prediction: find the shift first (votes of catalogue-detection offsets), so
        // that stage 1 can keep its tight radius and its power
        let base = (0.004 * self.w.max(self.h)).max(4.0);
        let mut r1 = base.max(search_px);
        let mut shifted = None;
        if search_px > 1.5 * base
            && let Some(o) = self.offset_vote(prior, &cat, &g, search_px)
        {
            let c = [(self.w - 1.0) / 2.0, (self.h - 1.0) / 2.0];
            let sky = prior.pixel_to_sky(c[0] - o[0], c[1] - o[1]);
            shifted = Some(Wcs {
                ra: sky[0],
                dec: sky[1],
                ..prior.clone()
            });
            r1 = 1.5 * base;
        }
        let prior = shifted.as_ref().unwrap_or(prior);
        let (k1, mut m) = self.count_matches(&g, &cat, r1, prior);
        let p1 = (img.len() as f64 * PI * r1 * r1 / self.open_area).min(0.999);
        let lt = log_binom_tail(cat.len(), k1, p1);
        let dbg = debug_on();
        if dbg {
            eprintln!(
                "  track: search {search_px:.1} shifted {} r1 {r1:.1} cat {} img {} k1 {k1} lnP {lt:.1}",
                shifted.is_some(),
                cat.len(),
                img.len()
            );
        }
        if lt > -14.0 || k1 < 5 {
            return None;
        }
        let mut wcs = prior.clone();
        let mut radius = r1;
        for _ in 0..3 {
            if m.len() < 6 {
                return None;
            }
            let (nw, rms, _) = self.refit_linear(prior, &self.pairs(&m))?;
            wcs = nw;
            radius = (4.0 * rms).clamp(2.0, radius);
            m = self.count_matches(&g, &cat, radius, &wcs).1;
        }
        let k = m.len();
        let lambda = img.len() as f64 * cat.len() as f64 * PI * radius.powi(2) / self.open_area;
        let kf = k as f64;
        let ll = if kf > lambda {
            kf * (kf / lambda).ln() - kf + lambda
        } else {
            0.0
        };
        let (mut xmin, mut xmax, mut ymin, mut ymax) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for &(i, _) in &m {
            let q = self.pts[i as usize];
            xmin = xmin.min(q[0]);
            xmax = xmax.max(q[0]);
            ymin = ymin.min(q[1]);
            ymax = ymax.max(q[1]);
        }
        let spread_ok = (xmax - xmin) >= 0.2 * self.w && (ymax - ymin) >= 0.2 * self.h;
        if dbg {
            eprintln!(
                "  track: k {k} lambda {lambda:.2} ll {ll:.1} spread_ok {spread_ok} radius {radius:.2}"
            );
        }
        if ll < self.cfg.min_log_odds || k < 8 || !spread_ok {
            return None;
        }
        let (mut pairs, mut fit_rms) = {
            let p = self.pairs(&m);
            let r = crate::wcsfit::rms(&crate::wcsfit::residuals(&wcs, &p));
            (p, r)
        };
        // precision: the finest level with <= ~1000 stars in the frame, brightest detections
        let mut mr = if self.idx.has_deep() { 8u32 } else { 7u32 };
        while mr > 1 && area / (PI * self.idx.level_radius(mr).powi(2)) > 1000.0 {
            mr -= 1;
        }
        if mr > mv
            && let Some(deep) = self.catalog_in_frame(&wcs, self.idx.level_radius(mr))
        {
            let n_img = (deep.len() * 3).max(50).min(self.pts.len());
            let ids: Vec<u32> = (0..n_img as u32).collect();
            let gd = Grid2::build(&self.pts, &ids, self.w, self.h, 2.0);
            let md = self
                .count_matches(&gd, &deep, (3.0 * fit_rms).clamp(1.5, radius), &wcs)
                .1;
            if md.len() >= 20
                && let Some((dw, drms, dinl)) = self.refit_linear(&wcs, &self.pairs(&md))
                && dinl.len() >= pairs.len()
            {
                wcs = dw;
                fit_rms = drms;
                pairs = dinl;
            }
        }
        let crpix = [(self.w - 1.0) / 2.0, (self.h - 1.0) / 2.0];
        let center = wcs.pixel_to_sky(crpix[0], crpix[1]);
        let det = wcs.determinant();
        Some(Solution {
            status: "solved".into(),
            width: self.w as u32,
            height: self.h as u32,
            ra: center[0],
            dec: center[1],
            scale_arcsec: wcs.scale_arcsec(),
            rotation_deg: wcs.rotation_deg(),
            parity: if det > 0.0 { 1 } else { -1 },
            fov_width_deg: self.fov_width(&wcs),
            n_matches: pairs.len(),
            n_catalog: cat.len(),
            rms_px: fit_rms,
            log_odds: ll,
            cluster_anchors: 0,
            verify_level: mv,
            margin_nats: f64::NAN,
            n_candidates: 1,
            window: None,
            sky_extent: self.sky_extent(),
            refraction: None,
            stats: SolveStats::default(),
            matches: if self.cfg.keep_matches {
                pairs
                    .iter()
                    .map(|p| {
                        let q = wcs.sky_to_pixel(p[2], p[3]).unwrap_or([f64::NAN; 2]);
                        MatchOut {
                            x: p[0],
                            y: p[1],
                            ra: p[2],
                            dec: p[3],
                            residual_px: (q[0] - p[0]).hypot(q[1] - p[1]),
                        }
                    })
                    .collect()
            } else {
                vec![]
            },
            wcs: Some(wcs),
        })
    }

    /// Fit weights 1/sigma^2 of the matched image stars (sigma clamped to [0.15, 2] px).
    fn weights(&self, m: &[(u32, [f64; 2])]) -> Vec<f64> {
        m.iter()
            .map(|&(i, _)| 1.0 / self.sig[i as usize].clamp(0.15, 2.0).powi(2))
            .collect()
    }

    fn pairs(&self, m: &[(u32, [f64; 2])]) -> Vec<[f64; 4]> {
        m.iter()
            .map(|&(i, sky)| {
                [
                    self.pts[i as usize][0],
                    self.pts[i as usize][1],
                    sky[0],
                    sky[1],
                ]
            })
            .collect()
    }

    /// Astrometric refinement against the finest level with <= ~1500 stars in the frame,
    /// matched to the brightest detections. Returns (wcs, inlier pairs, rms, sip order).
    fn refine(&self, wcs0: Wcs, start_radius: f64) -> (Wcs, Vec<[f64; 4]>, f64, u32) {
        let crpix = [(self.w - 1.0) / 2.0, (self.h - 1.0) / 2.0];
        let s = wcs0.scale_arcsec().to_radians() / 3600.0;
        let area = s * s * self.open_area;
        let mut mr = if self.idx.has_deep() { 8u32 } else { 7u32 };
        while mr > 1 && area / (PI * self.idx.level_radius(mr).powi(2)) > 1500.0 {
            mr -= 1;
        }
        let margin = 0.15 * self.w.max(self.h);
        let Some(cat) = self.catalog_in_frame_margin(&wcs0, self.idx.level_radius(mr), margin)
        else {
            return (wcs0, vec![], f64::INFINITY, 0);
        };
        let n_img = (cat.len() * 3).max(50).min(self.pts.len());
        let ids: Vec<u32> = (0..n_img as u32).collect();
        let g = Grid2::build(&self.pts, &ids, self.w, self.h, 2.0);
        let fov = s.to_degrees() * self.w.max(self.h);
        let max_order = if fov > 20.0 {
            5
        } else if fov > 3.0 {
            4
        } else {
            3
        };
        let half_diag = 0.5 * self.w.hypot(self.h);
        // typical separation of catalogue stars in pixels; matching radius must stay well below it
        let sep = (self.open_area / cat.len().max(1) as f64).sqrt();
        let rcap = (0.3 * sep).clamp(1.5, 25.0);
        // radial zones (fractions of the half diagonal); narrow fields are nearly distortion-free
        let zones: &[f64] = if fov > 8.0 {
            &[0.45, 0.75, 1.0, 1.2]
        } else {
            &[1.2, 1.2]
        };
        let mut wcs = wcs0;
        let mut rms = start_radius.clamp(2.0, 8.0) / 4.0;
        let mut best: (Wcs, Vec<[f64; 4]>, f64, u32) = (wcs.clone(), vec![], f64::INFINITY, 0);
        // (SIP order, radial extension degree) of the current model
        let mut model = (1u32, 0u32);
        let mut best_weights = vec![];
        // wide fields: lens-like radial terms (degree 5 and 7) that extrapolate to the corners
        let radial_max = if fov > 8.0 { 7 } else { 0 };
        // after the zones: one order-selection step on converged matches, then a final fit
        let total = zones.len() + 2;
        for step in 0..total {
            let zf = zones[step.min(zones.len() - 1)];
            let zr = zf * half_diag;
            // predicted positions with the current model; keep catalogue stars inside the zone
            let projector = wcs.prepare();
            let sub: Vec<[f64; 4]> = cat
                .iter()
                .filter_map(|c| {
                    let px = projector.sky_to_pixel_fast(c[0], c[1])?;
                    let r = (px[0] - crpix[0]).hypot(px[1] - crpix[1]);
                    (r <= zr
                        && px[0] > -margin
                        && px[1] > -margin
                        && px[0] < self.w + margin
                        && px[1] < self.h + margin
                        && self
                            .cfg
                            .mask
                            .as_deref()
                            .is_none_or(|m| !m.is_blocked(px[0], px[1])))
                    .then_some([c[0], c[1], px[0], px[1]])
                })
                .collect();
            // radius: model scatter plus an extrapolation allowance for the newly added ring
            let extra = if step < zones.len() { 0.004 * zr } else { 0.0 };
            let radius = (4.0 * rms + extra).clamp(1.5, rcap);
            // `sub` was projected with this exact WCS above. Reuse those pixels only
            // within this iteration; every updated WCS must project the catalogue anew.
            let (_, m) = match_projected(
                &self.pts,
                &g,
                sub.iter().map(|c| ([c[0], c[1]], [c[2], c[3]])),
                radius,
            );
            let pairs = self.pairs(&m);
            let wts = if unweighted_on() {
                vec![]
            } else {
                self.weights(&m)
            };
            if pairs.len() < 8 {
                if step == 0 {
                    return best;
                }
                continue;
            }
            let c = wcs.pixel_to_sky(crpix[0], crpix[1]);
            // order allowed by the number of pairs and the zone size
            let cap = match pairs.len() {
                0..=29 => 1,
                30..=59 => 2,
                60..=99 => 3,
                100..=199 => 4,
                _ => 5,
            }
            .min(max_order);
            // largest radial degree the pairs support for a base order `o`; high powers of rho
            // are unconstrained on a small zone (rho^6 <= 0.01 inside 0.45 of the half
            // diagonal) and would explode when extrapolated, so the degree grows with the zone
            let zone_cap = if zf < 0.6 {
                0
            } else if zf < 0.9 {
                5
            } else {
                7
            };
            let radial_for = |o: u32| {
                [7u32, 5]
                    .into_iter()
                    .find(|&r| {
                        r <= radial_max.min(zone_cap)
                            && pairs.len() >= 6 * crate::wcsfit::model_columns(o, r)
                    })
                    .unwrap_or(0)
            };
            let fitted = if step == zones.len() {
                crate::wcsfit::select_model_w(&pairs, &wts, crpix, c, cap, radial_for(1))
                    .map(|(f, k, _, m)| (f, k, m))
            } else if step > zones.len() {
                let o = model.0.min(cap);
                let r = if model.1 > 0 {
                    radial_for(o).min(model.1)
                } else {
                    0
                };
                crate::wcsfit::robust_fit_wm(&pairs, &wts, crpix, c, o, r)
                    .map(|(f, k)| (f, k, (o, r)))
            } else if fov > 8.0 {
                let o = cap.min(3);
                let r = radial_for(o);
                crate::wcsfit::robust_fit_wm(&pairs, &wts, crpix, c, o, r)
                    .map(|(f, k)| (f, k, (o, r)))
            } else {
                crate::wcsfit::robust_fit_w(&pairs, &wts, crpix, c, 1).map(|(f, k)| (f, k, (1, 0)))
            };
            let Some((fit, keep, m)) = fitted else {
                continue;
            };
            let inl: Vec<[f64; 4]> = pairs
                .iter()
                .zip(&keep)
                .filter(|(_, k)| **k)
                .map(|(p, _)| *p)
                .collect();
            let r = crate::wcsfit::rms(&crate::wcsfit::residuals(&fit, &inl));
            if !r.is_finite() {
                continue;
            }
            if debug_on() {
                eprintln!(
                    "  refine step {step}: zone {zf:.2} radius {radius:.1} pairs {} inliers {} model {:?} rms {r:.3}",
                    pairs.len(),
                    inl.len(),
                    m
                );
            }
            model = m;
            let order = fit.sip.as_ref().map(|x| x.order).unwrap_or(1);
            wcs = fit;
            rms = r;
            best_weights = keep
                .iter()
                .enumerate()
                .filter(|(_, k)| **k)
                .map(|(i, _)| wts.get(i).copied().unwrap_or(1.0))
                .collect();
            best = (wcs.clone(), inl, r, order);
        }
        if fov > 8.0
            && best.2 < 2.0
            && let Some(lens) =
                crate::wcsfit::select_lens(&best.1, &best_weights, &best.0, model, self.w, self.h)
        {
            let e = crate::wcsfit::rms(&crate::wcsfit::residuals(&lens, &best.1));
            if e.is_finite() && e <= (1.1 * best.2).max(0.3) {
                best = (lens, best.1, e, 9);
            }
        }
        best
    }

    /// Unique matches of the finest affordable level (<= ~1500 stars) within `tol` px under `wcs`.
    fn inliers_at(&self, wcs: &Wcs, tol: f64) -> usize {
        let s = wcs.scale_arcsec().to_radians() / 3600.0;
        let area = s * s * self.open_area;
        let mut mr = if self.idx.has_deep() { 8u32 } else { 7u32 };
        while mr > 1 && area / (PI * self.idx.level_radius(mr).powi(2)) > 1500.0 {
            mr -= 1;
        }
        let Some(cat) = self.catalog_in_frame(wcs, self.idx.level_radius(mr)) else {
            return 0;
        };
        let n_img = (cat.len() * 3).max(50).min(self.pts.len());
        let mut cache = self.inlier_grid.borrow_mut();
        if cache.as_ref().is_none_or(|(n, _)| *n != n_img) {
            let ids: Vec<u32> = (0..n_img as u32).collect();
            *cache = Some((n_img, Grid2::build(&self.pts, &ids, self.w, self.h, 2.0)));
        }
        self.count_matches(&cache.as_ref().unwrap().1, &cat, tol, wcs)
            .0
    }

    fn catalog_in_frame_margin(&self, wcs: &Wcs, rm: f64, margin: f64) -> Option<Vec<[f64; 4]>> {
        let c = unit(wcs.ra, wcs.dec);
        let radius = if wcs.proj.is_tan() {
            let corners = [
                [-margin, -margin],
                [self.w + margin, -margin],
                [-margin, self.h + margin],
                [self.w + margin, self.h + margin],
            ];
            let mut radius: f64 = 0.0;
            for q in corners {
                let sky = wcs.pixel_to_sky(q[0], q[1]);
                radius = radius.max(angle(c, unit(sky[0], sky[1])));
            }
            radius
        } else {
            self.frame_radius(wcs, c, margin)
        };
        if !radius.is_finite() || radius > self.max_frame_radius(wcs, 1.3) {
            return None;
        }
        Some(
            self.catalog_cone(c, radius * 1.02, rm)
                .iter()
                .map(|st| {
                    let (ra, dec) = radec(st.v);
                    [ra, dec, 0.0, 0.0]
                })
                .collect(),
        )
    }

    /// Level-`rm` catalogue stars (D >= rm, within the configured magnitude limit) in a cone.
    /// With a magnitude limit and no deep catalogue needed, the bright-star table answers the
    /// query in tens of microseconds (`Index::cone_stars`); otherwise the cell path is used.
    fn catalog_cone(&self, c: Vec3, radius: f64, rm: f64) -> Vec<crate::index::StarInfo> {
        let needs_deep = rm < self.idx.r7 * 0.999 && self.idx.has_deep();
        if let Some(ml) = self.cfg.mag_limit
            && !needs_deep
        {
            let mut bright = vec![];
            self.idx.cone_stars(c, radius, ml, rm * 0.999, &mut bright);
            return bright.iter().map(crate::index::StarInfo::from).collect();
        }
        self.idx.cached_cone_candidates(
            c,
            radius,
            rm * 0.999,
            self.cfg.mag_limit,
            needs_deep,
            &mut self.catalog_cache.borrow_mut(),
        )
    }

    /// Level-m catalogue stars (D >= rm) projecting inside the frame: (ra, dec, x, y).
    fn catalog_in_frame(&self, wcs: &Wcs, rm: f64) -> Option<Vec<[f64; 4]>> {
        let c = unit(wcs.ra, wcs.dec);
        let radius = if wcs.proj.is_tan() {
            let corners = [[0.0, 0.0], [self.w, 0.0], [0.0, self.h], [self.w, self.h]];
            let mut radius: f64 = 0.0;
            for q in corners {
                let sky = wcs.pixel_to_sky(q[0], q[1]);
                radius = radius.max(angle(c, unit(sky[0], sky[1])));
            }
            radius
        } else {
            self.frame_radius(wcs, c, 0.0)
        };
        if !(radius.is_finite()) || radius > self.max_frame_radius(wcs, 1.2) {
            return None;
        }
        let stars = self.catalog_cone(c, radius * 1.02, rm);
        let projector = wcs.prepare();
        let mut out = vec![];
        for st in &stars {
            let (ra, dec) = radec(st.v);
            if let Some(px) = projector.sky_to_pixel(ra, dec)
                && self.visible(px)
            {
                out.push([ra, dec, px[0], px[1]]);
            }
        }
        Some(out)
    }

    /// Unique nearest-neighbour matches of projected catalogue stars to image stars in `g`.
    fn count_matches(
        &self,
        g: &Grid2,
        cat: &[[f64; 4]],
        radius: f64,
        wcs: &Wcs,
    ) -> (usize, Vec<(u32, [f64; 2])>) {
        let projector = wcs.prepare();
        match_projected(
            &self.pts,
            g,
            cat.iter().filter_map(|c| {
                projector
                    .sky_to_pixel_fast(c[0], c[1])
                    .map(|px| ([c[0], c[1]], px))
            }),
            radius,
        )
    }
}

/// Match an ordered stream of (sky, pixel) projections. Sharing this implementation keeps
/// the uniqueness rule and catalogue order identical for fresh and reused projections.
fn match_projected(
    pts: &[[f64; 2]],
    grid: &Grid2,
    projected: impl Iterator<Item = ([f64; 2], [f64; 2])>,
    radius: f64,
) -> (usize, Vec<(u32, [f64; 2])>) {
    let mut used: FastMap<u32, ()> = FastMap::default();
    let mut out = vec![];
    for (sky, px) in projected {
        if let Some((_, i)) = grid.nearest_within(pts, px, radius, u32::MAX)
            && used.insert(i, ()).is_none()
        {
            out.push((i, sky));
        }
    }
    (out.len(), out)
}

/// Poisson log-odds of `k` matches against a chance expectation `lambda`:
/// ln [P(k | signal) / P(k | chance)] = k ln(k / lambda) - k + lambda for k > lambda, else 0
/// (the stage-2 evidence; shared with the tracker).
pub(crate) fn poisson_log_odds(k: usize, lambda: f64) -> f64 {
    let kf = k as f64;
    if kf > lambda {
        kf * (kf / lambda).ln() - kf + lambda
    } else {
        0.0
    }
}

/// ln P(X >= k) for X ~ Binomial(n, p) (Chernoff-tight upper bound via KL divergence).
pub(crate) fn log_binom_tail(n: usize, k: usize, p: f64) -> f64 {
    let n = n as f64;
    let k = k as f64;
    if k <= n * p {
        return 0.0;
    }
    if k >= n {
        return n * p.ln();
    }
    let q = k / n;
    -n * (q * (q / p).ln() + (1.0 - q) * ((1.0 - q) / (1.0 - p)).ln())
}

/// Detections sorted by brightness and the per-star quantities both modes need.
fn context<'a>(idx: &'a Index, field: &StarField, cfg: &'a SolveConfig, t0: Instant) -> Ctx<'a> {
    let mask = cfg.mask.as_deref();
    let mut stars: Vec<&Star> = field
        .stars
        .iter()
        .filter(|s| s.x.is_finite() && s.y.is_finite())
        .filter(|s| mask.is_none_or(|m| !m.is_blocked(s.x, s.y)))
        .collect();
    stars.sort_by(|a, b| {
        b.flux
            .partial_cmp(&a.flux)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    stars.truncate(cfg.max_stars);
    let (w, h) = (field.width as f64, field.height as f64);
    let pts: Vec<[f64; 2]> = stars.iter().map(|s| [s.x, s.y]).collect();
    let sig: Vec<f64> = stars.iter().map(|s| s.sigma.clamp(0.05, 2.0)).collect();
    let d = dominance_px(&pts, w, h);
    let edge = edge_px(&pts, w, h);
    // the open area exactly: a larger chance area would understate the chance density
    let open_area = mask.map_or(w * h, |m| (w * h - m.blocked_area(0.0, 0.0, w, h)).max(1.0));
    Ctx {
        idx,
        catalog_cache: RefCell::default(),
        inlier_grid: RefCell::default(),
        cfg,
        hint_cone: cfg.hint.and_then(|h| h.position).map(|[ra, dec]| {
            // Capture coordinates can be approximate. Include at least three degrees, and
            // the frame diagonal when CRVAL describes an off-centre reference pixel.
            let radius = cfg
                .hint
                .and_then(|h| h.fov_deg(field.width, field.height))
                .map_or(3.0, |f| (f * std::f64::consts::SQRT_2).max(3.0))
                .min(180.0);
            (unit(ra, dec), radius.to_radians().cos())
        }),
        pts,
        sig,
        d,
        edge,
        w,
        h,
        open_area,
        stats: SolveStats {
            n_stars: stars.len(),
            ..Default::default()
        },
        hits: Vec::with_capacity(4096),
        bins: FastMap::default(),
        t0,
        pa_sin: (0..1024)
            .map(|i| ((i as f64 + 0.5) * TAU / 1024.0).sin())
            .collect(),
        pa_cos: (0..1024)
            .map(|i| ((i as f64 + 0.5) * TAU / 1024.0).cos())
            .collect(),
        n_refs: cfg.n_refs,
        n_rest: cfg.n_rest,
        elem_cache: ElemCache::default(),
        cands: vec![],
        accept_ms: None,
    }
}

fn unsolved(field: &StarField, stats: SolveStats) -> Solution {
    let mut s = blank_solution("unsolved", field.width, field.height);
    s.stats = stats;
    s
}

/// Tracking mode (as a star tracker after its lost-in-space solve): verify a predicted WCS
/// against the catalogue instead of searching the index. `search_px` is the error allowed to the
/// prediction (the stage-1 matching radius). The test is the same as for a blind solve (stage-1
/// binomial tail, stage-2 evidence of `min_log_odds` nats, spread of the matches), so a wrong
/// prediction ends `unsolved`, never with a wrong WCS. With `refine` the distortion terms are
/// fitted again as in a blind solve; without, the prediction's are kept (fast path for video).
pub fn track_field(
    idx: &Index,
    field: &StarField,
    prior: &Wcs,
    search_px: f64,
    refine: bool,
    cfg: &SolveConfig,
) -> Solution {
    let t0 = Instant::now();
    let mut ctx = context(idx, field, cfg, t0);
    ctx.stats.select_ms = ctx.elapsed_ms();
    ctx.stats.verifications = 1;
    // the fast check first (it also absorbs a loose prediction); a distortion re-fit starts from
    // its result, so it needs no wide search of its own
    let fast = ctx.verify_tracked(prior, search_px);
    let sol = match fast {
        // fisheye and ultra-wide fields: the whole-frame refit of the window extension (ARC
        // when TAN cannot hold the field)
        Some(f) if refine && (!prior.proj.is_tan() || f.fov_width_deg > 60.0) => {
            wide::refit_whole(&ctx, &f).or(Some(f))
        }
        Some(f) if refine => {
            let tight = f.wcs.clone().expect("solved has a WCS");
            ctx.verify_wcs_within(tight, 0, 0.0).or(Some(f))
        }
        other => other,
    };
    ctx.stats.verify_ms = ctx.elapsed_ms() - ctx.stats.select_ms;
    ctx.stats.total_ms = ctx.elapsed_ms();
    match sol {
        Some(mut s) => {
            s.stats = ctx.stats;
            s
        }
        None => unsolved(field, ctx.stats),
    }
}

/// The window search of `wide` alone (no whole-frame search on the main index), within
/// `cfg.deadline_ms` and without the budget cap of frames without a sky mask: the last stage of
/// an image pipeline whose detection attempts all ended unsolved with time left.
/// `spent_verifications` are the verifications already made on the frame (trials).
pub fn solve_windows_only(
    idx: &Index,
    field: &StarField,
    cfg: &SolveConfig,
    spent_verifications: usize,
) -> Solution {
    let t0 = Instant::now();
    let c = SolveConfig {
        hint: None,
        extra_trials: cfg.extra_trials + spent_verifications,
        ..cfg.clone()
    };
    wide::solve_windows_within(idx, field, &c, t0, &SolveStats::default(), true)
        .unwrap_or_else(|| blank_solution("unsolved", field.width, field.height))
}

/// Solve a star list (sorted by brightness, brightest first).
pub fn solve_field(idx: &Index, field: &StarField, cfg: &SolveConfig) -> Solution {
    if cfg.hint.is_none() {
        return solve_field_blind(idx, field, cfg);
    }
    let Some(hint) = cfg
        .hint
        .and_then(|h| h.for_frame(field.width, field.height))
    else {
        return solve_field_blind(
            idx,
            field,
            &SolveConfig {
                hint: None,
                ..cfg.clone()
            },
        );
    };
    // A hint may spend at most a quarter of this attempt, capped at 200 ms. Even a valid
    // but stale header must leave most of the deadline for an unrestricted search.
    if cfg.deadline_ms < 20.0 {
        return solve_field_blind(
            idx,
            field,
            &SolveConfig {
                hint: None,
                ..cfg.clone()
            },
        );
    }
    let t0 = Instant::now();
    let c = hinted_config(idx, field, cfg, hint);
    let mut first = solve_field_blind(idx, field, &c);
    first.stats.hint_used = true;
    if first.status != "unsolved" {
        return first;
    }
    let mut sol = solve_field_blind(
        idx,
        field,
        &SolveConfig {
            hint: None,
            deadline_ms: (cfg.deadline_ms - t0.elapsed().as_secs_f64() * 1e3).max(0.0),
            extra_trials: cfg.extra_trials + first.stats.verifications,
            ..cfg.clone()
        },
    );
    // Keep the result's deep_search flag: it identifies the index of the returned WCS,
    // rather than the index of a discarded hinted attempt.
    let deep_search = sol.stats.deep_search;
    wide::accumulate(&mut sol.stats, &first.stats);
    sol.stats.deep_search = deep_search;
    sol.stats.hint_fallback = true;
    sol.stats.total_ms = t0.elapsed().as_secs_f64() * 1e3;
    sol
}

fn hinted_config(
    idx: &Index,
    field: &StarField,
    cfg: &SolveConfig,
    hint: crate::hints::SolveHint,
) -> SolveConfig {
    let fov = hint.fov_deg(field.width, field.height);
    let deep = cfg.mode == crate::api::SolveMode::Auto
        && cfg.deep_index.is_some()
        && fov.is_some_and(|f| f < crate::hints::DEEP_FOV_DEG);
    let deep_bounds =
        deep || cfg.mode == crate::api::SolveMode::Deep || idx.manifest.profile == "deep";
    let lower = if deep_bounds { 0.03 } else { cfg.min_fov_deg };
    let upper = if deep_bounds { 0.5 } else { cfg.max_fov_deg };
    SolveConfig {
        hint: Some(hint),
        mode: if deep {
            crate::api::SolveMode::Deep
        } else {
            cfg.mode
        },
        // A twofold tolerance also accommodates common binned-pixel conventions.
        min_fov_deg: fov.map_or(lower, |f| (f / 2.0).max(lower)),
        max_fov_deg: fov.map_or(upper, |f| (f * 2.0).min(upper)),
        deadline_ms: (cfg.deadline_ms * 0.25).min(200.0),
        windows: false,
        ..cfg.clone()
    }
}

fn solve_field_blind(idx: &Index, field: &StarField, cfg: &SolveConfig) -> Solution {
    use crate::api::SolveMode;
    let direct = idx.manifest.profile == "deep";
    let deep = if direct {
        Some(idx)
    } else {
        cfg.deep_index.as_ref().map(|d| d.0.as_ref())
    };
    let run_deep = |index: &Index, deadline_ms: f64, trials: usize| {
        let c = SolveConfig {
            deadline_ms,
            min_fov_deg: if cfg.hint.is_some() {
                cfg.min_fov_deg.max(0.03)
            } else {
                0.03
            },
            max_fov_deg: if cfg.hint.is_some() {
                cfg.max_fov_deg.min(0.5)
            } else {
                0.5
            },
            // Landscape masks can mistake resolved galaxies for foreground.
            mask: None,
            windows: false,
            wide_index: None,
            deep_index: None,
            mode: SolveMode::Standard,
            extra_trials: cfg.extra_trials + trials,
            ..cfg.clone()
        };
        let mut sol = solve_field_single(index, field, &c);
        sol.stats.deep_search = true;
        sol
    };
    if direct || cfg.mode == SolveMode::Deep {
        return deep.map_or_else(
            || blank_solution("unsolved", field.width, field.height),
            |index| run_deep(index, cfg.deadline_ms, 0),
        );
    }
    let Some(deep) = deep.filter(|_| cfg.mode == SolveMode::Auto) else {
        return solve_field_single(idx, field, cfg);
    };
    let t0 = Instant::now();
    let standard = solve_field_single(
        idx,
        field,
        &SolveConfig {
            deadline_ms: cfg.deadline_ms * 0.65,
            mode: SolveMode::Standard,
            deep_index: None,
            ..cfg.clone()
        },
    );
    if standard.status != "unsolved" {
        return standard;
    }
    let remaining = cfg.deadline_ms - t0.elapsed().as_secs_f64() * 1e3;
    if remaining <= 5.0 {
        return standard;
    }
    let mut sol = run_deep(deep, remaining, standard.stats.verifications);
    wide::accumulate(&mut sol.stats, &standard.stats);
    sol.stats.total_ms = t0.elapsed().as_secs_f64() * 1e3;
    sol
}

fn solve_field_single(idx: &Index, field: &StarField, cfg: &SolveConfig) -> Solution {
    let t0 = Instant::now();
    let (w, h) = (field.width as f64, field.height as f64);
    let mut ctx = context(idx, field, cfg, t0);
    ctx.stats.select_ms = ctx.elapsed_ms();
    let rungs = rung_order(&ctx.d, cfg.target_pool, cfg.min_pool, cfg.max_pool);
    let mut variants = Vec::with_capacity(128);
    let mut seen = std::collections::HashSet::with_hasher(Default::default());
    let mut nbuf = vec![];
    let mut passes: Vec<(bool, usize, usize)> =
        vec![(false, 2, 4), (false, cfg.n_refs, cfg.n_rest)];
    if cfg.exhaustive {
        passes.push((true, cfg.n_refs, cfg.n_rest));
    }
    let mut result: Option<Solution> = None;
    // a landscape (a mask with ground) is usually a lens too wide for the whole-frame search:
    // it keeps time for the windows
    let deadline = if cfg.windows && cfg.mask.as_deref().is_some_and(SkyMask::is_landscape) {
        cfg.deadline_ms * wide::LANDSCAPE_WHOLE_SHARE
    } else {
        cfg.deadline_ms
    };
    'outer: for &(exhaustive, n_refs, n_rest) in &passes {
        ctx.n_refs = n_refs;
        ctx.n_rest = n_rest;
        if exhaustive {
            ctx.stats.exhaustive_pass = true;
        }
        for &k in &rungs {
            if ctx.elapsed_ms() > deadline {
                break 'outer;
            }
            ctx.stats.rungs_tried += 1;
            let r = rung_radius(k);
            let pool: Vec<u32> = (0..ctx.pts.len() as u32)
                .filter(|&i| ctx.d[i as usize] >= r)
                .collect();
            if pool.len() < 7 {
                continue;
            }
            let grid = Grid2::build(&ctx.pts, &pool, w, h, 2.0);
            let anchors: Vec<u32> = pool
                .iter()
                .copied()
                .filter(|&i| {
                    let di = ctx.d[i as usize];
                    if exhaustive {
                        di < 2.0 * r
                    } else {
                        di >= 2.0 * r
                    }
                })
                .collect();
            for a in anchors {
                grid.knn(&ctx.pts, ctx.pts[a as usize], 6, a, &mut nbuf);
                if nbuf.len() < 4 {
                    continue;
                }
                // the core record stars (R, n2..n4) must be inside the frame
                if ctx.edge[a as usize] < nbuf[3].0.sqrt() {
                    continue;
                }
                ctx.stats.anchors += 1;
                let nb: Vec<(f64, u32)> = nbuf.clone();
                if let Some(sol) = ctx.process_anchor(a, &nb, r, &mut variants, &mut seen) {
                    result = Some(sol);
                    break 'outer;
                }
                // ambiguity continuation: budget spent or ambiguity established
                if ctx.continuation_over() {
                    break 'outer;
                }
                if ctx.elapsed_ms() > deadline {
                    break 'outer;
                }
            }
            // the continuation finishes the rung of the first acceptance, never the next one
            if ctx.continuing() {
                break 'outer;
            }
        }
    }
    if ctx.continuing() {
        ctx.sweep_pending_bins();
    }
    let result = result.or_else(|| ctx.conclude());
    ctx.stats.bins = ctx.bins.len();
    ctx.stats.total_ms = ctx.elapsed_ms();
    ctx.stats.probe_ms = ctx.stats.total_ms - ctx.stats.select_ms - ctx.stats.verify_ms;
    let mut s = result.unwrap_or_else(|| blank_solution("unsolved", field.width, field.height));
    s.stats = ctx.stats;
    // the whole-frame search ended unsolved with time left: search windows of the frame
    if s.status == "unsolved"
        && cfg.windows
        && let Some(ws) = wide::solve_windows(idx, field, cfg, t0, &s.stats)
    {
        return ws;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sol(ra: f64, dec: f64, scale: f64, rot: f64, parity: i32, log_odds: f64) -> Solution {
        let mut s = blank_solution("solved", 3000, 2000);
        s.ra = ra;
        s.dec = dec;
        s.scale_arcsec = scale;
        s.rotation_deg = rot;
        s.parity = parity;
        s.log_odds = log_odds;
        s
    }

    #[test]
    fn poisson_log_odds_is_zero_at_or_below_chance_and_grows_with_k() {
        assert_eq!(poisson_log_odds(0, 1.0), 0.0);
        assert_eq!(poisson_log_odds(3, 3.0), 0.0);
        let a = poisson_log_odds(10, 1.0);
        let b = poisson_log_odds(20, 1.0);
        assert!(a > 0.0 && b > a, "{a} {b}");
        // k ln(k/lambda) - k + lambda
        let expect = 10.0 * (10.0f64).ln() - 10.0 + 1.0;
        assert!((a - expect).abs() < 1e-12);
    }

    #[test]
    fn log_binom_tail_bounds() {
        assert_eq!(log_binom_tail(100, 5, 0.1), 0.0); // k <= n p
        assert!(log_binom_tail(100, 50, 0.1) < -50.0);
        assert!((log_binom_tail(10, 10, 0.5) - 10.0 * (0.5f64).ln()).abs() < 1e-12);
    }

    #[test]
    fn vote_cell_is_about_a_quarter_of_the_field() {
        // 2"/px on 3000 px: field 1.67 deg; cell between fov/2.8 and fov/1.4
        let s = (2.0f64 / 3600.0).to_radians();
        let fov = s * 3000.0;
        let cell = vote_cell_rad(s, 3000.0);
        assert!(
            cell > fov / 6.0 && cell < fov / 1.3,
            "cell {cell} fov {fov}"
        );
    }

    #[test]
    fn same_hypothesis_criterion() {
        let a = sol(83.8, -5.4, 2.0, 30.0, 1, 40.0);
        // tiny offsets: same place
        assert!(same_hypothesis(
            &a,
            &sol(83.81, -5.39, 2.01, 31.0, 1, 35.0),
            3000.0
        ));
        // opposite parity
        assert!(!same_hypothesis(
            &a,
            &sol(83.8, -5.4, 2.0, 30.0, -1, 40.0),
            3000.0
        ));
        // far away (10 deg on a 1.7 deg field)
        assert!(!same_hypothesis(
            &a,
            &sol(93.8, -5.4, 2.0, 30.0, 1, 40.0),
            3000.0
        ));
        // scale off by a factor 2
        assert!(!same_hypothesis(
            &a,
            &sol(83.8, -5.4, 4.0, 30.0, 1, 40.0),
            3000.0
        ));
        // rotation off by 90 deg
        assert!(!same_hypothesis(
            &a,
            &sol(83.8, -5.4, 2.0, 120.0, 1, 40.0),
            3000.0
        ));
        // rotation wrap-around
        assert!(same_hypothesis(
            &a,
            &sol(83.8, -5.4, 2.0, 30.0 + 359.0, 1, 40.0),
            3000.0
        ));
    }

    #[test]
    fn blank_solution_has_no_geometry() {
        let s = blank_solution("ambiguous", 10, 20);
        assert_eq!(s.status, "ambiguous");
        assert!(s.wcs.is_none() && s.ra.is_nan() && s.margin_nats.is_nan());
        assert_eq!(
            (s.width, s.height, s.n_candidates, s.parity),
            (10, 20, 0, 0)
        );
    }

    #[test]
    fn projected_matching_keeps_order_uniqueness_and_inclusive_radius() {
        let pts = [[10.0, 10.0], [20.0, 20.0]];
        let grid = Grid2::build(&pts, &[0, 1], 40.0, 40.0, 2.0);
        let projected = [
            ([1.0, 0.0], [10.0, 10.0]),
            ([2.0, 0.0], [10.1, 10.0]), // same detection: first catalogue star wins
            ([3.0, 0.0], [22.0, 20.0]), // exactly at the radius
            ([4.0, 0.0], [30.0, 30.0]), // outside the radius
        ];
        let got = match_projected(&pts, &grid, projected.into_iter(), 2.0);
        assert_eq!(got, (2, vec![(0, [1.0, 0.0]), (1, [3.0, 0.0])]));
        assert_eq!(
            match_projected(&pts, &grid, std::iter::empty(), 2.0),
            (0, vec![])
        );
    }

    #[test]
    fn solve_config_defaults_disable_the_new_features() {
        let c = SolveConfig::default();
        assert_eq!(c.min_margin_nats, 0.0);
        assert!(c.mag_limit.is_none());
        assert_eq!(c.continuation_factor, 0.5);
    }
}
