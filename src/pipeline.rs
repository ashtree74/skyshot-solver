//! Image -> solution pipeline with the re-detection fallback (F1): decode once, extract with the
//! default options, solve with part of the budget; if unsolved, re-extract against a local
//! annulus background with compact windows (nebulae, dense fields) and solve with the rest.

pub use crate::api::ATTEMPTS;
pub use crate::attempt::attempt_field;
use crate::{
    extract::{ExtractOptions, extract},
    image_io::{self, GrayImage, SourceInfo},
    index::Index,
    skymask::{self, SkyMask},
    solve::{Solution, SolveConfig, solve_field, track_field},
    types::{StarField, Wcs},
};
use anyhow::Result;
use serde::Serialize;
use std::{path::Path, sync::Arc, time::Instant};

#[derive(Clone, Debug, Default, Serialize)]
pub struct PipelineTimings {
    pub decode_ms: f64,
    pub extract_ms: f64,
    /// Sky mask estimate (part of `extract_ms`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask_ms: Option<f64>,
    pub solve_ms: f64,
    pub total_ms: f64,
    pub attempts: u32,
    pub fallback: Option<String>,
    /// Remote solves: time waiting for a free solver on a busy server (not solving).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queue_ms: Option<f64>,
    /// Remote solves: round trips minus the server's request time (network and HTTP).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub network_ms: Option<f64>,
    /// The solution came from tracking a predicted WCS (video), not from a blind search.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub tracked: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct PipelineResult {
    pub solution: Solution,
    pub timings: PipelineTimings,
    pub n_stars: usize,
    /// The sky mask the solve used (estimated from the pixels or given), if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask: Option<SkyMask>,
    /// Detections of the attempt that produced `solution` (the solving attempt, or the last
    /// one when unsolved), brightest first. Not serialized, so CLI output stays compact.
    #[serde(skip)]
    pub stars: Option<StarField>,
}

/// Largest final-window budget (ms) of ordinary frames without a landscape mask.
/// Wide panoramas may use the remaining [`solve_gray`] deadline.
pub const FINAL_WINDOWS_MS: f64 = 700.0;

/// Whether attempt `name` of a request with ready star lists ([`solve_attempts`], the API
/// server, where each attempt may come in its own request) may search windows of the frame
/// ([`SolveConfig::windows`]): the first attempt always, the later ones only on a landscape (a
/// mask with ground), so an ordinary field that does not solve pays for the windows once.
fn windows_for(cfg: &SolveConfig, name: &str, mask: Option<&SkyMask>) -> bool {
    cfg.windows && (name == ATTEMPTS[0].0 || mask.is_some_and(SkyMask::is_landscape))
}

/// Solve ready star lists of one frame in attempt order until one solves, splitting
/// `cfg.deadline_ms` as [`solve_gray`] does (the last list gets what is left). Returns the
/// solution and the index of the list that produced it (the last one tried when unsolved).
/// `attempts` must not be empty.
pub fn solve_attempts(
    idx: &Index,
    attempts: &[(&str, &StarField)],
    cfg: &SolveConfig,
) -> (Solution, usize) {
    let t0 = Instant::now();
    let mut last: Option<(Solution, usize)> = None;
    for (i, &(name, field)) in attempts.iter().enumerate() {
        let remaining = cfg.deadline_ms - t0.elapsed().as_secs_f64() * 1e3;
        if remaining <= 5.0 && last.is_some() {
            break;
        }
        let share = if i + 1 == attempts.len() {
            1.0
        } else {
            crate::api::attempt_share(name).unwrap_or(1.0)
        };
        let budget = if share < 1.0 {
            (cfg.deadline_ms * share).min(remaining)
        } else {
            remaining
        };
        let c = SolveConfig {
            deadline_ms: budget,
            windows: windows_for(cfg, name, cfg.mask.as_deref()),
            ..cfg.clone()
        };
        let sol = solve_field(idx, field, &c);
        let solved = sol.status == "solved";
        last = Some((sol, i));
        if solved {
            break;
        }
    }
    last.expect("solve_attempts needs at least one star list")
}

/// Solve an already decoded image.
pub fn solve_gray(
    idx: &Index,
    gray: &GrayImage,
    info: &SourceInfo,
    cfg: &SolveConfig,
) -> PipelineResult {
    let cfg = &SolveConfig {
        hint: crate::hints::SolveHint::merge(cfg.hint, info.hint),
        ..cfg.clone()
    };
    let t0 = Instant::now();
    let mut tm = PipelineTimings::default();
    let mut last: Option<(Solution, StarField)> = None;
    let mut mask = cfg.mask.clone();
    // the default detection and the verifications of all attempts, for the final window stage
    let mut default_field: Option<StarField> = None;
    let mut spent = 0usize;
    for (name, share) in ATTEMPTS {
        let elapsed = t0.elapsed().as_secs_f64() * 1e3;
        let remaining = cfg.deadline_ms - elapsed;
        // the first attempt always runs, so a tiny deadline yields `unsolved`, never a panic
        if remaining <= 5.0 && last.is_some() {
            break;
        }
        let te = Instant::now();
        let Some(field) = attempt_field(gray, info, name) else {
            continue;
        };
        // the sky mask comes from the pixels and the default detection (its PSF), once
        if mask.is_none() && cfg.auto_mask && last.is_none() {
            let tm0 = Instant::now();
            mask = skymask::estimate(gray, &field).map(Arc::new);
            tm.mask_ms = Some(tm0.elapsed().as_secs_f64() * 1e3);
        }
        tm.extract_ms += te.elapsed().as_secs_f64() * 1e3;
        // Detection belongs to this same deadline, including the last attempt.
        let remaining = (cfg.deadline_ms - t0.elapsed().as_secs_f64() * 1e3).max(0.0);
        let budget = if share < 1.0 {
            (cfg.deadline_ms * share).min(remaining)
        } else {
            remaining
        };
        let c = SolveConfig {
            deadline_ms: budget,
            // here a frame without a landscape mask searches windows only in the final stage
            // below, so a field the later detection attempts solve does not wait for them
            windows: cfg.windows && mask.as_deref().is_some_and(SkyMask::is_landscape),
            mask: mask.clone(),
            ..cfg.clone()
        };
        let ts = Instant::now();
        let sol = solve_field(idx, &field, &c);
        tm.solve_ms += ts.elapsed().as_secs_f64() * 1e3;
        tm.attempts += 1;
        spent += sol.stats.verifications;
        let solved = sol.status == "solved";
        if name == ATTEMPTS[0].0 {
            default_field = Some(field.clone());
        }
        let ambiguous = sol.status == "ambiguous";
        last = Some((sol, field));
        if solved || ambiguous {
            if solved {
                tm.fallback = (name != ATTEMPTS[0].0).then(|| name.to_string());
            }
            break;
        }
    }
    // every attempt ended unsolved with time left: the windows of the default detection get the
    // rest (ordinary fields: up to FINAL_WINDOWS_MS). Landscapes searched windows already.
    let remaining = cfg.deadline_ms - t0.elapsed().as_secs_f64() * 1e3;
    if cfg.windows
        && cfg.mode != crate::api::SolveMode::Deep
        && idx.manifest.profile != "deep"
        && remaining > 60.0
        && last.as_ref().is_some_and(|(s, _)| s.status == "unsolved")
        && !mask.as_deref().is_some_and(SkyMask::is_landscape)
        && let Some(field) = default_field
    {
        let ts = Instant::now();
        let c = SolveConfig {
            deadline_ms: if cfg.allow_panorama && field.width as f64 > 1.8 * field.height as f64 {
                remaining
            } else {
                remaining.min(FINAL_WINDOWS_MS)
            },
            mask: mask.clone(),
            ..cfg.clone()
        };
        let sol = crate::solve::solve_windows_only(idx, &field, &c, spent);
        tm.solve_ms += ts.elapsed().as_secs_f64() * 1e3;
        if sol.status != "unsolved" {
            tm.fallback = Some("windows".into());
            last = Some((sol, field));
        }
    }
    tm.total_ms = t0.elapsed().as_secs_f64() * 1e3;
    let (mut solution, field) = last.expect("at least one attempt");
    if solution.stats.deep_search {
        mask = None;
    }
    let refine_start = Instant::now();
    crate::wcsfit::registration::refine_solution(
        idx,
        &mut solution,
        &field,
        cfg,
        mask.as_deref(),
        cfg.deadline_ms - tm.total_ms,
    );
    tm.solve_ms += refine_start.elapsed().as_secs_f64() * 1e3;
    tm.total_ms = t0.elapsed().as_secs_f64() * 1e3;
    PipelineResult {
        solution,
        timings: tm,
        n_stars: field.stars.len(),
        mask: mask.map(|m| (*m).clone()),
        stars: Some(field),
    }
}

/// A predicted WCS for tracking and the error allowed to it (pixels).
#[derive(Clone, Debug)]
pub struct Prior {
    pub wcs: Wcs,
    pub search_px: f64,
    /// Fit the distortion terms again (otherwise the prediction's are kept).
    pub refine: bool,
    /// Budget of the blind solve when tracking fails (ms).
    pub fallback_ms: f64,
    /// The sky mask of the frame the prediction came from (a fixed camera keeps its
    /// foreground; a moving one gets a fresh estimate on the next blind solve).
    pub mask: Option<Arc<SkyMask>>,
}

/// Video frames: with a prior, extract once and verify the prediction (no search, a few ms);
/// when that fails (a cut, a zoom, clouds, a bad prediction), solve blind as usual.
pub fn solve_gray_tracked(
    idx: &Index,
    gray: &GrayImage,
    info: &SourceInfo,
    cfg: &SolveConfig,
    prior: Option<&Prior>,
) -> PipelineResult {
    let Some(prior) = prior else {
        return solve_gray(idx, gray, info, cfg);
    };
    let t0 = Instant::now();
    // a fixed camera keeps its foreground: the prior's mask holds for the next frames
    let cfg = &SolveConfig {
        mask: cfg.mask.clone().or_else(|| prior.mask.clone()),
        ..cfg.clone()
    };
    let te = Instant::now();
    let field = extract(gray, &ExtractOptions::for_source(info));
    let extract_ms = te.elapsed().as_secs_f64() * 1e3;
    let ts = Instant::now();
    let mut sol = track_field(idx, &field, &prior.wcs, prior.search_px, prior.refine, cfg);
    let mut solve_ms = ts.elapsed().as_secs_f64() * 1e3;
    if sol.status == "solved" {
        let refine_start = Instant::now();
        crate::wcsfit::registration::refine_solution(
            idx,
            &mut sol,
            &field,
            cfg,
            cfg.mask.as_deref(),
            cfg.deadline_ms - t0.elapsed().as_secs_f64() * 1e3,
        );
        solve_ms += refine_start.elapsed().as_secs_f64() * 1e3;
        return PipelineResult {
            solution: sol,
            timings: PipelineTimings {
                extract_ms,
                solve_ms,
                total_ms: t0.elapsed().as_secs_f64() * 1e3,
                attempts: 1,
                tracked: true,
                ..Default::default()
            },
            n_stars: field.stars.len(),
            mask: cfg.mask.as_deref().cloned(),
            stars: Some(field),
        };
    }
    if prior.fallback_ms <= 0.0 {
        // tracking only
        return PipelineResult {
            solution: sol,
            timings: PipelineTimings {
                extract_ms,
                solve_ms,
                total_ms: t0.elapsed().as_secs_f64() * 1e3,
                attempts: 1,
                ..Default::default()
            },
            n_stars: field.stars.len(),
            mask: cfg.mask.as_deref().cloned(),
            stars: Some(field),
        };
    }
    // blind fallback: estimate the mask afresh (a cut or a camera move changes the foreground)
    let rest = SolveConfig {
        deadline_ms: (cfg.deadline_ms - t0.elapsed().as_secs_f64() * 1e3)
            .min(prior.fallback_ms)
            .max(50.0),
        mask: None,
        ..cfg.clone()
    };
    let mut r = solve_gray(idx, gray, info, &rest);
    r.timings.extract_ms += extract_ms;
    r.timings.solve_ms += solve_ms;
    r.timings.total_ms = t0.elapsed().as_secs_f64() * 1e3;
    r
}

/// Decode `path` and solve it.
pub fn solve_path(idx: &Index, path: &Path, cfg: &SolveConfig) -> Result<PipelineResult> {
    let t0 = Instant::now();
    let (gray, info) = image_io::load_with_info(path)?;
    let decode_ms = t0.elapsed().as_secs_f64() * 1e3;
    let mut r = solve_gray(idx, &gray, &info, cfg);
    r.timings.decode_ms = decode_ms;
    r.timings.total_ms += decode_ms;
    Ok(r)
}

/// Approximate pointing supplied by capture software. Hints change only the order of the
/// attempts, never the acceptance test.
#[derive(Clone, Copy, Debug)]
pub struct Hint {
    pub ra: f64,
    pub dec: f64,
    pub radius_deg: f64,
    /// Approximate field size (deg), e.g. ASTAP's `-fov` (image height). Below
    /// [`DEEP_FIRST_FOV_DEG`] the regional deep index is tried before the blind solve.
    pub fov_deg: Option<f64>,
}

/// Fields narrower than this go to the hinted deep index first (the blind global index
/// rarely has enough stars below ~0.25 deg).
pub const DEEP_FIRST_FOV_DEG: f64 = 0.3;

/// Blind solve; if unsolved and a hint is given, build (or reuse from `cache`) a regional deep
/// index from D80 around the hint (radii two levels finer: R_7 = 0.485') and solve with it.
/// With a field-size hint below [`DEEP_FIRST_FOV_DEG`] the order is reversed.
pub fn solve_path_hinted(
    idx: &Index,
    path: &Path,
    cfg: &SolveConfig,
    hint: Option<Hint>,
    d80: &Path,
    cache: &Path,
) -> Result<PipelineResult> {
    let t0 = Instant::now();
    let (gray, info) = image_io::load_with_info(path)?;
    let decode_ms = t0.elapsed().as_secs_f64() * 1e3;
    let cfg = &SolveConfig {
        hint: crate::hints::SolveHint::merge(
            hint.map(|h| crate::hints::SolveHint {
                position: Some([h.ra, h.dec]),
                scale_arcsec: h.fov_deg.map(|f| f * 3600.0 / gray.height as f64),
            }),
            cfg.hint,
        ),
        ..cfg.clone()
    };
    let remaining = |floor: f64| (cfg.deadline_ms - t0.elapsed().as_secs_f64() * 1e3).max(floor);
    let blind = |deadline_ms: f64| {
        let mut r = solve_gray(
            idx,
            &gray,
            &info,
            &SolveConfig {
                deadline_ms,
                ..cfg.clone()
            },
        );
        r.timings.decode_ms = decode_ms;
        r.timings.total_ms = t0.elapsed().as_secs_f64() * 1e3;
        r
    };
    let Some(h) = hint else {
        return Ok(blind(cfg.deadline_ms));
    };
    let deep = |deadline_ms: f64| -> Result<PipelineResult> {
        let radius = h.radius_deg.clamp(0.5, 10.0);
        let key = format!("ra{:.2}_dec{:.2}_r{:.1}", h.ra, h.dec, radius);
        let dir = cache.join(key);
        if !dir.join("manifest.json").exists() {
            let cfgb = crate::builder::BuildConfig {
                d80: d80.to_path_buf(),
                out: dir.clone(),
                r7_arcmin: 0.485,
                dump_levels: false,
                ..Default::default()
            };
            crate::builder::build_region(&cfgb, h.ra, h.dec, radius)?;
        }
        let local = Index::open(&dir)?;
        let local_cfg = SolveConfig {
            deadline_ms,
            min_fov_deg: 0.03,
            ..cfg.clone()
        };
        let mut rl = solve_gray(&local, &gray, &info, &local_cfg);
        rl.timings.decode_ms = decode_ms;
        rl.timings.total_ms = t0.elapsed().as_secs_f64() * 1e3;
        rl.timings.fallback = Some(format!(
            "hinted_deep{}",
            rl.timings
                .fallback
                .as_deref()
                .map(|f| format!("+{f}"))
                .unwrap_or_default()
        ));
        Ok(rl)
    };
    if h.fov_deg.is_some_and(|f| f > 0.0 && f < DEEP_FIRST_FOV_DEG) {
        // very narrow field: the regional deep index first, the blind index as a fallback
        let rl = deep(cfg.deadline_ms.min(1500.0))?;
        if rl.solution.status == "solved" {
            return Ok(rl);
        }
        let r = blind(remaining(300.0));
        return Ok(if r.solution.status == "solved" { r } else { rl });
    }
    let r = blind(cfg.deadline_ms.min(800.0));
    if r.solution.status == "solved" {
        return Ok(r);
    }
    deep(remaining(500.0))
}
