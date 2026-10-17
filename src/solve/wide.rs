//! Window search for fields the whole-frame search cannot solve: fisheye and ultra-wide lenses
//! (the similarity model of the keys and votes breaks down far from the centre, and the
//! tangent-plane width exceeds `max_fov_deg`) and landscapes (a lit foreground outshines the
//! stars). A window of the frame is solved blind with the ordinary search and acceptance test,
//! the trials of every earlier search counted; its WCS is then grown over the whole frame in
//! rings around the window, with a zenithal equidistant (ARC) projection when TAN cannot hold
//! the field. The grown model must keep the window's identification: refinement never
//! overturns it.

use super::*;
use crate::{geometry::Projection, types::Proj};

mod panorama;

/// Windows as (x0, y0, x1, y1) fractions of the frame (of the open sky's bounding box when there
/// is a mask), in the order they are tried. The central windows come first: off the optical
/// axis a fisheye compresses radially and a rectilinear lens stretches radially, so the
/// similarity model of the keys holds best around the axis; then the upper windows (the sky of
/// a landscape). With a mask the most open windows go first anyway. Three sizes alternate:
/// 60 % of the frame keeps a diagonal fisheye of up to ~120 deg width within the search's
/// 75 deg, 40 % one of up to ~180 deg, 28 % a circular fisheye (180 deg across the short
/// side); trying all sizes early means a window too wide for the field costs one window's
/// time (such a search ends quickly: no hypothesis passes the field-size filter), not the
/// deadline.
const WINDOWS: [[f64; 4]; 20] = [
    [0.2, 0.2, 0.8, 0.8],
    [0.3, 0.3, 0.7, 0.7],
    [0.36, 0.36, 0.64, 0.64],
    [0.2, 0.0, 0.8, 0.6],
    [0.3, 0.05, 0.7, 0.45],
    [0.36, 0.08, 0.64, 0.36],
    [0.0, 0.0, 0.6, 0.6],
    [0.4, 0.0, 1.0, 0.6],
    [0.05, 0.05, 0.45, 0.45],
    [0.55, 0.05, 0.95, 0.45],
    [0.12, 0.12, 0.4, 0.4],
    [0.6, 0.12, 0.88, 0.4],
    [0.0, 0.2, 0.6, 0.8],
    [0.4, 0.2, 1.0, 0.8],
    [0.2, 0.4, 0.8, 1.0],
    [0.05, 0.3, 0.45, 0.7],
    [0.55, 0.3, 0.95, 0.7],
    [0.12, 0.36, 0.4, 0.64],
    [0.6, 0.36, 0.88, 0.64],
    [0.36, 0.6, 0.64, 0.88],
];

/// Fewest detections a window needs to be searched.
const MIN_WINDOW_STARS: usize = 40;
/// Smallest budget (ms) worth starting a window search with.
const MIN_WINDOW_MS: f64 = 40.0;
/// Each window gets this share of the time left (the first windows are the likely ones).
const WINDOW_SHARE: f64 = 0.4;
/// Share of the deadline the whole-frame search may use on a landscape (the rest is for the
/// windows).
pub(super) const LANDSCAPE_WHOLE_SHARE: f64 = 0.25;
/// Total window budget (ms) of a frame without a sky mask. A landscape (a mask with ground)
/// gets all the time left; a frame without one is usually an ordinary field that did not
/// solve, where the windows rarely help, so they may not take the whole deadline.
const NO_MASK_BUDGET_MS: f64 = 300.0;

/// The windows worth searching, best first: enough detections, at least half of the window
/// open under the mask; with a mask, the most open windows first (stable, so the order above
/// breaks ties). With a mask the fractions apply to the bounding box of the open sky, so a
/// circular fisheye's windows are fractions of its image circle, not of the frame.
fn plan(field: &StarField, mask: Option<&SkyMask>) -> Vec<[f64; 4]> {
    let (w, h) = (field.width as f64, field.height as f64);
    let b = mask.map_or([0.0, 0.0, w, h], |m| m.open_bbox(w, h));
    let (bw, bh) = (b[2] - b[0], b[3] - b[1]);
    let mut out: Vec<([f64; 4], f64)> = vec![];
    let mut rectangles = vec![];
    for (i, f) in WINDOWS.into_iter().enumerate() {
        if i == 0 && bw.max(bh) > 2.0 * bw.min(bh) {
            let side = 0.2 * bw.min(bh);
            let mut compact: Vec<_> = [(0.5, 0.5), (0.25, 0.5), (0.75, 0.5), (0.5, 0.25)]
                .into_iter()
                .map(|(cx, cy)| {
                    let (x, y) = (b[0] + cx * bw, b[1] + cy * bh);
                    [
                        x - side / 2.0,
                        y - side / 2.0,
                        x + side / 2.0,
                        y + side / 2.0,
                    ]
                })
                .collect();
            compact.sort_by_cached_key(|r| {
                std::cmp::Reverse(
                    field
                        .stars
                        .iter()
                        .filter(|s| s.x >= r[0] && s.x < r[2] && s.y >= r[1] && s.y < r[3])
                        .filter(|s| mask.is_none_or(|m| !m.is_blocked(s.x, s.y)))
                        .count(),
                )
            });
            rectangles.extend(compact);
        }
        if i == 3 && bw.max(bh) > 1.5 * bw.min(bh) {
            // A panorama's short dimension sets the locally near-linear patch size.
            // Percentage windows alone remain too long on very wide/tall frames.
            for size in [0.4, 0.28] {
                let side = size * bw.min(bh);
                for (cx, cy) in [(0.5, 0.5), (0.5, 0.25), (0.25, 0.5), (0.75, 0.5)] {
                    let (x, y) = (b[0] + cx * bw, b[1] + cy * bh);
                    rectangles.push([x - side / 2., y - side / 2., x + side / 2., y + side / 2.]);
                }
            }
        }
        rectangles.push([
            b[0] + f[0] * bw,
            b[1] + f[1] * bh,
            b[0] + f[2] * bw,
            b[1] + f[3] * bh,
        ]);
    }
    for r in rectangles {
        let open = mask.map_or(1.0, |m| m.open_fraction(r[0], r[1], r[2], r[3]));
        if open < 0.5 {
            continue;
        }
        let n = field
            .stars
            .iter()
            .filter(|s| s.x >= r[0] && s.x < r[2] && s.y >= r[1] && s.y < r[3])
            .filter(|s| mask.is_none_or(|m| !m.is_blocked(s.x, s.y)))
            .count();
        if n < MIN_WINDOW_STARS {
            continue;
        }
        // Small foreground intrusions must not demote every central window behind
        // the entire top-of-sky sweep. Above 90% open, retain the lens-centred
        // multiscale order; substantial occlusion still changes the priority.
        out.push((
            r,
            if open >= 0.9 {
                10.0
            } else {
                (open * 10.0).round()
            },
        ));
    }
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out.into_iter().map(|(r, _)| r).collect()
}

/// The detections of window `r` = (x0, y0, x1, y1) in window coordinates, brightest first.
fn window_field(field: &StarField, r: [f64; 4]) -> StarField {
    let (x0, y0) = (r[0], r[1]);
    StarField {
        width: (r[2] - r[0]).round().max(1.0) as u32,
        height: (r[3] - r[1]).round().max(1.0) as u32,
        stars: field
            .stars
            .iter()
            .filter(|s| s.x >= r[0] && s.x < r[2] && s.y >= r[1] && s.y < r[3])
            .map(|s| Star {
                x: s.x - x0,
                y: s.y - y0,
                ..s.clone()
            })
            .collect(),
        epoch: field.epoch,
    }
}

/// Solve the windows of `field` in [`plan`] order until one verifies, then extend its WCS to
/// the whole frame. `t0` and `cfg.deadline_ms` bound the total time; `spent` are the
/// statistics of the whole-frame search (its verifications count as trials).
pub(super) fn solve_windows(
    idx: &Index,
    field: &StarField,
    cfg: &SolveConfig,
    t0: Instant,
    spent: &SolveStats,
) -> Option<Solution> {
    solve_windows_within(idx, field, cfg, t0, spent, false)
}

/// [`solve_windows`]; `uncapped`: the whole deadline even without a landscape mask (the last
/// stage of a pipeline, [`super::solve_windows_only`]).
pub(super) fn solve_windows_within(
    idx: &Index,
    field: &StarField,
    cfg: &SolveConfig,
    t0: Instant,
    spent: &SolveStats,
    uncapped: bool,
) -> Option<Solution> {
    let mut trials = cfg.extra_trials + spent.verifications;
    let mut stats = spent.clone();
    let landscape = cfg.mask.as_deref().is_some_and(SkyMask::is_landscape);
    let start = t0.elapsed().as_secs_f64() * 1e3;
    let end = if landscape || uncapped {
        cfg.deadline_ms
    } else {
        cfg.deadline_ms.min(start + NO_MASK_BUDGET_MS)
    };
    // the searches in order: the whole frame on the companion index (coarse levels), then each
    // window on the main index and on the companion index
    let (w, h) = (field.width as f64, field.height as f64);
    let wide = cfg.wide_index.as_ref().map(|wi| &*wi.0);
    if cfg.allow_panorama
        && w > 1.8 * h
        && let Some(sol) = panorama::search(
            idx,
            wide,
            field,
            cfg,
            t0,
            start + 0.55 * (end - start),
            (&mut stats, &mut trials),
        )
    {
        return Some(sol);
    }
    let mut searches: Vec<([f64; 4], &Index, bool)> = vec![];
    for (i, r) in plan(field, cfg.mask.as_deref()).into_iter().enumerate() {
        // the whole frame on the companion index after the first window: it pays off for
        // fields of 10-100 deg with few stars, while fisheyes need the windows
        if i == 1
            && let Some(wi) = wide
        {
            searches.push(([0.0, 0.0, w, h], wi, true));
        }
        searches.push((r, idx, false));
        if let Some(wi) = wide {
            searches.push((r, wi, true));
        }
    }
    if searches.len() <= 2
        && let Some(wi) = wide
    {
        searches.push(([0.0, 0.0, w, h], wi, true));
    }
    for (r, index, coarse) in searches {
        let left = end - t0.elapsed().as_secs_f64() * 1e3;
        if left < MIN_WINDOW_MS {
            break;
        }
        let sub = window_field(field, r);
        let (min_fov, max_fov) = if coarse {
            WIDE_INDEX_FOV
        } else {
            (cfg.min_fov_deg, cfg.max_fov_deg)
        };
        let sub_cfg = SolveConfig {
            mode: crate::api::SolveMode::Standard,
            deep_index: None,
            deadline_ms: (left * WINDOW_SHARE).max(MIN_WINDOW_MS).min(left),
            windows: false,
            // The extension needs the verified seed correspondences even when
            // the caller does not request matches in the final response.
            keep_matches: true,
            extra_trials: trials,
            mask: cfg.mask.as_ref().map(|m| Arc::new(m.shifted(r[0], r[1]))),
            min_fov_deg: min_fov,
            max_fov_deg: max_fov,
            wide_index: None,
            ..cfg.clone()
        };
        let ws = solve_field(index, &sub, &sub_cfg);
        trials += ws.stats.verifications;
        accumulate(&mut stats, &ws.stats);
        if debug_on() {
            eprintln!(
                "  window {:?}{}: {} ({} stars, {} verifications, {:.1} ms, at {:.1} ms)",
                r.map(|v| v.round()),
                if coarse { " (wide index)" } else { "" },
                ws.status,
                sub.stars.len(),
                ws.stats.verifications,
                ws.stats.total_ms,
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        match ws.status.as_str() {
            "solved" => {
                let win = [r[0], r[1], sub.width as f64, sub.height as f64];
                let te = Instant::now();
                let ext_cfg = SolveConfig {
                    deadline_ms: (end - t0.elapsed().as_secs_f64() * 1e3).max(0.0),
                    extra_trials: trials,
                    ..cfg.clone()
                };
                let ext = extend(idx, field, &ext_cfg, &ws, win);
                if debug_on() {
                    eprintln!(
                        "  extend: {} in {:.1} ms",
                        if ext.is_some() { "kept" } else { "failed" },
                        te.elapsed().as_secs_f64() * 1e3
                    );
                }
                // a window solution that does not hold over the rest of the sky is not reported
                // (an anisotropic window far off-axis can verify a model that is wrong at the
                // frame centre): search on
                let Some(mut sol) = ext else {
                    continue;
                };
                stats.total_ms = t0.elapsed().as_secs_f64() * 1e3;
                stats.verifications = trials - cfg.extra_trials;
                sol.stats = stats;
                return Some(sol);
            }
            // two inconsistent verified hypotheses in a window: report it, never pick one
            "ambiguous" => {
                let mut s = blank_solution("ambiguous", field.width, field.height);
                s.margin_nats = ws.margin_nats;
                s.n_candidates = ws.n_candidates;
                s.stats = stats;
                return Some(s);
            }
            _ => {}
        }
    }
    None
}

pub(super) fn accumulate(total: &mut SolveStats, s: &SolveStats) {
    total.hint_used |= s.hint_used;
    total.hint_fallback |= s.hint_fallback;
    total.exhaustive_pass |= s.exhaustive_pass;
    total.deep_search |= s.deep_search;
    total.rungs_tried += s.rungs_tried;
    total.anchors += s.anchors;
    total.keys += s.keys;
    total.probes += s.probes;
    total.records += s.records;
    total.hits += s.hits;
    total.bins += s.bins;
    total.verifications += s.verifications;
    total.cheap_rejects += s.cheap_rejects;
    total.stage2 += s.stage2;
    total.branch2 += s.branch2;
    total.keygen_ms += s.keygen_ms;
    total.select_ms += s.select_ms;
    total.probe_ms += s.probe_ms;
    total.verify_ms += s.verify_ms;
}

/// Largest rms (px) of a grown model: a right field fits its stars to a pixel or two, a few with
/// strong refraction at the horizon of a fisheye; a model that does not (the wrong
/// extrapolation of an off-axis window seen in the synthetic set: 13 px) is rejected.
const GROWN_MAX_RMS: f64 = 4.0;

/// Whether a grown model holds over the frame: enough pairs, a tight fit, and pairs spread over
/// at least 40 % of the open sky's extent in both axes.
fn grown_holds(ctx: &Ctx<'_>, wcs: &Wcs, pairs: &[[f64; 4]], rms: f64) -> bool {
    if pairs.len() < 20 || !rms.is_finite() || rms > GROWN_MAX_RMS {
        return false;
    }
    if !mapping_holds(ctx, wcs) {
        return false;
    }
    let b = ctx
        .cfg
        .mask
        .as_deref()
        .map_or([0.0, 0.0, ctx.w, ctx.h], |m| m.open_bbox(ctx.w, ctx.h));
    let (mut x0, mut x1, mut y0, mut y1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for p in pairs {
        x0 = x0.min(p[0]);
        x1 = x1.max(p[0]);
        y0 = y0.min(p[1]);
        y1 = y1.max(p[1]);
    }
    x1 - x0 >= 0.4 * (b[2] - b[0]) && y1 - y0 >= 0.4 * (b[3] - b[1])
}

/// A high-order fit can keep central stars yet fold at the edge. Check the open sky
/// for both a recovered curved seed and its final full-frame extension.
pub(super) fn mapping_holds(ctx: &Ctx<'_>, wcs: &Wcs) -> bool {
    let b = ctx
        .cfg
        .mask
        .as_deref()
        .map_or([0.0, 0.0, ctx.w, ctx.h], |m| m.open_bbox(ctx.w, ctx.h));
    for i in 0..=8 {
        for j in 0..=8 {
            let x = b[0] + i as f64 * (b[2] - b[0] - 1.0).max(0.0) / 8.0;
            let y = b[1] + j as f64 * (b[3] - b[1] - 1.0).max(0.0) / 8.0;
            if !ctx.visible([x, y]) {
                continue;
            }
            let sky = wcs.pixel_to_sky(x, y);
            let Some(back) = wcs.sky_to_pixel(sky[0], sky[1]) else {
                return false;
            };
            if !back.iter().all(|v| v.is_finite()) || (back[0] - x).hypot(back[1] - y) > 0.1 {
                return false;
            }
        }
    }
    true
}

/// Grow the solution `ws` of window `win` = (x0, y0, width, height) over the whole frame.
fn extend(
    idx: &Index,
    field: &StarField,
    cfg: &SolveConfig,
    ws: &Solution,
    win: [f64; 4],
) -> Option<Solution> {
    let t = Instant::now();
    let ctx = context(idx, field, cfg, t);
    let mut w0 = ws.wcs.clone()?;
    w0.crpix = [w0.crpix[0] + win[0], w0.crpix[1] + win[1]];
    let mut fc = ctx.frame_catalog(&w0, ctx.frame_level(&w0))?;
    // the window's identification must survive: compare both models on the window's stars
    let rect = [win[0], win[1], win[0] + win[2], win[1] + win[3]];
    let n_w0 = ctx.inliers_in(&fc, &w0, rect, 2.5);
    let accept = |g: Option<(Wcs, Vec<[f64; 4]>, f64)>, fc: &FrameCat| {
        g.filter(|(g, pairs, rms)| {
            ctx.inliers_in(fc, g, rect, 2.5) as f64 >= 0.9 * n_w0 as f64
                && grown_holds(&ctx, g, pairs, *rms)
        })
    };
    let mut grown = accept(
        ctx.grow(&w0, win, ws.rms_px, &mut fc, Proj::Tan.into()),
        &fc,
    );
    // a fisheye can lead a TAN start astray before the rings are wide enough to choose ARC:
    // grow once more with ARC from the first ring
    if grown.is_none() && cfg.allow_arc {
        if debug_on() {
            eprintln!("  extend: again with ARC from the start");
        }
        grown = accept(
            ctx.grow(&w0, win, ws.rms_px, &mut fc, Proj::Arc.into()),
            &fc,
        );
    }
    let open_bbox = cfg
        .mask
        .as_deref()
        .map_or([0., 0., ctx.w, ctx.h], |m| m.open_bbox(ctx.w, ctx.h));
    let (bw, bh) = (open_bbox[2] - open_bbox[0], open_bbox[3] - open_bbox[1]);
    let elongated = bw.max(bh) > 1.5 * bw.min(bh);
    if (grown.is_none() || elongated) && cfg.allow_panorama && ctx.elapsed_ms() < cfg.deadline_ms {
        let seed_pairs: Vec<_> = ws
            .matches
            .iter()
            .map(|m| [m.x + win[0], m.y + win[1], m.ra, m.dec])
            .collect();
        if let Some((seed, _)) = crate::wcsfit::fit_stereographic(&seed_pairs, &[])
            && let Some(fit) = ctx.grow_stereographic(&seed)
        {
            let retained = seed_pairs
                .iter()
                .filter(|p| {
                    fit.0
                        .sky_to_pixel(p[2], p[3])
                        .is_some_and(|q| (q[0] - p[0]).hypot(q[1] - p[1]) <= 2.5)
                })
                .count();
            if retained as f64 >= 0.9 * seed_pairs.len() as f64
                && grown_holds(&ctx, &fit.0, &fit.1, fit.2)
                && ctx.extension_evidence(&fit.0, rect)
            {
                let n = ctx.inliers_in(&fc, &fit.0, open_bbox, 2.5);
                let old = grown
                    .as_ref()
                    .map_or(0, |(w, _, _)| ctx.inliers_in(&fc, w, open_bbox, 2.5));
                if grown.is_none() || n as f64 > 1.1 * old as f64 {
                    grown = Some(fit);
                }
            }
        }
        let seed_cat = ctx.zone_stars(
            &fc,
            &w0,
            [win[0] + win[2] / 2.0, win[1] + win[3] / 2.0],
            0.5 * win[2].hypot(win[3]),
        );
        let ids: Vec<u32> = (0..ctx.pts.len() as u32).collect();
        let grid = Grid2::build(&ctx.pts, &ids, ctx.w, ctx.h, 2.0);
        let (_, matches) = match_projected(
            &ctx.pts,
            &grid,
            seed_cat.iter().map(|c| ([c[0], c[1]], [c[2], c[3]])),
            2.5,
        );
        let pairs = ctx.pairs(&matches);
        let cp = [win[0] + win[2] / 2.0, win[1] + win[3] / 2.0];
        let tangent = w0.pixel_to_sky(cp[0], cp[1]);
        let mut candidates = crate::wcsfit::panorama_candidates(&pairs, cp, tangent);
        candidates.push(Proj::Stg.into());
        for candidate in candidates {
            if ctx.elapsed_ms() >= cfg.deadline_ms {
                break;
            }
            if debug_on() {
                eprintln!("  extend: panorama {:?}", candidate);
            }
            // Put the new projection's reference point inside the identified patch
            // before extrapolating it to the full frame's centre.
            let seed = crate::wcsfit::robust_fit_wmp(&pairs, &[], cp, tangent, 1, 0, candidate)
                .map(|(w, _)| w)
                .unwrap_or_else(|| w0.clone());
            if let Some(fit) = accept(ctx.grow(&seed, win, ws.rms_px, &mut fc, candidate), &fc) {
                let n = ctx.inliers_in(&fc, &fit.0, open_bbox, 2.5);
                let support = grown
                    .as_ref()
                    .map_or(0, |(w, _, _)| ctx.inliers_in(&fc, w, open_bbox, 2.5));
                // Prefer the existing model unless the new geometry explains clearly
                // more catalogue stars, at the same pixel tolerance and with the same seed.
                if grown.is_none() || n as f64 > 1.1 * support as f64 {
                    grown = Some(fit);
                }
            }
        }
    }
    finish(&ctx, field, cfg, ws, win, grown?)
}

fn finish(
    ctx: &Ctx<'_>,
    field: &StarField,
    cfg: &SolveConfig,
    ws: &Solution,
    win: [f64; 4],
    (wcs, pairs, rms): (Wcs, Vec<[f64; 4]>, f64),
) -> Option<Solution> {
    let crpix = [(ctx.w - 1.0) / 2.0, (ctx.h - 1.0) / 2.0];
    let center = wcs.pixel_to_sky(crpix[0], crpix[1]);
    let n_matches = if pairs.is_empty() {
        ws.n_matches
    } else {
        pairs.len()
    };
    let refraction = ctx.measure_refraction(&wcs, &pairs);
    Some(Solution {
        status: "solved".into(),
        width: field.width,
        height: field.height,
        ra: center[0],
        dec: center[1],
        scale_arcsec: wcs.scale_arcsec(),
        rotation_deg: wcs.rotation_deg(),
        parity: if wcs.determinant() > 0.0 { 1 } else { -1 },
        fov_width_deg: ctx.fov_width(&wcs),
        n_matches,
        n_catalog: ws.n_catalog,
        rms_px: rms,
        log_odds: ws.log_odds,
        cluster_anchors: ws.cluster_anchors,
        verify_level: ws.verify_level,
        stats: SolveStats::default(),
        matches: if cfg.keep_matches {
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
        margin_nats: ws.margin_nats,
        n_candidates: ws.n_candidates,
        window: Some(win),
        refraction,
        sky_extent: ctx.sky_extent(),
    })
}

/// Tracking with a distortion refit on a fisheye or ultra-wide field: refit the verified
/// prediction `f` over the whole frame with the projection choice of the window extension
/// (TAN cannot hold such a field; the ordinary refinement is TAN). `None` when the refit does
/// not keep the prediction's matches; the caller then keeps `f`.
pub(super) fn refit_whole(ctx: &Ctx<'_>, f: &Solution) -> Option<Solution> {
    let w0 = f.wcs.clone()?;
    let mut fc = ctx.frame_catalog(&w0, ctx.frame_level(&w0))?;
    let rect = [0.0, 0.0, ctx.w, ctx.h];
    let start = if (w0.proj == Proj::Arc && ctx.cfg.allow_arc)
        || (w0.proj != Proj::Arc && !w0.proj.is_tan() && ctx.cfg.allow_panorama)
    {
        w0.projection()
    } else {
        Proj::Tan.into()
    };
    let (wcs, pairs, rms) = ctx.grow(&w0, [0.0, 0.0, ctx.w, ctx.h], f.rms_px, &mut fc, start)?;
    let (n_new, n_old) = (
        ctx.inliers_in(&fc, &wcs, rect, 2.5),
        ctx.inliers_in(&fc, &w0, rect, 2.5),
    );
    if (n_new as f64) < 0.9 * n_old as f64 || !grown_holds(ctx, &wcs, &pairs, rms) {
        return None;
    }
    let crpix = [(ctx.w - 1.0) / 2.0, (ctx.h - 1.0) / 2.0];
    let center = wcs.pixel_to_sky(crpix[0], crpix[1]);
    Some(Solution {
        ra: center[0],
        dec: center[1],
        scale_arcsec: wcs.scale_arcsec(),
        rotation_deg: wcs.rotation_deg(),
        parity: if wcs.determinant() > 0.0 { 1 } else { -1 },
        fov_width_deg: ctx.fov_width(&wcs),
        n_matches: pairs.len(),
        rms_px: rms,
        matches: if ctx.cfg.keep_matches {
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
        ..f.clone()
    })
}

/// The catalogue of one extension: the level's stars in a cone around the frame, queried once
/// and re-queried only when a grown model needs a wider cone.
struct FrameCat {
    stars: Vec<[f64; 2]>,
    level: u32,
    center: Vec3,
    radius: f64,
}

impl Ctx<'_> {
    /// Finest level with <= ~1500 stars in the open frame under `wcs`.
    fn frame_level(&self, wcs: &Wcs) -> u32 {
        let s = wcs.scale_arcsec().to_radians() / 3600.0;
        let area = s * s * self.open_area;
        let mut mr = if self.idx.has_deep() { 8u32 } else { 7u32 };
        while mr > 1 && area / (PI * self.idx.level_radius(mr).powi(2)) > 1500.0 {
            mr -= 1;
        }
        mr
    }

    /// Level-`level` stars (ra, dec) within `radius` (rad) of `c`. Levels without the deep
    /// catalogue use the index's raw-byte cone scan (one pass even for a hemisphere).
    fn cone_radec(&self, c: Vec3, radius: f64, level: u32) -> Vec<[f64; 2]> {
        let rm = self.idx.level_radius(level);
        if rm < self.idx.r7 * 0.999 && self.idx.has_deep() {
            return self
                .catalog_cone(c, radius, rm)
                .iter()
                .map(|st| {
                    let (ra, dec) = radec(st.v);
                    [ra, dec]
                })
                .collect();
        }
        let mut out = vec![];
        self.idx.cone_stars(
            c,
            radius.min(PI),
            self.cfg.mag_limit.unwrap_or(f32::MAX),
            rm * 0.999,
            &mut out,
        );
        out.iter()
            .map(|st| {
                let (ra, dec) = radec(st.v);
                [ra, dec]
            })
            .collect()
    }

    /// The catalogue for extending `wcs` over the frame: a cone around the frame centre 1.5x
    /// the frame's angular radius under `wcs` (a model fitted on a window underestimates the
    /// angles of a fisheye far from the window) plus 0.15 rad.
    fn frame_catalog(&self, wcs: &Wcs, level: u32) -> Option<FrameCat> {
        let crpix = [(self.w - 1.0) / 2.0, (self.h - 1.0) / 2.0];
        let c = wcs.pixel_to_sky(crpix[0], crpix[1]);
        let center = unit(c[0], c[1]);
        let fr = self.frame_radius(wcs, center, 0.0);
        if !fr.is_finite() {
            return None;
        }
        let radius = (1.5 * fr + 0.15).min(PI);
        Some(FrameCat {
            stars: self.cone_radec(center, radius, level),
            level,
            center,
            radius,
        })
    }

    /// Visible stars of `fc` (ra, dec, x, y) predicted by `wcs` within `r` px of `pc`.
    fn zone_stars(&self, fc: &FrameCat, wcs: &Wcs, pc: [f64; 2], r: f64) -> Vec<[f64; 4]> {
        let mut out = Vec::with_capacity(fc.stars.len() / 2);
        let projector = wcs.prepare();
        for s in &fc.stars {
            if let Some(px) = projector.sky_to_pixel_fast(s[0], s[1])
                && (px[0] - pc[0]).hypot(px[1] - pc[1]) <= r
                && self.visible(px)
            {
                out.push([s[0], s[1], px[0], px[1]]);
            }
        }
        out
    }

    /// Stars of `fc` matched within `tol` px under `wcs` inside the pixel rectangle `rect`
    /// (unique nearest neighbours).
    fn inliers_in(&self, fc: &FrameCat, wcs: &Wcs, rect: [f64; 4], tol: f64) -> usize {
        let pc = [(rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0];
        let r = 0.5 * (rect[2] - rect[0]).hypot(rect[3] - rect[1]);
        let inside: Vec<[f64; 4]> = self
            .zone_stars(fc, wcs, pc, r)
            .into_iter()
            .filter(|c| c[2] >= rect[0] && c[2] < rect[2] && c[3] >= rect[1] && c[3] < rect[3])
            .collect();
        let n_img = (inside.len() * 3).max(50).min(self.pts.len());
        let ids: Vec<u32> = (0..n_img as u32).collect();
        let g = Grid2::build(&self.pts, &ids, self.w, self.h, 2.0);
        match_projected(
            &self.pts,
            &g,
            inside.iter().map(|c| ([c[0], c[1]], [c[2], c[3]])),
            tol,
        )
        .0
    }

    /// Extend a rigid stereographic seed without forcing its optical centre to
    /// the middle of a cropped image. Catalogue and detection gates remain shared.
    fn grow_stereographic(&self, seed: &Wcs) -> Option<(Wcs, Vec<[f64; 4]>, f64)> {
        let mut model = seed.clone();
        let pc = [(self.w - 1.0) / 2.0, (self.h - 1.0) / 2.0];
        let radius = 0.5 * self.w.hypot(self.h);
        let mut best = None;
        for tolerance in [12.0, 6.0, 3.0, 2.5] {
            if self.elapsed_ms() >= self.cfg.deadline_ms {
                break;
            }
            let fc = self.frame_catalog(&model, self.frame_level(&model))?;
            let cat = self.zone_stars(&fc, &model, pc, radius);
            let n_img = (cat.len() * 3).max(50).min(self.pts.len());
            let ids: Vec<_> = (0..n_img as u32).collect();
            let grid = Grid2::build(&self.pts, &ids, self.w, self.h, 2.0);
            let (_, matches) = match_projected(
                &self.pts,
                &grid,
                cat.iter().map(|c| ([c[0], c[1]], [c[2], c[3]])),
                tolerance,
            );
            let pairs = self.pairs(&matches);
            let weights = self.weights(&matches);
            let (fit, keep) = crate::wcsfit::fit_stereographic(&pairs, &weights)?;
            let inliers: Vec<_> = pairs
                .iter()
                .zip(&keep)
                .filter(|(_, k)| **k)
                .map(|(p, _)| *p)
                .collect();
            let rms = crate::wcsfit::rms(&crate::wcsfit::residuals(&fit, &inliers));
            if debug_on() {
                eprintln!(
                    "  grow STG: {} inliers, rms {rms:.3}, centre ({:.3}, {:.3}), CRPIX {:?}",
                    inliers.len(),
                    fit.ra,
                    fit.dec,
                    fit.crpix
                );
            }
            model = fit;
            best = Some((model.clone(), inliers, rms));
        }
        best
    }

    /// A projection can fit its seed yet collect scattered chance matches elsewhere.
    /// Require catalogue evidence outside that seed, using all visible detections
    /// for the chance expectation, rather than the brightness prefix used to fit.
    fn extension_evidence(&self, wcs: &Wcs, seed: [f64; 4]) -> bool {
        let outside =
            |p: [f64; 2]| p[0] < seed[0] || p[0] >= seed[2] || p[1] < seed[1] || p[1] >= seed[3];
        let pc = [(self.w - 1.) / 2., (self.h - 1.) / 2.];
        let Some(fc) = self.frame_catalog(wcs, self.frame_level(wcs)) else {
            return false;
        };
        let cat = self.zone_stars(&fc, wcs, pc, 0.5 * self.w.hypot(self.h));
        let n_fit = (cat.len() * 3).max(50).min(self.pts.len());
        let cat: Vec<_> = cat.into_iter().filter(|p| outside([p[2], p[3]])).collect();
        let ids: Vec<_> = self
            .pts
            .iter()
            .enumerate()
            .take(n_fit)
            .filter(|(_, p)| outside(**p))
            .map(|(i, _)| i as u32)
            .collect();
        let tolerance: f64 = 2.5;
        let grid = Grid2::build(&self.pts, &ids, self.w, self.h, 2.0);
        let k = match_projected(
            &self.pts,
            &grid,
            cat.iter().map(|c| ([c[0], c[1]], [c[2], c[3]])),
            tolerance,
        )
        .0;
        let blocked = self
            .cfg
            .mask
            .as_deref()
            .map_or(0., |m| m.blocked_area(seed[0], seed[1], seed[2], seed[3]));
        let area = self.open_area - ((seed[2] - seed[0]) * (seed[3] - seed[1]) - blocked);
        if area <= 0. || cat.is_empty() || ids.is_empty() {
            return false;
        }
        // Count evidence on the depth-matched brightness prefix used for fitting.
        // All visible detections set the chance rate: deliberately conservative
        // when a locally correct model gathers chance pairs in the dense sky.
        let n_img = self.pts.iter().filter(|p| outside(**p)).count();
        let lambda = cat.len() as f64 * n_img as f64 * PI * tolerance.powi(2) / area;
        let ll = poisson_log_odds(k, lambda);
        let threshold = self.cfg.min_log_odds + self.trials_ln();
        if debug_on() {
            eprintln!("  STG extension: k={k} lambda={lambda:.2} ll={ll:.1} thr={threshold:.1}");
        }
        k >= 24 && ll >= threshold
    }

    /// Grow `w0` (identified on window `win`) over the frame: rings of 1.35x the radius around
    /// the window centre, each matched with the current model and refitted (TAN, or ARC once
    /// the field is too wide for TAN; `start` = ARC fits ARC from the first ring), then a model
    /// choice on held-out stars. Returns the model, its inlier pairs and their rms (px).
    fn grow(
        &self,
        w0: &Wcs,
        win: [f64; 4],
        rms0: f64,
        fc: &mut FrameCat,
        start: Projection,
    ) -> Option<(Wcs, Vec<[f64; 4]>, f64)> {
        let crpix = [(self.w - 1.0) / 2.0, (self.h - 1.0) / 2.0];
        let pc = [win[0] + win[2] / 2.0, win[1] + win[3] / 2.0];
        let r_end = [[0.0, 0.0], [self.w, 0.0], [0.0, self.h], [self.w, self.h]]
            .iter()
            .map(|q| (q[0] - pc[0]).hypot(q[1] - pc[1]))
            .fold(0.0, f64::max);
        let mut r = 0.5 * win[2].hypot(win[3]);
        let mut r_prev = 0.0;
        let mut model = w0.clone();
        let mut rms = if rms0.is_finite() {
            rms0.clamp(0.3, 3.0)
        } else {
            1.0
        };
        let mut best: Option<(Wcs, Vec<[f64; 4]>, f64)> = None;
        let mut last_pairs: (Vec<[f64; 4]>, Vec<f64>) = (vec![], vec![]);
        let mut proj = start.fixed([w0.ra, w0.dec]);
        loop {
            if self.elapsed_ms() >= self.cfg.deadline_ms {
                break;
            }
            let full = r >= r_end;
            // a model grown wider than the cone of the catalogue: query a wider cone
            if self.frame_radius(&model, fc.center, 0.0) > 0.95 * fc.radius && fc.radius < PI {
                *fc = self.frame_catalog(&model, fc.level)?;
            }
            let cat = self.zone_stars(fc, &model, pc, r.min(r_end));
            let n_img = (cat.len() * 3).max(50).min(self.pts.len());
            let ids: Vec<u32> = (0..n_img as u32).collect();
            let g = Grid2::build(&self.pts, &ids, self.w, self.h, 2.0);
            let sep = (self.open_area / cat.len().max(1) as f64).sqrt();
            let rcap = (0.3 * sep).clamp(1.5, 25.0);
            let tol = (4.0 * rms + 0.01 * (r - r_prev)).clamp(1.5, rcap);
            let (_, m) = match_projected(
                &self.pts,
                &g,
                cat.iter().map(|c| ([c[0], c[1]], [c[2], c[3]])),
                tol,
            );
            let pairs = self.pairs(&m);
            let wts = if unweighted_on() {
                vec![]
            } else {
                self.weights(&m)
            };
            if pairs.len() < 12 {
                break;
            }
            let order = match pairs.len() {
                0..=29 => 1,
                30..=59 => 2,
                60..=99 => 3,
                _ => 4,
            };
            let radial = if pairs.len() >= 120 { 7 } else { 5 };
            let tangent = model.pixel_to_sky(crpix[0], crpix[1]);
            // the zone's angular radius decides whether ARC is worth trying
            let wide = self.zone_angle(&model, pc, r.min(r_end)) > 0.35;
            let mut fits = vec![];
            if proj.proj == Proj::Tan {
                fits.push(crate::wcsfit::robust_fit_wmp(
                    &pairs,
                    &wts,
                    crpix,
                    tangent,
                    order,
                    radial,
                    Proj::Tan,
                ));
            }
            if (wide || proj.proj == Proj::Arc)
                && self.cfg.allow_arc
                && !start.proj.is_cylindrical()
                && start.proj != Proj::Stg
            {
                fits.push(crate::wcsfit::robust_fit_wmp(
                    &pairs,
                    &wts,
                    crpix,
                    tangent,
                    order,
                    radial,
                    Proj::Arc,
                ));
            }
            if start.proj.is_cylindrical() || start.proj == Proj::Stg {
                fits.push(crate::wcsfit::robust_fit_wmp(
                    &pairs, &wts, crpix, tangent, order, 0, proj,
                ));
            }
            let mut chosen: Option<(Wcs, Vec<bool>, f64)> = None;
            for (fit, keep) in fits.into_iter().flatten() {
                let inl: Vec<[f64; 4]> = pairs
                    .iter()
                    .zip(&keep)
                    .filter(|(_, k)| **k)
                    .map(|(p, _)| *p)
                    .collect();
                let e = crate::wcsfit::rms(&crate::wcsfit::residuals(&fit, &inl));
                // ARC must do clearly better than TAN to replace it
                let e_cmp = if fit.proj == Proj::Arc { e * 1.1 } else { e };
                if e.is_finite() && chosen.as_ref().is_none_or(|c| e_cmp < c.2) {
                    chosen = Some((fit, keep, e_cmp));
                }
            }
            let Some((fit, keep, _)) = chosen else {
                break;
            };
            let inl: Vec<[f64; 4]> = pairs
                .iter()
                .zip(&keep)
                .filter(|(_, k)| **k)
                .map(|(p, _)| *p)
                .collect();
            rms = crate::wcsfit::rms(&crate::wcsfit::residuals(&fit, &inl)).max(0.05);
            proj = fit.projection();
            if debug_on() {
                eprintln!(
                    "  grow: r {:.0}/{:.0} px tol {tol:.1} cat {} pairs {} inliers {} {:?} order {order} rms {rms:.3}",
                    r,
                    r_end,
                    cat.len(),
                    pairs.len(),
                    inl.len(),
                    proj
                );
            }
            model = fit;
            best = Some((model.clone(), inl, rms));
            last_pairs = (pairs, wts);
            if full {
                break;
            }
            r_prev = r;
            r *= 1.35;
        }
        // model choice on held-out stars of the last ring (all of the frame when it grew so far)
        let (pairs, wts) = last_pairs;
        if pairs.len() >= 40 {
            let tangent = model.pixel_to_sky(crpix[0], crpix[1]);
            let max_order = if pairs.len() >= 160 { 5 } else { 4 };
            if let Some((fit, keep, _, m)) = crate::wcsfit::select_model_wp(
                &pairs,
                &wts,
                crpix,
                tangent,
                max_order,
                if model.proj.is_cylindrical() { 0 } else { 7 },
                model.projection(),
            ) {
                let inl: Vec<[f64; 4]> = pairs
                    .iter()
                    .zip(&keep)
                    .filter(|(_, k)| **k)
                    .map(|(p, _)| *p)
                    .collect();
                let e = crate::wcsfit::rms(&crate::wcsfit::residuals(&fit, &inl));
                if debug_on() {
                    eprintln!(
                        "  grow: model choice {m:?} rms {e:.3} on {} inliers",
                        inl.len()
                    );
                }
                if e.is_finite() && inl.len() * 10 >= best.as_ref().map_or(0, |b| b.1.len()) * 9 {
                    best = Some((fit, inl, e));
                }
            }
        }
        best
    }

    /// Angular radius (rad) under `wcs` of the circle of `r` px around `pc`, clamped to the frame.
    fn zone_angle(&self, wcs: &Wcs, pc: [f64; 2], r: f64) -> f64 {
        let c = wcs.pixel_to_sky(pc[0], pc[1]);
        let cv = unit(c[0], c[1]);
        let mut ang: f64 = 0.0;
        for k in 0..16 {
            let a = k as f64 * TAU / 16.0;
            let x = (pc[0] + r * a.cos()).clamp(0.0, self.w - 1.0);
            let y = (pc[1] + r * a.sin()).clamp(0.0, self.h - 1.0);
            let sky = wcs.pixel_to_sky(x, y);
            ang = ang.max(angle(cv, unit(sky[0], sky[1])));
        }
        ang
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(w: u32, h: u32, n: usize) -> StarField {
        let mut q = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = || {
            q ^= q << 13;
            q ^= q >> 7;
            q ^= q << 17;
            (q >> 11) as f64 / (1u64 << 53) as f64
        };
        StarField {
            width: w,
            height: h,
            stars: (0..n)
                .map(|i| Star {
                    x: rnd() * w as f64,
                    y: rnd() * h as f64,
                    flux: (n - i) as f64,
                    ..Default::default()
                })
                .collect(),
            epoch: None,
        }
    }

    #[test]
    fn plan_skips_masked_and_sparse_windows_and_prefers_open_ones() {
        let f = field(2000, 1333, 3000);
        let all = plan(&f, None);
        assert!(all.len() >= WINDOWS.len());
        assert_eq!(all[0], [400.0, 0.2 * 1333.0, 1600.0, 0.8 * 1333.0]);
        // ground over the lower 45 %: the windows are fractions of the open sky above it
        let mut m = SkyMask::open(2000, 1333, 40, 20);
        for j in 11..20 {
            for i in 0..40 {
                m.blocked[j * 40 + i] = true;
            }
        }
        let p = plan(&f, Some(&m));
        assert!(p.len() >= WINDOWS.len());
        for r in &p {
            assert!(m.open_fraction(r[0], r[1], r[2], r[3]) > 0.99, "{r:?}");
        }
        // the ground with a tower on the right half: the windows over it come last or go
        let before_tower = p.len();
        for j in 3..11 {
            for i in 30..40 {
                m.blocked[j * 40 + i] = true;
            }
        }
        let p = plan(&f, Some(&m));
        assert!(p.len() < before_tower);
        for r in &p {
            assert!(m.open_fraction(r[0], r[1], r[2], r[3]) >= 0.5);
        }
        let first = p[0];
        assert!(m.open_fraction(first[0], first[1], first[2], first[3]) >= 0.9);
        // A narrow tower leaves the central window >90% open. Preserve it ahead
        // of the wholly open but strongly off-axis top-of-frame windows.
        let expected = [
            400.0,
            0.2 * (11.0 * m.cell[1]),
            1600.0,
            0.8 * (11.0 * m.cell[1]),
        ];
        assert!(
            first
                .iter()
                .zip(expected)
                .all(|(a, b)| (a - b).abs() < 1e-10),
            "{first:?}"
        );
        assert!(plan(&field(2000, 1333, 30), None).is_empty());
    }

    #[test]
    fn long_panoramas_get_compact_square_windows() {
        for (w, h) in [(2400, 600), (600, 2400)] {
            let f = field(w, h, 6000);
            let windows = plan(&f, None);
            assert!(windows.iter().any(|r| {
                let (rw, rh) = (r[2] - r[0], r[3] - r[1]);
                (rw - rh).abs() < 1e-8 && rw < 0.5 * w.min(h) as f64
            }));
            assert!(
                windows
                    .iter()
                    .all(|r| r[0] >= 0. && r[1] >= 0. && r[2] <= w as f64 && r[3] <= h as f64)
            );
        }
    }

    #[test]
    fn window_field_shifts_into_window_coordinates() {
        let f = field(1000, 800, 500);
        let r = [200.0, 100.0, 800.0, 500.0];
        let sub = window_field(&f, r);
        assert_eq!((sub.width, sub.height), (600, 400));
        let n = f
            .stars
            .iter()
            .filter(|s| s.x >= 200.0 && s.x < 800.0 && s.y >= 100.0 && s.y < 500.0)
            .count();
        assert_eq!(sub.stars.len(), n);
        assert!(
            sub.stars
                .iter()
                .all(|s| s.x >= 0.0 && s.x < 600.0 && s.y >= 0.0 && s.y < 400.0)
        );
        // brightness order kept
        assert!(sub.stars.windows(2).all(|p| p[0].flux >= p[1].flux));
    }
}
