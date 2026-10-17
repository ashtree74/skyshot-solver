//! Residual registration after blind identification. Independent local translation
//! evidence recovers stitched edge tiles that the global inlier gate excluded.
//! Bright stars and a deterministic quarter of the catalogue never train the map.
use crate::{pixelgrid::PixelGrid, select::Grid2, solve::poisson_log_odds};
use nalgebra::{Matrix3, Vector3};
use std::{collections::HashSet, time::Instant};

struct Tie {
    cat: usize,
    image: u32,
    distance: f64,
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Run once after the image pipeline has accepted a blind solution. Use the primary,
/// deeper catalogue and the remaining frame budget, not a window's short search share.
pub(crate) fn refine_solution(
    idx: &crate::index::Index,
    solution: &mut crate::solve::Solution,
    field: &crate::types::StarField,
    cfg: &crate::solve::SolveConfig,
    mask: Option<&crate::skymask::SkyMask>,
    remaining_ms: f64,
) {
    use crate::{
        sphere::{radec, unit},
        types::Proj,
    };
    if !cfg.allow_local_grid || solution.status != "solved" || remaining_ms < 30. {
        return;
    }
    let Some(base) = solution.wcs.as_ref() else {
        return;
    };
    if (base.proj != Proj::Stg && !base.proj.is_cylindrical()) || base.local_grid.is_some() {
        return;
    }
    let start = Instant::now();
    let (width, height) = (field.width as f64, field.height as f64);
    let visible = |p: [f64; 2]| {
        p[0] >= 0.
            && p[0] < width
            && p[1] >= 0.
            && p[1] < height
            && mask.is_none_or(|m| !m.is_blocked(p[0], p[1]))
    };
    let mut image: Vec<_> = field.stars.iter().filter(|s| visible([s.x, s.y])).collect();
    image.sort_by(|a, b| b.flux.total_cmp(&a.flux));
    image.truncate(cfg.max_stars);
    let image: Vec<_> = image.iter().map(|s| [s.x, s.y]).collect();
    let mut stars = vec![];
    idx.cone_stars(
        unit(base.ra, base.dec),
        std::f64::consts::PI,
        cfg.mag_limit.unwrap_or(9.5).min(9.5),
        0.,
        &mut stars,
    );
    let prepared = base.prepare();
    let cat: Vec<_> = stars
        .iter()
        .filter_map(|s| {
            let (ra, dec) = radec(s.v);
            let p = prepared.sky_to_pixel_fast(ra, dec)?;
            visible(p).then_some([p[0], p[1], ra, dec, s.mag as f64])
        })
        .collect();
    let Some(grid) = fit(
        &cat,
        &image,
        width,
        height,
        visible,
        mask,
        remaining_ms - start.elapsed().as_secs_f64() * 1000.,
    ) else {
        return;
    };
    let mut candidate = base.clone();
    candidate.local_grid = Some(grid);
    for y in 0..=12 {
        for x in 0..=20 {
            let p = [
                x as f64 * (width - 1.) / 20.,
                y as f64 * (height - 1.) / 12.,
            ];
            if !visible(p) {
                continue;
            }
            let s = candidate.pixel_to_sky(p[0], p[1]);
            let Some(q) = candidate.sky_to_pixel(s[0], s[1]) else {
                return;
            };
            if (q[0] - p[0]).hypot(q[1] - p[1]) > 1e-3 {
                return;
            }
        }
    }
    for m in &mut solution.matches {
        if let Some(q) = candidate.sky_to_pixel(m.ra, m.dec) {
            m.residual_px = (m.x - q[0]).hypot(m.y - q[1]);
        }
    }
    if !solution.matches.is_empty() {
        solution.rms_px = (solution
            .matches
            .iter()
            .map(|m| m.residual_px.powi(2))
            .sum::<f64>()
            / solution.matches.len() as f64)
            .sqrt();
    }
    let center = candidate.pixel_to_sky((width - 1.) / 2., (height - 1.) / 2.);
    solution.ra = center[0];
    solution.dec = center[1];
    // The residual field now includes stitch warps, so a physical atmosphere fit
    // measured against the base model is no longer an independent measurement.
    solution.refraction = None;
    solution.wcs = Some(candidate);
}

/// Catalogue rows [base x, base y, ra, dec, magnitude], image points in brightness order.
/// `visible` excludes terrain on both sides. None leaves the verified base model intact.
pub(crate) fn fit(
    cat: &[[f64; 5]],
    image: &[[f64; 2]],
    width: f64,
    height: f64,
    visible: impl Fn([f64; 2]) -> bool,
    mask: Option<&crate::skymask::SkyMask>,
    budget_ms: f64,
) -> Option<PixelGrid> {
    if cat.len() < 100 || image.len() < 100 || budget_ms < 20. {
        return None;
    }
    let start = Instant::now();
    let ids: Vec<_> = (0..image.len() as u32)
        .filter(|i| visible(image[*i as usize]))
        .collect();
    let detections = Grid2::build(image, &ids, width, height, 3.);
    let train = |i: usize| cat[i][4] >= 5.5 && !i.is_multiple_of(4);
    let training: Vec<_> = (0..cat.len())
        .filter(|&i| train(i) && visible([cat[i][0], cat[i][1]]))
        .collect();
    let mut ties: Vec<Option<Tie>> = (0..cat.len()).map(|_| None).collect();
    let spacing = (width.max(height) / 16.).max(48.);
    let radius = spacing.max(96.);
    let search = (width.max(height) * 0.016).clamp(8., 32.);
    let bin = search / 16.;
    let (nx, ny) = (
        (width / spacing).ceil() as usize,
        (height / spacing).ceil() as usize,
    );
    let threshold = 14. + (((nx + 1) * (ny + 1) * 33 * 33) as f64).ln();
    let mut nearby = vec![];
    for gy in 0..=ny {
        for gx in 0..=nx {
            if start.elapsed().as_secs_f64() * 1000. > budget_ms {
                return None;
            }
            let center = [
                (gx as f64 * spacing).min(width - 1.),
                (gy as f64 * spacing).min(height - 1.),
            ];
            let mut local: Vec<_> = training
                .iter()
                .copied()
                .filter(|&i| {
                    (cat[i][0] - center[0]).abs() < radius && (cat[i][1] - center[1]).abs() < radius
                })
                .collect();
            // Match catalogue depth to detections in this patch. A deep Gaia
            // cone contains many unresolved faint companions; letting all of
            // them vote biases the translation towards dense chance peaks.
            let local_detections = ids
                .iter()
                .filter(|&&i| {
                    (image[i as usize][0] - center[0]).abs() < radius
                        && (image[i as usize][1] - center[1]).abs() < radius
                })
                .count();
            local.sort_by(|&a, &b| cat[a][4].total_cmp(&cat[b][4]));
            local.truncate(local_detections);
            if local.len() < 12 {
                continue;
            }
            let mut hist = [0u32; 33 * 33];
            for &i in &local {
                let p = [cat[i][0], cat[i][1]];
                detections.knn(image, p, 32, u32::MAX, &mut nearby);
                for &(distance, j) in &nearby {
                    if distance > search * search {
                        break;
                    }
                    let d = [
                        (image[j as usize][0] - p[0]) / bin,
                        (image[j as usize][1] - p[1]) / bin,
                    ];
                    let x = (d[0].round() as i32 + 16).clamp(0, 32) as usize;
                    let y = (d[1].round() as i32 + 16).clamp(0, 32) as usize;
                    hist[y * 33 + x] += 1;
                }
            }
            let mut best = (0u32, 16usize, 16usize);
            for y in 1..32 {
                for x in 1..32 {
                    let mut value = 0;
                    for (dy, wy) in [(-1, 1), (0, 2), (1, 1)] {
                        for (dx, wx) in [(-1, 1), (0, 2), (1, 1)] {
                            value += hist[(y as i32 + dy) as usize * 33 + (x as i32 + dx) as usize]
                                * wy
                                * wx;
                        }
                    }
                    if value > best.0 {
                        best = (value, x, y);
                    }
                }
            }
            let mut shift = [(best.1 as f64 - 16.) * bin, (best.2 as f64 - 16.) * bin];
            let mut matched = vec![];
            for _ in 0..2 {
                matched.clear();
                let mut used = HashSet::new();
                for &i in &local {
                    let q = [cat[i][0] + shift[0], cat[i][1] + shift[1]];
                    if let Some((_, j)) = detections.nearest_within(image, q, 2.8, u32::MAX)
                        && used.insert(j)
                        && visible(image[j as usize])
                    {
                        matched.push((i, j));
                    }
                }
                if matched.len() < 12 {
                    break;
                }
                shift = std::array::from_fn(|k| {
                    median(
                        matched
                            .iter()
                            .map(|&(i, j)| image[j as usize][k] - cat[i][k])
                            .collect(),
                    )
                });
            }
            if matched.len() < 12 {
                continue;
            }
            let x0 = (center[0] - radius - search).max(0.);
            let x1 = (center[0] + radius + search).min(width);
            let y0 = (center[1] - radius - search).max(0.);
            let y1 = (center[1] + radius + search).min(height);
            let n_image = ids
                .iter()
                .filter(|&&j| {
                    image[j as usize][0] >= x0
                        && image[j as usize][0] < x1
                        && image[j as usize][1] >= y0
                        && image[j as usize][1] < y1
                })
                .count();
            let area = ((x1 - x0) * (y1 - y0)
                - mask.map_or(0., |m| m.blocked_area(x0, y0, x1, y1)))
            .max(1.);
            let lambda =
                local.len() as f64 * n_image as f64 * std::f64::consts::PI * 2.8 * 2.8 / area;
            if cfg!(test) && gy < 2 && gx < 3 {
                eprintln!(
                    "  patch {gx},{gy}: {} candidates, {} matches, shift {:?}, ll {:.1}/{threshold:.1}",
                    local.len(),
                    matched.len(),
                    shift,
                    poisson_log_odds(matched.len(), lambda)
                );
            }
            if poisson_log_odds(matched.len(), lambda) < threshold {
                continue;
            }
            for (i, j) in matched {
                let distance = (cat[i][0] - center[0]).hypot(cat[i][1] - center[1]);
                if ties[i].as_ref().is_none_or(|t| distance < t.distance) {
                    ties[i] = Some(Tie {
                        cat: i,
                        image: j,
                        distance,
                    });
                }
            }
        }
    }
    let mut seen = HashSet::new();
    let ties: Vec<_> = ties
        .into_iter()
        .flatten()
        .filter(|t| seen.insert(t.image))
        .collect();
    if cfg!(test) || crate::solve::debug_on() {
        eprintln!("  registration: {} local training ties", ties.len());
    }
    if ties.len() < 80 {
        return None;
    }
    let q: Vec<_> = ties
        .iter()
        .map(|t| [cat[t.cat][0], cat[t.cat][1]])
        .collect();
    let delta: Vec<_> = ties
        .iter()
        .map(|t| {
            [
                image[t.image as usize][0] - cat[t.cat][0],
                image[t.image as usize][1] - cat[t.cat][1],
            ]
        })
        .collect();
    let tie_ids: Vec<_> = (0..ties.len() as u32).collect();
    let tie_grid = Grid2::build(&q, &tie_ids, width, height, 3.);
    let cols = 65;
    let rows = ((height / width * 64.).ceil() as usize + 1).clamp(2, 129);
    let step = [
        (width - 1.) / (cols - 1) as f64,
        (height - 1.) / (rows - 1) as f64,
    ];
    let bandwidth = (width.max(height) / 64.).max(8.);
    let mut offsets = vec![];
    for y in 0..rows {
        if start.elapsed().as_secs_f64() * 1000. > budget_ms {
            return None;
        }
        for x in 0..cols {
            let at = [x as f64 * step[0], y as f64 * step[1]];
            tie_grid.knn(&q, at, 32, u32::MAX, &mut nearby);
            let local: Vec<_> = nearby
                .iter()
                .filter(|(distance, _)| *distance <= (3. * bandwidth).powi(2))
                .map(|(_, i)| *i as usize)
                .collect();
            if local.len() < 6 {
                offsets.push([0.; 2]);
                continue;
            }
            let median_delta: [f64; 2] =
                std::array::from_fn(|k| median(local.iter().map(|&i| delta[i][k]).collect()));
            let mut lhs = Matrix3::zeros();
            let mut rhs = [Vector3::zeros(); 2];
            let mut count = 0;
            let mut weight = 0.;
            for &(distance, i) in &nearby {
                if distance > (3. * bandwidth).powi(2) {
                    break;
                }
                let i = i as usize;
                // A patch can contain faint chance neighbours even after its
                // translation has strong evidence. Robust local fitting keeps
                // these from pulling an edge into a different stitch's field.
                if (delta[i][0] - median_delta[0]).hypot(delta[i][1] - median_delta[1]) > 2. {
                    continue;
                }
                let z = Vector3::new(
                    1.,
                    (q[i][0] - at[0]) / bandwidth,
                    (q[i][1] - at[1]) / bandwidth,
                );
                let w = (-distance / (2. * bandwidth * bandwidth)).exp();
                lhs += w * z * z.transpose();
                for k in 0..2 {
                    rhs[k] += w * z * delta[i][k];
                }
                count += 1;
                weight += w;
            }
            lhs[(1, 1)] += 0.2;
            lhs[(2, 2)] += 0.2;
            let d = if count >= 6 && weight >= 1.5 {
                lhs.try_inverse()
                    .map(|inv| [(inv * rhs[0])[0], (inv * rhs[1])[0]])
                    .unwrap_or([0.; 2])
            } else {
                [0.; 2]
            };
            offsets.push(if d[0].hypot(d[1]) < 0.5 { [0.; 2] } else { d });
        }
    }
    let mut grid = PixelGrid {
        cols,
        rows,
        origin: [0.; 2],
        step,
        offsets,
    };
    // Select local corrections against held-out stars before accepting the whole
    // map. A good global gain must not drag an already aligned bright star into
    // an adjacent stitch's correction, or extrapolate into an unsupported cell.
    let validation: Vec<_> = cat
        .iter()
        .enumerate()
        .filter(|(i, c)| !train(*i) && visible([c[0], c[1]]))
        .filter_map(|(_, c)| {
            let p = [c[0], c[1]];
            let corrected = grid.forward(p);
            let before = detections
                .nearest_within(image, p, search, u32::MAX)
                .map_or(search, |m| m.0.sqrt());
            let after = detections
                .nearest_within(image, corrected, search, u32::MAX)
                .map_or(search, |m| m.0.sqrt());
            (before < 2. || after < 2.).then_some((p, before, after, c[4] < 5.5))
        })
        .collect();
    let validation_points: Vec<_> = validation.iter().map(|v| v.0).collect();
    let validation_ids: Vec<_> = (0..validation.len() as u32).collect();
    let validation_grid = Grid2::build(&validation_points, &validation_ids, width, height, 3.);
    for y in 0..rows {
        if start.elapsed().as_secs_f64() * 1000. > budget_ms {
            return None;
        }
        for x in 0..cols {
            let at = [x as f64 * step[0], y as f64 * step[1]];
            let mut before = 0.;
            let mut after = 0.;
            let mut count = 0;
            let mut weight = 0.;
            let mut veto = false;
            validation_grid.knn(&validation_points, at, 64, u32::MAX, &mut nearby);
            for &(distance, j) in &nearby {
                if distance > (2. * bandwidth).powi(2) {
                    break;
                }
                let (_, a, b, bright) = validation[j as usize];
                let w = (-distance / (2. * bandwidth * bandwidth)).exp();
                before += w * a.min(6.).powi(2);
                after += w * b.min(6.).powi(2);
                weight += w;
                count += 1;
                if bright && a < 1. && b > 1.5 {
                    veto = true;
                }
            }
            if veto || count < 3 || weight < 0.5 || after >= 0.8 * before {
                grid.offsets[y * cols + x] = [0.; 2];
            }
        }
    }
    for _ in 0..4 {
        if grid.validate().is_ok() {
            break;
        }
        let old = grid.offsets.clone();
        for y in 0..rows {
            for x in 0..cols {
                for (k, old_value) in old[y * cols + x].iter().copied().enumerate() {
                    let mut sum = 0.;
                    let mut n = 0.;
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let xx = (x as i32 + dx).clamp(0, cols as i32 - 1) as usize;
                            let yy = (y as i32 + dy).clamp(0, rows as i32 - 1) as usize;
                            sum += old[yy * cols + xx][k];
                            n += 1.;
                        }
                    }
                    grid.offsets[y * cols + x][k] = 0.5 * old_value + 0.5 * sum / n;
                }
            }
        }
    }
    grid.validate().ok()?;
    // Discovery and fitting above never see these stars. Judge gains in affected
    // regions, not a whole-image RMS diluted by thousands of unchanged positions.
    let mut before = vec![];
    let mut after = vec![];
    let mut gained = 0;
    let mut lost = 0;
    let mut seen_before = HashSet::new();
    let mut seen_after = HashSet::new();
    for (_, c) in cat.iter().enumerate().filter(|(i, _)| !train(*i)) {
        let p = [c[0], c[1]];
        if !visible(p) {
            continue;
        }
        let corrected = grid.forward(p);
        let a = detections.nearest_within(image, p, search, u32::MAX);
        let b = detections.nearest_within(image, corrected, search, u32::MAX);
        let a = a
            .filter(|x| x.0 >= 4. || seen_before.insert(x.1))
            .map_or(search, |x| x.0.sqrt());
        let b = b
            .filter(|x| x.0 >= 4. || seen_after.insert(x.1))
            .map_or(search, |x| x.0.sqrt());
        if b < 2. && a >= 2. {
            gained += 1;
        }
        if a < 2. && b >= 2. {
            lost += 1;
        }
        if (a < 2. || b < 2.) && grid.offset(p).iter().any(|d| d.abs() > 2.) {
            before.push(a.min(6.));
            after.push(b.min(6.));
        }
    }
    if cfg!(test) || crate::solve::debug_on() {
        eprintln!(
            "  registration: held-out {} affected, {gained} gained, {lost} lost, median {:?}",
            after.len(),
            if after.is_empty() {
                None
            } else {
                Some(median(after.clone()))
            }
        );
    }
    if after.len() < 12 || gained < lost + 8 || median(after.clone()) > 1.5 {
        return None;
    }
    let rms = |v: &[f64]| (v.iter().map(|e| e * e).sum::<f64>() / v.len() as f64).sqrt();
    if rms(&after) > 0.8 * rms(&before) {
        return None;
    }
    if cfg!(test) || crate::solve::debug_on() {
        eprintln!(
            "  local grid: {} train ties, {} held-out affected stars, {gained} gained/{lost} lost, RMS {:.3} -> {:.3}",
            ties.len(),
            after.len(),
            rms(&before),
            rms(&after)
        );
    }
    Some(grid)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn synthetic(shifted: bool, unrelated: bool) -> (Vec<[f64; 5]>, Vec<[f64; 2]>) {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(73);
        let mut cat = vec![];
        let mut image = vec![];
        for i in 0..3600 {
            let x = rng.gen_range(10.0..1014.0);
            let y = rng.gen_range(10.0..630.0);
            cat.push([x, y, 0., 0., if i % 13 == 0 { 4.0 } else { 7.0 }]);
            let f = if shifted {
                (-((x / 170.0f64).powi(2) + (y / 150.0f64).powi(2))).exp()
            } else {
                0.
            };
            image.push(if unrelated {
                [rng.gen_range(0.0..1024.), rng.gen_range(0.0..640.)]
            } else {
                [
                    x - 7. * f + rng.gen_range(-0.05..0.05),
                    y + 8. * f + rng.gen_range(-0.05..0.05),
                ]
            });
        }
        (cat, image)
    }
    #[test]
    fn held_out_stars_validate_a_local_edge_model() {
        let (cat, image) = synthetic(true, false);
        let grid = fit(
            &cat,
            &image,
            1024.,
            640.,
            |p| p[0] >= 0. && p[0] < 1024. && p[1] >= 0. && p[1] < 640.,
            None,
            10000.,
        )
        .expect("coherent edge distortion");
        let errors: Vec<_> = cat
            .iter()
            .enumerate()
            .filter(|(i, c)| i % 4 == 0 || c[4] < 5.5)
            .filter(|(_, c)| c[0] < 200. && c[1] < 180.)
            .map(|(i, c)| {
                let p = grid.forward([c[0], c[1]]);
                (p[0] - image[i][0]).hypot(p[1] - image[i][1])
            })
            .collect();
        assert!(errors.len() > 12);
        assert!(median(errors) < 0.5);
    }
    #[test]
    fn unchanged_or_unrelated_stars_cannot_create_a_local_warp() {
        for unrelated in [false, true] {
            let (cat, image) = synthetic(false, unrelated);
            assert!(fit(&cat, &image, 1024., 640., |_| true, None, 10000.).is_none());
            assert!(fit(&cat, &image, 1024., 640., |_| true, None, 0.).is_none());
        }
    }
    #[test]
    fn supplied_edge_registration() {
        let Ok(path) = std::env::var("SKYSHOT_REGISTRATION_CATALOG") else {
            return;
        };
        let catalog: Vec<[f64; 5]> = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let catalog: Vec<_> = catalog
            .into_iter()
            .filter(|c| c[4] <= 9.5 && c[1] < 820.)
            .collect();
        let stars: crate::types::StarField = serde_json::from_slice(
            &std::fs::read(std::env::var("SKYSHOT_REGISTRATION_STARS").unwrap()).unwrap(),
        )
        .unwrap();
        let image: Vec<_> = stars
            .stars
            .iter()
            .filter(|s| s.y < 820.)
            .map(|s| [s.x, s.y])
            .collect();
        let grid = fit(
            &catalog,
            &image,
            2048.,
            1365.,
            |p| p[0] >= 0. && p[0] < 2048. && p[1] >= 0. && p[1] < 820.,
            None,
            30000.,
        )
        .expect("independently supported local correction");
        let at = grid.forward([82.9808375, 36.93983808]);
        assert!(
            (at[0] - 70.3538293).hypot(at[1] - 51.3342793) < 1.5,
            "bright held-out Biham: {at:?}"
        );
        std::fs::write(
            "outputs/panorama-edge/native-grid.json",
            serde_json::to_vec_pretty(&grid).unwrap(),
        )
        .unwrap();
    }
}
