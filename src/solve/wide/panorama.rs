//! Blind identification on tangent tiles of a cylinder. This changes the search
//! coordinates only; all fitting, validation and output use original pixels.
use super::*;
use nalgebra::{Matrix3, Vector3};

const TILE: f64 = 512.;
const FOCAL: f64 = 443.4050067376326; // 60-degree tangent tile

#[derive(Clone, Copy)]
struct Tile {
    focal: f64,
    horizon: f64,
    center: [f64; 2],
    c: [f64; 3],
    e: [f64; 3],
    n: [f64; 3],
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl Tile {
    fn new(width: f64, horizon: f64, fov: f64, center: [f64; 2]) -> Self {
        let focal = width / fov.to_radians();
        let (s, c) = ((horizon - center[1]) / focal).sin_cos();
        Self {
            focal,
            horizon,
            center,
            c: [c, 0., s],
            e: [0., 1., 0.],
            n: [-s, 0., c],
        }
    }
    fn ray(&self, p: [f64; 2]) -> Option<[f64; 3]> {
        let lon = (p[0] - self.center[0]) / self.focal;
        let lat = (self.horizon - p[1]) / self.focal;
        // A tile uses one longitude branch. In a near-360 panorama, polar
        // stars across the seam must not acquire a different original x.
        if lat.abs() >= PI / 2. || lon.abs() >= PI {
            return None;
        }
        Some([lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()])
    }
    fn forward(&self, p: [f64; 2]) -> Option<[f64; 2]> {
        let v = self.ray(p)?;
        let z = dot(v, self.c);
        if z <= 0. {
            return None;
        }
        let q = [
            TILE / 2. + FOCAL * dot(v, self.e) / z,
            TILE / 2. - FOCAL * dot(v, self.n) / z,
        ];
        q.iter().all(|&x| (0. ..TILE).contains(&x)).then_some(q)
    }
    fn inverse(&self, q: [f64; 2]) -> [f64; 2] {
        let x = (q[0] - TILE / 2.) / FOCAL;
        let y = -(q[1] - TILE / 2.) / FOCAL;
        let v = crate::sphere::normalize(std::array::from_fn(|k| {
            self.c[k] + x * self.e[k] + y * self.n[k]
        }));
        [
            self.center[0] + self.focal * v[1].atan2(v[0]),
            self.horizon - self.focal * v[2].asin(),
        ]
    }
}

pub(super) fn search(
    idx: &Index,
    wide: Option<&Index>,
    field: &StarField,
    cfg: &SolveConfig,
    t0: Instant,
    end: f64,
    progress: (&mut SolveStats, &mut usize),
) -> Option<Solution> {
    let (stats, trials) = progress;
    let index = wide.unwrap_or(idx);
    // A coarse companion is necessary: do not spend narrow-field searches here.
    if index.r7.to_degrees() * 60. < 4. {
        return None;
    }
    let (w, h) = (field.width as f64, field.height as f64);
    let mut widths: Vec<_> = field
        .stars
        .iter()
        .filter(|s| s.fwhm.is_finite() && s.fwhm > 0.)
        .map(|s| s.fwhm)
        .collect();
    widths.sort_by(f64::total_cmp);
    let psf_max = widths
        .get(widths.len() / 4)
        .map_or(f64::INFINITY, |p| 2. * p);
    for (cx, cy) in [(0.5, 0.45), (0.75, 0.45), (0.25, 0.45), (0.5, 0.25)] {
        for (fov, horizon_fraction) in [
            (240., 2. / 3.),
            (260., 2. / 3.),
            (220., 2. / 3.),
            (180., 2. / 3.),
            (300., 2. / 3.),
            (280., 2. / 3.),
            (320., 2. / 3.),
            (360., 2. / 3.),
            (240., 0.5),
            (240., 0.85),
        ] {
            let left = end - t0.elapsed().as_secs_f64() * 1e3;
            if left < MIN_WINDOW_MS {
                return None;
            }
            let tile = Tile::new(w, h * horizon_fraction, fov, [w * cx, h * cy]);
            let stars = field
                .stars
                .iter()
                .filter(|s| s.fwhm <= psf_max || !s.fwhm.is_finite())
                .filter(|s| cfg.mask.as_deref().is_none_or(|m| !m.is_blocked(s.x, s.y)))
                .filter_map(|s| {
                    let p = tile.forward([s.x, s.y])?;
                    Some(Star {
                        x: p[0],
                        y: p[1],
                        ..s.clone()
                    })
                })
                .collect::<Vec<_>>();
            if stars.len() < MIN_WINDOW_STARS {
                continue;
            }
            let sub = StarField {
                width: TILE as u32,
                height: TILE as u32,
                stars,
                epoch: field.epoch,
            };
            let c = SolveConfig {
                deadline_ms: (left * 0.25).clamp(MIN_WINDOW_MS, 180.),
                windows: false,
                keep_matches: true,
                extra_trials: *trials,
                mask: None,
                wide_index: None,
                min_fov_deg: WIDE_INDEX_FOV.0,
                max_fov_deg: WIDE_INDEX_FOV.1,
                ..cfg.clone()
            };
            let ws = solve_field(index, &sub, &c);
            *trials += ws.stats.verifications;
            accumulate(stats, &ws.stats);
            if debug_on() {
                eprintln!(
                    "  cylinder tile fov={fov} horizon={horizon_fraction:.2} ({cx},{cy}): {} {} pairs {:.1} ms",
                    ws.status, ws.n_matches, ws.stats.total_ms
                );
            }
            if ws.status == "ambiguous" {
                let mut sol = blank_solution("ambiguous", field.width, field.height);
                sol.margin_nats = ws.margin_nats;
                sol.n_candidates = ws.n_candidates;
                stats.total_ms = t0.elapsed().as_secs_f64() * 1e3;
                stats.verifications = *trials - cfg.extra_trials;
                sol.stats = stats.clone();
                return Some(sol);
            }
            if ws.status != "solved" {
                continue;
            }
            let pairs: Vec<_> = ws
                .matches
                .iter()
                .map(|m| {
                    let p = tile.inverse([m.x, m.y]);
                    [p[0], p[1], m.ra, m.dec]
                })
                .collect();
            let mut cross = Matrix3::zeros();
            let mut mean = [0.; 3];
            for (i, p) in pairs.iter().enumerate() {
                // The validation quarter cannot initialize the fitted axis either.
                if i % 4 == 0 {
                    continue;
                }
                let v = tile.ray([p[0], p[1]])?;
                let sky = unit(p[2], p[3]);
                cross += Vector3::from(v) * Vector3::from(sky).transpose();
                for k in 0..3 {
                    mean[k] += sky[k];
                }
            }
            let svd = cross.svd(true, true);
            let rotation = svd.u? * svd.v_t?;
            let (a, d) =
                crate::sphere::radec([rotation[(2, 0)], rotation[(2, 1)], rotation[(2, 2)]]);
            let cp = [w * cx, h * cy];
            let (ra, dec) = radec(crate::sphere::normalize(mean));
            let tangent = [ra, dec];
            let Some(seed) = crate::wcsfit::panorama::cylinder_seed(&pairs, cp, tangent, [a, d])
            else {
                continue;
            };
            let pairs: Vec<_> = pairs
                .into_iter()
                .filter(|p| {
                    seed.sky_to_pixel(p[2], p[3])
                        .is_some_and(|q| (q[0] - p[0]).hypot(q[1] - p[1]) < 2.5)
                })
                .collect();
            if pairs.len() < 32 {
                continue;
            }
            let extent = pairs.iter().fold(
                [
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                ],
                |mut b, p| {
                    b[0] = b[0].min(p[0]);
                    b[1] = b[1].min(p[1]);
                    b[2] = b[2].max(p[0]);
                    b[3] = b[3].max(p[1]);
                    b
                },
            );
            let win = [
                extent[0],
                extent[1],
                extent[2] - extent[0],
                extent[3] - extent[1],
            ];
            let c = SolveConfig {
                deadline_ms: (end - t0.elapsed().as_secs_f64() * 1e3).max(0.),
                extra_trials: *trials,
                ..cfg.clone()
            };
            let ctx = context(idx, field, &c, Instant::now());
            let Some(mut fc) = ctx.frame_catalog(&seed, ctx.frame_level(&seed)) else {
                continue;
            };
            if let Some(grown) = ctx.grow(&seed, win, 1., &mut fc, seed.projection()) {
                let retained = pairs
                    .iter()
                    .filter(|p| {
                        grown
                            .0
                            .sky_to_pixel(p[2], p[3])
                            .is_some_and(|q| (q[0] - p[0]).hypot(q[1] - p[1]) < 2.5)
                    })
                    .count();
                if retained * 10 >= pairs.len() * 9
                    && grown_holds(&ctx, &grown.0, &grown.1, grown.2)
                    && ctx.extension_evidence(&grown.0, extent)
                {
                    let mut sol = finish(&ctx, field, &c, &ws, win, grown)?;
                    stats.total_ms = t0.elapsed().as_secs_f64() * 1e3;
                    stats.verifications = *trials - cfg.extra_trials;
                    sol.stats = stats.clone();
                    return Some(sol);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identifies_cylinders_at_different_spans_and_crops() {
        let Ok(path) = std::env::var("SKYSHOT_INDEX") else {
            return;
        };
        let idx = Index::open(std::path::Path::new(&path)).unwrap();
        let Some(dir) = crate::index::companion_wide_dir(std::path::Path::new(&path)) else {
            return;
        };
        let wide = Index::open(&dir).unwrap();
        let mut catalog = vec![];
        idx.cone_stars([1., 0., 0.], PI, 8., 0., &mut catalog);
        // Both crops stay below the native pole; crossing it would correctly
        // fail the full-frame invertibility gate unless the dead band was masked.
        for (fov, horizon) in [(235., 580.), (285., 630.)] {
            let truth = Wcs {
                ra: 341.,
                dec: 0.,
                crpix: [1024., horizon],
                cd: [[-fov / 2048., 0.], [0., -fov / 2048.]],
                sip: None,
                proj: Proj::Car,
                native_pole: Some([251., 45.]),
                local_grid: None,
            };
            let stars = catalog
                .iter()
                .filter_map(|s| {
                    let (a, d) = radec(s.v);
                    let p = truth.sky_to_pixel(a, d)?;
                    (p[0] >= 0. && p[0] < 2048. && p[1] >= 0. && p[1] < 550.).then_some(Star {
                        x: p[0],
                        y: p[1],
                        flux: 10f64.powf(-0.4 * s.mag as f64),
                        sigma: 0.3,
                        ..Default::default()
                    })
                })
                .collect();
            let field = StarField {
                width: 2048,
                height: 964,
                stars,
                epoch: None,
            };
            let cfg = SolveConfig {
                deadline_ms: 5000.,
                keep_matches: true,
                ..Default::default()
            };
            let sol = search(
                &idx,
                Some(&wide),
                &field,
                &cfg,
                Instant::now(),
                5000.,
                (&mut SolveStats::default(), &mut 0),
            )
            .expect("deprojected tile must identify the panorama");
            let fit = sol.wcs.unwrap();
            for p in [[50., 100.], [1950., 100.], [50., 450.], [1950., 450.]] {
                let sky = truth.pixel_to_sky(p[0], p[1]);
                let q = fit.sky_to_pixel(sky[0], sky[1]).unwrap();
                assert!(
                    (q[0] - p[0]).hypot(q[1] - p[1]) < 2.,
                    "span {fov}, horizon {horizon}, {p:?}: {q:?}"
                );
            }
        }
    }
    #[test]
    fn cylinder_tiles_round_trip_original_pixels() {
        for fov in [180., 240., 300., 360.] {
            for center in [[1024., 434.], [1536., 240.]] {
                let tile = Tile::new(2048., 964. * 2. / 3., fov, center);
                let mut checked = 0;
                for i in 0..=40 {
                    for j in 0..=20 {
                        let p = [i as f64 * 51., j as f64 * 32.];
                        if let Some(q) = tile.forward(p) {
                            let back = tile.inverse(q);
                            assert!((back[0] - p[0]).hypot(back[1] - p[1]) < 1e-9);
                            checked += 1;
                        }
                    }
                }
                assert!(checked > 10);
                assert!(tile.ray([1024., tile.horizon - tile.focal * PI]).is_none());
            }
        }
    }
}
