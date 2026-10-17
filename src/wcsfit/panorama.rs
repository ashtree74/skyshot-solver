//! Recover a cylinder's axis from an already identified window. Only catalogue
//! correspondences are used; this does not generate or accept blind hypotheses.
use super::*;
use crate::{
    geometry::Projection,
    sphere::{radec, tangent_basis, unit},
};

/// Fit a rigid cylinder after a deprojected tile has been identified. Trim only
/// the training objective: a few accidental tile pairs must not tilt the axis.
/// The untouched quarter must independently agree before this seed can grow.
pub(crate) fn cylinder_seed(
    pairs: &[[f64; 4]],
    crpix: [f64; 2],
    tangent: [f64; 2],
    pole: [f64; 2],
) -> Option<Wcs> {
    if pairs.len() < 32 {
        return None;
    }
    let train: Vec<_> = pairs
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 4 != 0)
        .map(|(_, p)| *p)
        .collect();
    let test: Vec<_> = pairs.iter().step_by(4).copied().collect();
    let at = |pole| -> Option<(f64, Wcs)> {
        let spec = Projection {
            proj: Proj::Car,
            native_pole: Some(pole),
        };
        let mut sub = train.clone();
        let mut fit = fit_model(&sub, &[], crpix, tangent, 1, 0, spec)?;
        for _ in 0..3 {
            let errors = residuals(&fit, &train);
            let mut ids: Vec<_> = (0..train.len())
                .filter(|&i| errors[i].is_finite())
                .collect();
            ids.sort_by(|&a, &b| errors[a].total_cmp(&errors[b]));
            ids.truncate(train.len() * 4 / 5);
            if ids.len() < 18 {
                return None;
            }
            sub = ids.iter().map(|&i| train[i]).collect();
            fit = fit_model(&sub, &[], crpix, tangent, 1, 0, spec)?;
        }
        Some((rms(&residuals(&fit, &sub)), fit))
    };
    let (mut score, mut fit) = at(pole)?;
    let mut pole = pole;
    let mut step = 12f64.to_radians();
    for _ in 0..40 {
        let v = unit(pole[0], pole[1]);
        let (east, north) = tangent_basis(v);
        let old = score;
        for (dx, dy) in [(step, 0.), (-step, 0.), (0., step), (0., -step)] {
            let q = crate::sphere::normalize(std::array::from_fn(|k| {
                v[k] + dx * east[k] + dy * north[k]
            }));
            let (ra, dec) = radec(q);
            if let Some((s, f)) = at([ra, dec])
                && s < score
            {
                score = s;
                pole = [ra, dec];
                fit = f;
            }
        }
        if score >= old {
            step *= 0.5;
        }
        if step < 1e-6 {
            break;
        }
    }
    let errors = residuals(&fit, &test);
    if errors.iter().filter(|&&e| e.is_finite() && e < 2.5).count() * 10 < test.len() * 7 {
        return None;
    }
    // Return the retained training fit. Test stars never select the axis.
    Some(fit)
}

/// Rigid panorama candidates, best first. Fit the axis on three quarters of the
/// window's pairs and rank on the held-out quarter; distortion must not conceal
/// an incorrect axis during this search. Pole sign is immaterial (CD absorbs it).
pub(crate) fn candidates(
    pairs: &[[f64; 4]],
    crpix: [f64; 2],
    tangent: [f64; 2],
) -> Vec<Projection> {
    if pairs.len() < 24 {
        return vec![];
    }
    let train: Vec<_> = pairs
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 4 != 0)
        .map(|(_, p)| *p)
        .collect();
    let test: Vec<_> = pairs.iter().step_by(4).copied().collect();
    let mut out = vec![];
    for proj in [Proj::Car, Proj::Mer, Proj::Cyp] {
        let score = |pole: [f64; 2]| -> f64 {
            let spec = Projection {
                proj,
                native_pole: Some(pole),
            };
            let Some(fit) = fit_model(&train, &[], crpix, tangent, 1, 0, spec) else {
                return f64::INFINITY;
            };
            // Fit the axis on training pairs only. The test quarter must not
            // influence the grid search or its local refinement.
            let errors = residuals(&fit, &train);
            if errors.iter().any(|e| !e.is_finite()) {
                return f64::INFINITY;
            }
            rms(&errors)
        };
        let mut best = (f64::INFINITY, [0.0; 2]);
        let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        for i in 0..48 {
            let z = (i as f64 + 0.5) / 48.0;
            let r = (1.0 - z * z).sqrt();
            let (ra, dec) = radec([
                r * (golden * i as f64).cos(),
                r * (golden * i as f64).sin(),
                z,
            ]);
            let pole = [ra, dec];
            let e = score(pole);
            if e < best.0 {
                best = (e, pole);
            }
        }
        let mut step = 12f64.to_radians();
        for _ in 0..24 {
            let v = unit(best.1[0], best.1[1]);
            let (east, north) = tangent_basis(v);
            let old = best.0;
            for (dx, dy) in [(step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)] {
                let q = crate::sphere::normalize(std::array::from_fn(|k| {
                    v[k] + dx * east[k] + dy * north[k]
                }));
                let (ra, dec) = radec(q);
                let pole = [ra, dec];
                let e = score(pole);
                if e < best.0 {
                    best = (e, pole);
                }
            }
            if best.0 >= old {
                step *= 0.5;
            }
        }
        if best.0.is_finite() {
            let spec = Projection {
                proj,
                native_pole: Some(best.1),
            };
            if let Some(fit) = fit_model(&train, &[], crpix, tangent, 1, 0, spec) {
                // Compare all projections on the same untouched test set.
                let errors = residuals(&fit, &test);
                if errors.iter().all(|e| e.is_finite()) {
                    out.push((rms(&errors), spec));
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out.into_iter().map(|(_, p)| p).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cylinder_axis_ignores_mismatches_and_requires_held_out_agreement() {
        let truth = Wcs {
            ra: 301.,
            dec: 30.,
            crpix: [1024., 400.],
            cd: [[-0.11, 0.001], [0.002, -0.11]],
            sip: None,
            proj: Proj::Car,
            native_pole: Some([252., 46.]),
            local_grid: None,
        };
        let mut pairs: Vec<_> = (0..12)
            .flat_map(|i| (0..10).map(move |j| [600. + i as f64 * 65., 150. + j as f64 * 32.]))
            .map(|p| {
                let s = truth.pixel_to_sky(p[0], p[1]);
                [p[0], p[1], s[0], s[1]]
            })
            .collect();
        for (i, p) in pairs.iter_mut().enumerate() {
            if i % 9 == 1 {
                p[2] += 12.;
            }
        }
        let fit = cylinder_seed(&pairs, truth.crpix, [truth.ra, truth.dec], [246., 43.]).unwrap();
        for p in [[100., 100.], [1950., 100.], [100., 600.], [1950., 600.]] {
            let s = truth.pixel_to_sky(p[0], p[1]);
            let q = fit.sky_to_pixel(s[0], s[1]).unwrap();
            assert!((q[0] - p[0]).hypot(q[1] - p[1]) < 0.5, "{p:?}: {q:?}");
        }
        for p in pairs.iter_mut().step_by(4) {
            p[2] += 40.;
        }
        assert!(cylinder_seed(&pairs, truth.crpix, [truth.ra, truth.dec], [246., 43.]).is_none());
    }
    #[test]
    fn recovers_cylindrical_axes_from_a_verified_window() {
        for proj in [Proj::Car, Proj::Mer, Proj::Cyp] {
            for pole in [[40., 60.], [140., -15.]] {
                let truth = Wcs {
                    ra: 283.,
                    dec: -4.,
                    crpix: [600., 400.],
                    cd: [[-0.06, 0.002], [0.003, 0.065]],
                    sip: None,
                    proj,
                    native_pole: Some(pole),
                    local_grid: None,
                };
                let pairs: Vec<_> = (0..11)
                    .flat_map(|i| (0..9).map(move |j| [i as f64 * 120., j as f64 * 100.]))
                    .map(|px| {
                        let sky = truth.pixel_to_sky(px[0], px[1]);
                        [px[0], px[1], sky[0], sky[1]]
                    })
                    .collect();
                let specs = candidates(&pairs, truth.crpix, [truth.ra, truth.dec]);
                let best = specs[0];
                let fit =
                    fit_model(&pairs, &[], truth.crpix, [truth.ra, truth.dec], 1, 0, best).unwrap();
                let e = rms(&residuals(&fit, &pairs));
                assert_eq!(best.proj, proj, "axis fit rms {e}");
                assert!(e < 0.5, "{proj:?} {pole:?}: axis fit rms {e}");
                assert!(fit.native_pole.is_some());
                let shifted = fit_model(
                    &pairs,
                    &[],
                    [620., 430.],
                    [truth.ra, truth.dec],
                    1,
                    0,
                    truth.projection(),
                )
                .unwrap();
                assert!(
                    rms(&residuals(&shifted, &pairs)) < 1e-5,
                    "axis survives reference-point iteration"
                );
            }
        }
    }
}
