//! Stereographic fitting with a free optical centre and reference pixel.
//! A cropped panorama need not contain its projection centre at the image centre.
//! For a unit sky vector v, stereographic pixel coordinates have the form
//! x = (a.v + d) / (1 + c.v), where c is the optical axis. An algebraic fit
//! initializes c; a two-parameter geometric fit then profiles out CD and CRPIX.
use crate::{
    geometry::PreparedProjection,
    sphere::{normalize, radec, tangent_basis, unit},
    types::{Proj, Wcs},
};
use nalgebra::{DMatrix, DVector, Matrix2, Vector2};

struct GeometricFit {
    wcs: Wcs,
    residuals: Vec<[f64; 2]>,
    score: f64,
}

fn fit(pairs: &[[f64; 4]], weights: &[f64]) -> Option<Wcs> {
    if pairs.len() < 24 || pairs.iter().flatten().any(|v| !v.is_finite()) {
        return None;
    }
    let cp = [
        pairs.iter().map(|p| p[0]).sum::<f64>() / pairs.len() as f64,
        pairs.iter().map(|p| p[1]).sum::<f64>() / pairs.len() as f64,
    ];
    let scale = pairs
        .iter()
        .map(|p| (p[0] - cp[0]).abs().max((p[1] - cp[1]).abs()))
        .fold(1., f64::max);
    let mut m = DMatrix::zeros(2 * pairs.len(), 11);
    let mut y = DVector::zeros(2 * pairs.len());
    for (i, p) in pairs.iter().enumerate() {
        let v = unit(p[2], p[3]);
        let x = (p[0] - cp[0]) / scale;
        let yy = (p[1] - cp[1]) / scale;
        for j in 0..3 {
            m[(2 * i, j)] = v[j];
            m[(2 * i + 1, j + 4)] = v[j];
            m[(2 * i, j + 8)] = -x * v[j];
            m[(2 * i + 1, j + 8)] = -yy * v[j];
        }
        m[(2 * i, 3)] = 1.;
        m[(2 * i + 1, 7)] = 1.;
        y[2 * i] = x;
        y[2 * i + 1] = yy;
    }
    let svd = m.svd(true, true);
    let largest = svd.singular_values[0];
    if svd.singular_values[10] < largest * 1e-11 {
        return None;
    }
    let p = svd.solve(&y, largest * 1e-12).ok()?;
    let c = normalize([p[8], p[9], p[10]]);
    if c.iter().any(|x| !x.is_finite()) {
        return None;
    }
    // A few mismatches can put the algebraic axis near the patch's antipode,
    // where the geometric objective has a poor local minimum. The mean sky
    // direction gives an independent, nonsingular start; compare pixel errors.
    let mean = normalize(pairs.iter().fold([0.; 3], |mut sum, p| {
        let v = unit(p[2], p[3]);
        for k in 0..3 {
            sum[k] += v[k];
        }
        sum
    }));
    geometric_fit(pairs, weights, c)
        .into_iter()
        .chain(geometric_fit(pairs, weights, mean))
        .min_by(|a, b| a.score.total_cmp(&b.score))
        .map(|fit| fit.wcs)
}

fn geometric_fit(pairs: &[[f64; 4]], weights: &[f64], mut c: [f64; 3]) -> Option<GeometricFit> {
    let mut current = at(pairs, weights, c)?;
    let mut damping = 1e-3;
    for _ in 0..60 {
        let (e, n) = tangent_basis(c);
        let eps = 1e-4;
        let ce = normalize(std::array::from_fn(|k| c[k] + eps * e[k]));
        let cn = normalize(std::array::from_fn(|k| c[k] + eps * n[k]));
        let er = at(pairs, weights, ce)?;
        let nr = at(pairs, weights, cn)?;
        let mut h = Matrix2::zeros();
        let mut g = Vector2::zeros();
        for i in 0..pairs.len() {
            let w = weights.get(i).copied().unwrap_or(1.);
            for axis in 0..2 {
                let j = Vector2::new(
                    (er.residuals[i][axis] - current.residuals[i][axis]) / eps,
                    (nr.residuals[i][axis] - current.residuals[i][axis]) / eps,
                );
                h += w * j * j.transpose();
                g += w * j * current.residuals[i][axis];
            }
        }
        let mut progressed = false;
        for _ in 0..12 {
            let hd = h + Matrix2::new(
                damping * h[(0, 0)].max(1e-10),
                0.,
                0.,
                damping * h[(1, 1)].max(1e-10),
            );
            let mut step = hd.lu().solve(&(-g))?;
            let norm = step.norm();
            if norm > 0.3 {
                step *= 0.3 / norm;
            }
            if step.norm() < 1e-9 {
                return Some(current);
            }
            let next = normalize(std::array::from_fn(|k| {
                c[k] + step[0] * e[k] + step[1] * n[k]
            }));
            if let Some(candidate) = at(pairs, weights, next)
                && candidate.score < current.score
            {
                let improvement = current.score - candidate.score;
                c = next;
                current = candidate;
                damping = (damping * 0.3).max(1e-9);
                progressed = true;
                if improvement < 1e-10 {
                    return Some(current);
                }
                break;
            }
            damping *= 10.;
        }
        if !progressed {
            break;
        }
    }
    Some(current)
}

fn at(pairs: &[[f64; 4]], weights: &[f64], center: [f64; 3]) -> Option<GeometricFit> {
    let (ra, dec) = radec(center);
    let proj = PreparedProjection::new(Proj::Stg, [ra, dec]);
    let q: Vec<_> = pairs
        .iter()
        .map(|p| proj.project(p[2], p[3]))
        .collect::<Option<_>>()?;
    let mut sw = 0.;
    let (mut qm, mut pm) = ([0.; 2], [0.; 2]);
    for (i, p) in pairs.iter().enumerate() {
        let w = weights.get(i).copied().unwrap_or(1.);
        sw += w;
        for j in 0..2 {
            qm[j] += w * q[i][j];
            pm[j] += w * p[j];
        }
    }
    if !(sw.is_finite() && sw > 0.) {
        return None;
    }
    for j in 0..2 {
        qm[j] /= sw;
        pm[j] /= sw;
    }
    let mut cov = Matrix2::zeros();
    let mut cross = Matrix2::zeros();
    for (i, p) in pairs.iter().enumerate() {
        let w = weights.get(i).copied().unwrap_or(1.);
        let a = Vector2::new(q[i][0] - qm[0], q[i][1] - qm[1]);
        let b = Vector2::new(p[0] - pm[0], p[1] - pm[1]);
        cov += w * a * a.transpose();
        cross += w * b * a.transpose();
    }
    let a = cross * cov.try_inverse()?;
    let cd = a.try_inverse()?;
    let cp = Vector2::new(pm[0], pm[1]) - a * Vector2::new(qm[0], qm[1]);
    let mut residuals = Vec::with_capacity(pairs.len());
    let mut score = 0.;
    for (i, p) in pairs.iter().enumerate() {
        let pred = a * Vector2::new(q[i][0], q[i][1]) + cp;
        let r = [pred[0] - p[0], pred[1] - p[1]];
        score += weights.get(i).copied().unwrap_or(1.) * (r[0] * r[0] + r[1] * r[1]);
        residuals.push(r);
    }
    if !score.is_finite() {
        return None;
    }
    Some(GeometricFit {
        wcs: Wcs {
            ra,
            dec,
            crpix: [cp[0], cp[1]],
            cd: [[cd[(0, 0)], cd[(0, 1)]], [cd[(1, 0)], cd[(1, 1)]]],
            sip: None,
            proj: Proj::Stg,
            native_pole: None,
            local_grid: None,
        },
        residuals,
        score,
    })
}

/// Robust rigid STG fit. Only already identified catalogue correspondences enter here.
pub(crate) fn robust_fit(pairs: &[[f64; 4]], weights: &[f64]) -> Option<(Wcs, Vec<bool>)> {
    if !weights.is_empty()
        && (weights.len() != pairs.len() || weights.iter().any(|w| !w.is_finite() || *w <= 0.0))
    {
        return None;
    }
    let mut keep = vec![true; pairs.len()];
    for pass in 0..=4 {
        let indices: Vec<_> = (0..pairs.len()).filter(|&i| keep[i]).collect();
        let sub: Vec<_> = indices.iter().map(|&i| pairs[i]).collect();
        let w: Vec<_> = indices
            .iter()
            .map(|&i| weights.get(i).copied().unwrap_or(1.0))
            .collect();
        let fitted = fit(&sub, &w)?;
        let errors = super::residuals(&fitted, pairs);
        let mut finite: Vec<_> = errors.iter().copied().filter(|e| e.is_finite()).collect();
        finite.sort_by(f64::total_cmp);
        let median = *finite.get(finite.len() / 2)?;
        let threshold = (3.0 * 1.4826 * median).max(1.0);
        let next: Vec<_> = errors
            .iter()
            .map(|e| e.is_finite() && *e <= threshold)
            .collect();
        if next == keep || pass == 4 {
            // Return the mask used by this fit, rather than a final unrefitted mask.
            return Some((fitted, keep));
        }
        keep = next;
    }
    unreachable!("bounded fitting loop returns on its last pass")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn truth() -> Wcs {
        Wcs {
            ra: 171.,
            dec: 49.,
            crpix: [1023., 925.],
            cd: [[-0.1464, 0.0674], [-0.0674, -0.1465]],
            sip: None,
            proj: Proj::Stg,
            native_pole: None,
            local_grid: None,
        }
    }

    fn patch(wcs: &Wcs) -> Vec<[f64; 4]> {
        (0..10)
            .flat_map(|i| {
                (0..10).map(move |j| {
                    let (x, y) = (1440. + i as f64 * 22., 350. + j as f64 * 22.);
                    let sky = wcs.pixel_to_sky(x, y);
                    [x, y, sky[0], sky[1]]
                })
            })
            .collect()
    }

    #[test]
    fn a_local_patch_recovers_the_off_centre_stereographic_model() {
        let truth = truth();
        let pairs = patch(&truth);
        // This patch is already too curved for the old linear identification gate.
        let (linear, _) = super::super::robust_fit(&pairs, [1539., 449.], [67., 12.], 1)
            .expect("linear comparison");
        assert!(super::super::rms(&super::super::residuals(&linear, &pairs)) > 4.);
        let (fitted, keep) = robust_fit(&pairs, &[]).expect("fit local patch");
        assert!(keep.iter().all(|k| *k));
        assert!(fitted.sip.is_none());
        // Held-out sky covers the entire image, far outside the fitted 200px patch.
        for i in 0..=20 {
            for j in 0..=13 {
                let (x, y) = (i as f64 * 100., j as f64 * 100.);
                let sky = truth.pixel_to_sky(x, y);
                let px = fitted.sky_to_pixel(sky[0], sky[1]).unwrap();
                assert!((px[0] - x).hypot(px[1] - y) < 1e-3, "{x},{y}: {px:?}");
            }
        }
    }

    #[test]
    fn robust_fit_keeps_noisy_stars_and_rejects_mismatches() {
        let truth = truth();
        let mut pairs = patch(&truth);
        for (i, p) in pairs.iter_mut().enumerate() {
            p[0] += 0.1 * (i as f64 * 1.7).sin();
            p[1] += 0.1 * (i as f64 * 2.3).cos();
        }
        for p in pairs.iter_mut().take(5) {
            p[0] += 15.;
            p[1] -= 12.;
        }
        let weights: Vec<_> = (0..pairs.len()).map(|i| 1. + (i % 3) as f64).collect();
        let (fitted, keep) = robust_fit(&pairs, &weights).expect("robust fit");
        assert!(keep[..5].iter().all(|k| !k));
        assert!(keep[5..].iter().all(|k| *k));
        let residuals = super::super::residuals(&fitted, &pairs[5..]);
        assert!(super::super::rms(&residuals) < 0.2);
    }

    #[test]
    fn rejects_insufficient_degenerate_and_invalid_input() {
        let pairs = patch(&truth());
        assert!(robust_fit(&pairs[..23], &[]).is_none());
        assert!(robust_fit(&vec![pairs[0]; 30], &[]).is_none());
        for weights in [vec![1.], vec![0.; 100], vec![-1.; 100], vec![f64::NAN; 100]] {
            assert!(robust_fit(&pairs, &weights).is_none());
        }
        let mut invalid = pairs;
        invalid[10][2] = f64::INFINITY;
        assert!(robust_fit(&invalid, &[]).is_none());
    }
}
