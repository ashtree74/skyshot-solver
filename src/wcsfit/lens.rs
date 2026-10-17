//! A low-dimensional radial camera model, selected on held-out correspondences.
//! The fitted direction is sensor radius = ideal radius * (1 + k1*r² + k2*r⁴).
//! Export an ordinary SIP (<=9), validating its approximation over the entire frame.
use super::*;
use nalgebra::{Matrix3, Vector3};

fn inverse_radius(rd: f64, k: [f64; 2]) -> Option<f64> {
    if rd == 0.0 {
        return Some(0.0);
    }
    let mut r = rd;
    for _ in 0..24 {
        let r2 = r * r;
        let f = r * (1.0 + k[0] * r2 + k[1] * r2 * r2) - rd;
        let d = 1.0 + 3.0 * k[0] * r2 + 5.0 * k[1] * r2 * r2;
        if !d.is_finite() || d < 0.05 || !r.is_finite() || r < 0.0 {
            return None;
        }
        if f.abs() < 1e-12 {
            return Some(r);
        }
        r -= (f / d).clamp(-0.2, 0.2);
    }
    None
}

struct Evaluation {
    coeff: [Vector3<f64>; 2],
    errors: Vec<f64>,
    cost: f64,
}

fn evaluate(uv: &[[f64; 2]], q: &[[f64; 2]], weights: &[f64], k: [f64; 2]) -> Option<Evaluation> {
    let mut rows = Vec::with_capacity(uv.len());
    let mut a = Matrix3::zeros();
    let mut b = [Vector3::zeros(), Vector3::zeros()];
    for (i, (&p, &target)) in uv.iter().zip(q).enumerate() {
        let rd = p[0].hypot(p[1]);
        let g = if rd > 0.0 {
            inverse_radius(rd, k)? / rd
        } else {
            1.0
        };
        let row = Vector3::new(1.0, p[0] * g, p[1] * g);
        let w = weights.get(i).copied().unwrap_or(1.0);
        if !w.is_finite() || w <= 0.0 {
            return None;
        }
        a += w * row * row.transpose();
        b[0] += w * row * target[0];
        b[1] += w * row * target[1];
        rows.push(row);
    }
    let chol = a.cholesky()?;
    let coeff = [chol.solve(&b[0]), chol.solve(&b[1])];
    let mut errors = Vec::with_capacity(2 * uv.len());
    for (i, (row, target)) in rows.iter().zip(q).enumerate() {
        let w = weights.get(i).copied().unwrap_or(1.0).sqrt();
        errors.push(w * (row.dot(&coeff[0]) - target[0]));
        errors.push(w * (row.dot(&coeff[1]) - target[1]));
    }
    let cost = errors.iter().map(|v| v * v).sum::<f64>();
    cost.is_finite().then_some(Evaluation {
        coeff,
        errors,
        cost,
    })
}

fn optimize(
    uv: &[[f64; 2]],
    q: &[[f64; 2]],
    weights: &[f64],
    k: &mut [f64; 2],
) -> Option<Evaluation> {
    let mut current = evaluate(uv, q, weights, *k)?;
    let mut lambda = 1e-3;
    for _ in 0..16 {
        let mut jac = [Vec::new(), Vec::new()];
        for j in 0..2 {
            let mut perturbed = *k;
            perturbed[j] += 1e-6;
            let shifted = evaluate(uv, q, weights, perturbed)?;
            jac[j] = shifted
                .errors
                .iter()
                .zip(&current.errors)
                .map(|(a, b)| (a - b) / 1e-6)
                .collect();
        }
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let h00 = dot(&jac[0], &jac[0]);
        let h01 = dot(&jac[0], &jac[1]);
        let h11 = dot(&jac[1], &jac[1]);
        let g = [dot(&jac[0], &current.errors), dot(&jac[1], &current.errors)];
        let mut accepted = None;
        for _ in 0..8 {
            let a = h00 * (1.0 + lambda) + 1e-20;
            let d = h11 * (1.0 + lambda) + 1e-20;
            let det = a * d - h01 * h01;
            if !det.is_finite() || det <= 0.0 {
                return None;
            }
            let mut step = [
                (-d * g[0] + h01 * g[1]) / det,
                (h01 * g[0] - a * g[1]) / det,
            ];
            let norm = step[0].hypot(step[1]);
            if norm < 1e-9 {
                return Some(current);
            }
            if norm > 0.2 {
                step.iter_mut().for_each(|v| *v *= 0.2 / norm);
            }
            let trial = [k[0] + step[0], k[1] + step[1]];
            if let Some(next) = evaluate(uv, q, weights, trial)
                && next.cost < current.cost
            {
                accepted = Some((trial, next));
                lambda = (lambda * 0.2).max(1e-9);
                break;
            }
            lambda *= 10.0;
        }
        let Some((next_k, next)) = accepted else {
            break;
        };
        let improvement = current.cost - next.cost;
        *k = next_k;
        current = next;
        if improvement <= 1e-10 * current.cost.max(1e-20) {
            break;
        }
    }
    Some(current)
}

/// Fit only eight camera parameters (tangent, CD and two shared radial coefficients).
/// The normal radius is the frame's half diagonal, not the extent of the matched subset.
fn fit_lens(
    pairs: &[[f64; 4]],
    weights: &[f64],
    crpix: [f64; 2],
    tangent: [f64; 2],
    radius: f64,
) -> Option<Wcs> {
    if pairs.len() < 40 || !radius.is_finite() || radius <= 0.0 {
        return None;
    }
    let uv: Vec<_> = pairs
        .iter()
        .map(|p| [(p[0] - crpix[0]) / radius, (p[1] - crpix[1]) / radius])
        .collect();
    // Do not extrapolate a camera calibrated only near its optical axis.
    if uv.iter().map(|p| p[0].hypot(p[1])).fold(0.0, f64::max) < 0.65 {
        return None;
    }
    let mut tp = tangent;
    let mut k = [0.0, 0.0];
    let mut final_fit = None;
    for _ in 0..10 {
        let projection = PreparedProjection::new(Proj::Tan, tp);
        let q: Vec<_> = pairs
            .iter()
            .map(|p| projection.project(p[2], p[3]))
            .collect::<Option<_>>()?;
        let fit = optimize(&uv, &q, weights, &mut k)?;
        let off = [fit.coeff[0][0], fit.coeff[1][0]];
        if off[0].abs().max(off[1].abs()) < 1e-8 {
            final_fit = Some(fit);
            break;
        }
        tp = unproject_p(Proj::Tan, off[0], off[1], tp);
    }
    let fit = final_fit?;
    let edge = inverse_radius(1.0, k)?;
    let derivative = |r2: f64| 1.0 + 3.0 * k[0] * r2 + 5.0 * k[1] * r2 * r2;
    let critical = if k[1] > 0.0 {
        (-3.0 * k[0] / (10.0 * k[1])).clamp(0.0, edge * edge)
    } else {
        0.0
    };
    if derivative(0.0)
        .min(derivative(edge * edge))
        .min(derivative(critical))
        < 0.05
    {
        return None;
    }
    // Least-squares approximation of r_ideal-r_sensor, with the centre slope fixed to one.
    // A degree-9 SIP remains compatible with the existing writers/client validation.
    let mut a = DMatrix::<f64>::zeros(4, 4);
    let mut b = DVector::<f64>::zeros(4);
    for i in 1..=128 {
        let r = i as f64 / 128.0;
        let row = [r.powi(3), r.powi(5), r.powi(7), r.powi(9)];
        let y = inverse_radius(r, k)? - r;
        for j in 0..4 {
            b[j] += row[j] * y;
            for l in 0..4 {
                a[(j, l)] += row[j] * row[l];
            }
        }
    }
    let coeff = a.cholesky()?.solve(&b);
    // Separate, interleaved samples check the approximation rather than its training nodes.
    for i in 0..=256 {
        let r = (i as f64 + 0.5) / 257.0;
        let approx = r
            + (0..4)
                .map(|j| coeff[j] * r.powi(2 * j as i32 + 3))
                .sum::<f64>();
        if (approx - inverse_radius(r, k)?).abs() * radius > 0.25 {
            return None;
        }
    }
    let terms = Sip::terms(9);
    let mut sip = Sip {
        order: 9,
        a: vec![0.0; terms.len()],
        b: vec![0.0; terms.len()],
    };
    for j in 1..=4u32 {
        let scale = coeff[(j - 1) as usize] / radius.powi(2 * j as i32);
        let mut binom = 1.0;
        for i in 0..=j {
            let a = terms.iter().position(|&t| t == (2 * i + 1, 2 * (j - i)))?;
            let b = terms.iter().position(|&t| t == (2 * i, 2 * (j - i) + 1))?;
            sip.a[a] += scale * binom;
            sip.b[b] += scale * binom;
            binom *= (j - i) as f64 / (i + 1) as f64;
        }
    }
    Some(Wcs {
        ra: tp[0],
        dec: tp[1],
        crpix,
        cd: [
            [fit.coeff[0][1] / radius, fit.coeff[0][2] / radius],
            [fit.coeff[1][1] / radius, fit.coeff[1][2] / radius],
        ],
        sip: Some(sip),
        proj: Proj::Tan,
        native_pole: None,
        local_grid: None,
    })
}

/// Prefer the smaller physical model only when independent spatial holdout supports it.
pub(crate) fn select_lens(
    pairs: &[[f64; 4]],
    weights: &[f64],
    reference: &Wcs,
    model: (u32, u32),
    width: f64,
    height: f64,
) -> Option<Wcs> {
    if !reference.proj.is_tan() || pairs.len() < 80 || model.1 == 0 {
        return None;
    }
    let radius = 0.5 * width.hypot(height);
    let mut counts = [0usize; 16];
    let (mut train, mut train_w, mut test, mut test_w) = (vec![], vec![], vec![], vec![]);
    for (i, &p) in pairs.iter().enumerate() {
        let x = (4.0 * p[0] / width).floor().clamp(0.0, 3.0) as usize;
        let y = (4.0 * p[1] / height).floor().clamp(0.0, 3.0) as usize;
        let cell = 4 * y + x;
        let wt = weights.get(i).copied().unwrap_or(1.0);
        if counts[cell].is_multiple_of(4) {
            test.push(p);
            test_w.push(wt);
        } else {
            train.push(p);
            train_w.push(wt);
        }
        counts[cell] += 1;
    }
    if test.len() < 16 {
        return None;
    }
    let tangent = [reference.ra, reference.dec];
    let lens = fit_lens(&train, &train_w, reference.crpix, tangent, radius)?;
    let (base, _) = robust_fit_wmp(
        &train,
        &train_w,
        reference.crpix,
        tangent,
        model.0,
        model.1,
        reference.proj,
    )?;
    let le = residuals(&lens, &test);
    let be = residuals(&base, &test);
    let lr = robust_test_rms(&le, &test_w);
    let br = robust_test_rms(&be, &test_w);
    if !lr.is_finite() || !br.is_finite() || lr > 1.05 * br {
        return None;
    }
    // Complexity-penalized holdout score for two residual coordinates per star.
    let n = 2.0 * test.len() as f64;
    let score = |e: f64, k: f64| n * (e * e).max(1e-20).ln() + k * n.ln();
    if score(lr, 8.0) >= score(br, 2.0 * model_columns(model.0, model.1) as f64) {
        return None;
    }
    // A globally low error must not hide a locally worse sector of the frame.
    for cell in 0..16 {
        let mut a = 0.0;
        let mut b = 0.0;
        let mut count = 0;
        for ((p, &l), &r) in test.iter().zip(&le).zip(&be) {
            let x = (4.0 * p[0] / width).floor().clamp(0.0, 3.0) as usize;
            let y = (4.0 * p[1] / height).floor().clamp(0.0, 3.0) as usize;
            if !l.is_finite() || !r.is_finite() {
                return None;
            }
            if cell == 4 * y + x {
                a += l * l;
                b += r * r;
                count += 1;
            }
        }
        if count > 0 && (a / count as f64).sqrt() > 1.25 * (b / count as f64).sqrt() + 0.1 {
            return None;
        }
    }
    let fit = fit_lens(pairs, weights, reference.crpix, tangent, radius)?;
    if crate::solve::debug_on() {
        eprintln!("  radial lens: held-out {lr:.3} vs polynomial {br:.3}, SIP9");
    }
    Some(fit)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent bisection inversion of a forward radial camera; no SIP used to generate truth.
    fn sky_at(x: f64, y: f64, k: [f64; 2]) -> [f64; 2] {
        let u = x - 3000.0;
        let v = y - 2000.0;
        let radius = 3000.0f64.hypot(2000.0);
        let rd = u.hypot(v) / radius;
        let (mut lo, mut hi) = (0.0, 1.5);
        for _ in 0..64 {
            let r: f64 = (lo + hi) * 0.5;
            if r * (1.0 + k[0] * r * r + k[1] * r.powi(4)) < rd {
                lo = r;
            } else {
                hi = r;
            }
        }
        let g = if rd == 0.0 { 1.0 } else { (lo + hi) * 0.5 / rd };
        unproject_p(
            Proj::Tan,
            (-u + 0.2 * v) * g * 0.006,
            (0.2 * u + v) * g * 0.006,
            [124.0, -34.0],
        )
    }

    fn pairs(k: [f64; 2]) -> Vec<[f64; 4]> {
        (0..17)
            .flat_map(|j| {
                (0..23).map(move |i| {
                    let x = 200.0 + i as f64 * 250.0;
                    let y = 180.0 + j as f64 * 225.0;
                    let sky = sky_at(x, y, k);
                    [x, y, sky[0], sky[1]]
                })
            })
            .collect()
    }

    #[test]
    fn radial_camera_predicts_independent_frame_grid() {
        for k in [[-0.1, 0.02], [-0.095, 0.003], [0.08, 0.005], [0.0, 0.0]] {
            let p = pairs(k);
            let fit = fit_lens(
                &p,
                &[],
                [3000.0, 2000.0],
                [124.02, -34.01],
                3000.0f64.hypot(2000.0),
            )
            .unwrap_or_else(|| panic!("physical camera fit: {k:?}"));
            assert_eq!(fit.sip.as_ref().unwrap().order, 9);
            let mut worst: f64 = 0.0;
            for j in 0..=20 {
                for i in 0..=30 {
                    let (x, y) = (200.0 * i as f64, 200.0 * j as f64);
                    let s = sky_at(x, y, k);
                    let q = fit.sky_to_pixel(s[0], s[1]).expect("finite inversion");
                    worst = worst.max((q[0] - x).hypot(q[1] - y));
                }
            }
            assert!(worst < 0.3, "{k:?}: {worst} px");
        }
    }

    #[test]
    fn radial_camera_rejects_degenerate_and_invalid_data() {
        let p = pairs([-0.1, 0.02]);
        let fit =
            |p: &[[f64; 4]], w: &[f64]| fit_lens(p, w, [3000.0, 2000.0], [124.0, -34.0], 3605.55);
        assert!(fit(&p[..20], &[]).is_none());
        assert!(fit(&[[3000.0, 2000.0, 124.0, -34.0]; 80], &[]).is_none());
        assert!(fit(&p, &[f64::NAN]).is_none());
        assert!(fit(&p, &[-1.0]).is_none());
        assert!(inverse_radius(1.0, [-1.0, 0.0]).is_none());
    }

    #[test]
    fn holdout_rejects_nonradial_camera() {
        let mut p = pairs([0.0, 0.0]);
        for q in &mut p {
            let u = (q[0] - 3000.0) / 3000.0;
            let v = (q[1] - 2000.0) / 2000.0;
            let s = sky_at(q[0] + 12.0 * u * v, q[1] + 10.0 * u * u, [0.0, 0.0]);
            q[2] = s[0];
            q[3] = s[1];
        }
        let reference = fit_tan_sip_model(&p, &[], [3000.0, 2000.0], [124.0, -34.0], 2, 7).unwrap();
        assert!(select_lens(&p, &[], &reference, (2, 7), 6000.0, 4000.0).is_none());
    }
}
