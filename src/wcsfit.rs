//! Robust TAN + SIP fitting from pixel <-> sky correspondences.
//!
//! The model is fitted as a polynomial (u, v) -> (xi, eta) around a fixed CRPIX, with the tangent
//! point iterated until the constant terms vanish; the linear part becomes CD and the nonlinear
//! part becomes SIP A/B (CD^-1 applied term-wise). SIP order is chosen on held-out stars.

use crate::{
    geometry::{PreparedProjection, Projection, unproject_p},
    types::{Proj, Sip, Wcs},
};
use nalgebra::{DMatrix, DVector};

mod lens;
pub(crate) mod panorama;
pub(crate) mod registration;
mod stereographic;
pub(crate) use lens::select_lens;
pub(crate) use panorama::candidates as panorama_candidates;
pub(crate) use stereographic::robust_fit as fit_stereographic;

/// Result of [`select_model_w`]: fit, inlier mask, held-out error, (SIP order, radial degree).
pub type ModelChoice = (Wcs, Vec<bool>, f64, (u32, u32));

/// Weighted least squares fit of TAN + SIP(order) with CRPIX fixed. `pairs` = (x, y, ra, dec).
pub fn fit_tan_sip(
    pairs: &[[f64; 4]],
    weights: &[f64],
    crpix: [f64; 2],
    tangent: [f64; 2],
    order: u32,
) -> Option<Wcs> {
    fit_tan_sip_model(pairs, weights, crpix, tangent, order, 0)
}

/// Odd degrees `d` (order < d <= radial) of the radial extension of an order-`order` model.
fn radial_degrees(order: u32, radial: u32) -> Vec<u32> {
    (order + 1..=radial).filter(|d| d % 2 == 1).collect()
}

/// Number of unknowns per axis of the model (order, radial).
pub fn model_columns(order: u32, radial: u32) -> usize {
    3 + Sip::terms(order).len() + 2 * radial_degrees(order, radial).len()
}

/// [`fit_tan_sip`] plus a lens-like radial extension: for every odd degree d = 2j + 1 with
/// order < d <= `radial`, the terms M_d (u, v) rho^(2j) with a free 2x2 matrix M_d (rho = distance
/// from CRPIX). Unlike free high-order monomials these extrapolate like radial lens distortion,
/// so a fit on the inner part of a wide frame still predicts the corners. The result is an
/// ordinary SIP of order max(order, highest radial degree).
pub fn fit_tan_sip_model(
    pairs: &[[f64; 4]],
    weights: &[f64],
    crpix: [f64; 2],
    tangent: [f64; 2],
    order: u32,
    radial: u32,
) -> Option<Wcs> {
    fit_model(pairs, weights, crpix, tangent, order, radial, Proj::Tan)
}

/// [`fit_tan_sip_model`] with the projection `proj` instead of TAN (ARC for fisheye fields).
pub fn fit_model(
    pairs: &[[f64; 4]],
    weights: &[f64],
    crpix: [f64; 2],
    tangent: [f64; 2],
    order: u32,
    radial: u32,
    proj: impl Into<Projection>,
) -> Option<Wcs> {
    let proj = proj.into().fixed(tangent);
    let PolynomialFit {
        scale,
        tangent: tp,
        x: cx,
        y: cy,
    } = fit_polynomial(pairs, weights, crpix, tangent, order, radial, proj)?;
    let terms = Sip::terms(order);
    let rdeg = radial_degrees(order, radial);
    // linear part (per pixel)
    let cd = [
        [cx[1] / scale, cx[2] / scale],
        [cy[1] / scale, cy[2] / scale],
    ];
    let det = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
    if det.abs() < 1e-30 {
        return None;
    }
    let inv = [
        [cd[1][1] / det, -cd[0][1] / det],
        [-cd[1][0] / det, cd[0][0] / det],
    ];
    let out_order = rdeg.last().copied().unwrap_or(order).max(order);
    let sip = if out_order >= 2 {
        let all = Sip::terms(out_order);
        let pos = |p: u32, q: u32| all.iter().position(|&t| t == (p, q)).expect("SIP term");
        // (xi, eta) coefficients per monomial in normalised units
        let mut ex = vec![0.0; all.len()];
        let mut ey = vec![0.0; all.len()];
        for (k, &(pp, qq)) in terms.iter().enumerate() {
            let i = pos(pp, qq);
            ex[i] += cx[3 + k];
            ey[i] += cy[3 + k];
        }
        // radial terms: u rho^(2j) = sum_i C(j, i) u^(2i+1) v^(2j-2i), likewise for v
        for (k, &d) in rdeg.iter().enumerate() {
            let j = (d - 1) / 2;
            let c0 = 3 + terms.len() + 2 * k;
            let mut binom = 1.0;
            for i in 0..=j {
                let (pe, qe) = (2 * i, 2 * (j - i));
                let iu = pos(pe + 1, qe);
                let iv = pos(pe, qe + 1);
                ex[iu] += binom * cx[c0];
                ey[iu] += binom * cy[c0];
                ex[iv] += binom * cx[c0 + 1];
                ey[iv] += binom * cy[c0 + 1];
                binom = binom * (j - i) as f64 / (i + 1) as f64;
            }
        }
        let mut sa = Vec::with_capacity(all.len());
        let mut sb = Vec::with_capacity(all.len());
        for (i, &(pp, qq)) in all.iter().enumerate() {
            let f = scale.powi((pp + qq) as i32);
            let (x, y) = (ex[i] / f, ey[i] / f);
            sa.push(inv[0][0] * x + inv[0][1] * y);
            sb.push(inv[1][0] * x + inv[1][1] * y);
        }
        Some(Sip {
            order: out_order,
            a: sa,
            b: sb,
        })
    } else {
        None
    };
    Some(Wcs {
        ra: tp[0],
        dec: tp[1],
        crpix,
        cd,
        sip,
        proj: proj.proj,
        native_pole: proj.native_pole,
        local_grid: None,
    })
}

/// The pixel basis and weights do not change while the tangent point is iterated. Keep
/// their normal matrix factorization and weighted rows; only the projected targets change.
struct PolynomialFit {
    scale: f64,
    tangent: [f64; 2],
    x: Vec<f64>,
    y: Vec<f64>,
}

fn fit_polynomial(
    pairs: &[[f64; 4]],
    weights: &[f64],
    crpix: [f64; 2],
    tangent: [f64; 2],
    order: u32,
    radial: u32,
    proj: impl Into<Projection>,
) -> Option<PolynomialFit> {
    let proj = proj.into().fixed(tangent);
    let terms = Sip::terms(order);
    let rdeg = radial_degrees(order, radial);
    let ncol = 3 + terms.len() + 2 * rdeg.len();
    if pairs.len() < ncol + 3 {
        return None;
    }
    let scale = pairs
        .iter()
        .map(|p| (p[0] - crpix[0]).abs().max((p[1] - crpix[1]).abs()))
        .fold(1.0, f64::max);
    let mut row = vec![0.0; ncol];
    let mut weighted_rows = Vec::with_capacity(pairs.len() * ncol);
    let mut ata = DMatrix::<f64>::zeros(ncol, ncol);
    for (i, p) in pairs.iter().enumerate() {
        let w = weights.get(i).copied().unwrap_or(1.0);
        let u = (p[0] - crpix[0]) / scale;
        let v = (p[1] - crpix[1]) / scale;
        row[0] = 1.0;
        row[1] = u;
        row[2] = v;
        let (mut pu, mut pv) = ([1.0f64; 16], [1.0f64; 16]);
        for i in 1..=(order as usize).min(15) {
            pu[i] = pu[i - 1] * u;
            pv[i] = pv[i - 1] * v;
        }
        for (k, &(pp, qq)) in terms.iter().enumerate() {
            row[3 + k] = pu[pp as usize] * pv[qq as usize];
        }
        let rho2 = u * u + v * v;
        for (k, &d) in rdeg.iter().enumerate() {
            let f = rho2.powi(((d - 1) / 2) as i32);
            row[3 + terms.len() + 2 * k] = u * f;
            row[3 + terms.len() + 2 * k + 1] = v * f;
        }
        // Preserve the original accumulation order for both the Gram matrix and RHS.
        for r in 0..ncol {
            let wr = w * row[r];
            weighted_rows.push(wr);
            for c in r..ncol {
                ata[(r, c)] += wr * row[c];
            }
        }
    }
    for r in 0..ncol {
        for c in 0..r {
            ata[(r, c)] = ata[(c, r)];
        }
    }
    let chol = ata.cholesky()?;
    let mut tp = tangent;
    let mut sol = None;
    for _ in 0..3 {
        let projection = proj.prepare(tp);
        let mut atx = DVector::<f64>::zeros(ncol);
        let mut aty = DVector::<f64>::zeros(ncol);
        for (p, wr) in pairs.iter().zip(weighted_rows.chunks_exact(ncol)) {
            let q = projection.project(p[2], p[3])?;
            for r in 0..ncol {
                atx[r] += wr[r] * q[0];
                aty[r] += wr[r] * q[1];
            }
        }
        let cx = chol.solve(&atx);
        let cy = chol.solve(&aty);
        let off = [cx[0], cy[0]];
        tp = proj.unproject(off[0], off[1], tp);
        let done = off[0].abs() < 1e-11 && off[1].abs() < 1e-11;
        sol = Some((cx.as_slice().to_vec(), cy.as_slice().to_vec()));
        if done {
            break;
        }
    }
    let (x, y) = sol?;
    Some(PolynomialFit {
        scale,
        tangent: tp,
        x,
        y,
    })
}

/// Pixel residuals of pairs under `wcs`, computed in the forward direction (pixel -> tangent
/// plane, no SIP inversion): |CD (u + A, v + B) - proj(sky)| / scale.
pub fn residuals(wcs: &Wcs, pairs: &[[f64; 4]]) -> Vec<f64> {
    let scale = wcs.determinant().abs().sqrt();
    let projection = wcs.projection().prepare([wcs.ra, wcs.dec]);
    pairs
        .iter()
        .map(|p| {
            if wcs.local_grid.is_some() {
                return wcs
                    .sky_to_pixel(p[2], p[3])
                    .map_or(f64::INFINITY, |q| (q[0] - p[0]).hypot(q[1] - p[1]));
            }
            let Some(q) = projection.project(p[2], p[3]) else {
                return f64::INFINITY;
            };
            let (mut u, mut v) = (p[0] - wcs.crpix[0], p[1] - wcs.crpix[1]);
            if let Some(s) = &wcs.sip {
                let (fa, fb) = s.eval(u, v);
                u += fa;
                v += fb;
            }
            let xi = wcs.cd[0][0] * u + wcs.cd[0][1] * v;
            let eta = wcs.cd[1][0] * u + wcs.cd[1][1] * v;
            (xi - q[0]).hypot(eta - q[1]) / scale
        })
        .collect()
}

pub fn rms(r: &[f64]) -> f64 {
    let v: Vec<f64> = r.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::INFINITY;
    }
    (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let n = v.len() / 2;
    *v.select_nth_unstable_by(n, |a, b| a.partial_cmp(b).unwrap())
        .1
}

/// Robust fit: iterative outlier rejection (> max(3 * 1.4826 * MAD-based sigma, floor)).
pub fn robust_fit(
    pairs: &[[f64; 4]],
    crpix: [f64; 2],
    tangent: [f64; 2],
    order: u32,
) -> Option<(Wcs, Vec<bool>)> {
    robust_fit_w(pairs, &[], crpix, tangent, order)
}

/// Weighted robust fit (`weights` empty = unit weights; otherwise one per pair, e.g. 1/sigma^2).
pub fn robust_fit_w(
    pairs: &[[f64; 4]],
    weights: &[f64],
    crpix: [f64; 2],
    tangent: [f64; 2],
    order: u32,
) -> Option<(Wcs, Vec<bool>)> {
    robust_fit_wm(pairs, weights, crpix, tangent, order, 0)
}

/// [`robust_fit_w`] for the model (order, radial) of [`fit_tan_sip_model`].
pub fn robust_fit_wm(
    pairs: &[[f64; 4]],
    weights: &[f64],
    crpix: [f64; 2],
    tangent: [f64; 2],
    order: u32,
    radial: u32,
) -> Option<(Wcs, Vec<bool>)> {
    robust_fit_wmp(pairs, weights, crpix, tangent, order, radial, Proj::Tan)
}

/// [`robust_fit_wm`] with the projection `proj`.
pub fn robust_fit_wmp(
    pairs: &[[f64; 4]],
    weights: &[f64],
    crpix: [f64; 2],
    tangent: [f64; 2],
    order: u32,
    radial: u32,
    proj: impl Into<Projection>,
) -> Option<(Wcs, Vec<bool>)> {
    let proj = proj.into().fixed(tangent);
    let wt = |i: usize| weights.get(i).copied().unwrap_or(1.0);
    let mut keep = vec![true; pairs.len()];
    let mut wcs = None;
    for _ in 0..4 {
        let idx: Vec<usize> = (0..pairs.len()).filter(|&i| keep[i]).collect();
        let sub: Vec<[f64; 4]> = idx.iter().map(|&i| pairs[i]).collect();
        let w: Vec<f64> = idx.iter().map(|&i| wt(i)).collect();
        let fit = fit_model(&sub, &w, crpix, tangent, order, radial, proj)?;
        let r = residuals(&fit, pairs);
        let sigma =
            (1.4826 * median(r.iter().copied().filter(|x| x.is_finite()).collect())).max(0.3);
        let new_keep: Vec<bool> = r.iter().map(|&x| x <= 3.0 * sigma).collect();
        let changed = new_keep != keep;
        keep = new_keep;
        wcs = Some(fit);
        if !changed {
            break;
        }
    }
    Some((wcs?, keep))
}

/// Choose SIP order (1..=max_order) by held-out residuals; returns the refit on all inliers.
pub fn select_order(
    pairs: &[[f64; 4]],
    crpix: [f64; 2],
    tangent: [f64; 2],
    max_order: u32,
) -> Option<(Wcs, Vec<bool>, f64)> {
    select_order_w(pairs, &[], crpix, tangent, max_order)
}

/// Held-out error for model choice: weighted RMS with each residual capped at
/// max(3 px, 5 x median) and each weight at 4 x the median weight, so that one wrong pair or
/// one over-weighted bright star cannot decide between models, while a model that fails in a
/// whole region (e.g. the corners) is still penalised there.
fn robust_test_rms(r: &[f64], w: &[f64]) -> f64 {
    let fin: Vec<f64> = r.iter().copied().filter(|x| x.is_finite()).collect();
    if fin.is_empty() {
        return f64::INFINITY;
    }
    let cap = (5.0 * median(fin)).max(3.0);
    let wmed = if w.is_empty() {
        1.0
    } else {
        median(w.to_vec())
    };
    let wcap = 4.0 * wmed;
    let (mut s, mut n) = (0.0, 0.0);
    for (i, &x) in r.iter().enumerate() {
        let x = if x.is_finite() { x.min(cap) } else { cap };
        let wi = w.get(i).copied().unwrap_or(1.0).min(wcap);
        s += wi * x * x;
        n += wi;
    }
    if n > 0.0 {
        (s / n).sqrt()
    } else {
        f64::INFINITY
    }
}

/// Weighted variant of [`select_order`].
pub fn select_order_w(
    pairs: &[[f64; 4]],
    weights: &[f64],
    crpix: [f64; 2],
    tangent: [f64; 2],
    max_order: u32,
) -> Option<(Wcs, Vec<bool>, f64)> {
    select_model_w(pairs, weights, crpix, tangent, max_order, 0).map(|(w, k, e, _)| (w, k, e))
}

/// Choose the model on held-out stars among SIP orders 1..=max_order and, when `radial` > 0,
/// the same orders extended by radial terms up to degree `radial` (see [`fit_tan_sip_model`]).
/// Candidates are tried from the fewest unknowns up; a more complex one must cut the held-out
/// error by 10 %; radial candidates also need >= 6 inliers per unknown. Returns (fit, inliers, held-out error,
/// (order, radial)).
pub fn select_model_w(
    pairs: &[[f64; 4]],
    weights: &[f64],
    crpix: [f64; 2],
    tangent: [f64; 2],
    max_order: u32,
    radial: u32,
) -> Option<ModelChoice> {
    select_model_wp(pairs, weights, crpix, tangent, max_order, radial, Proj::Tan)
}

/// [`select_model_w`] with the projection `proj`.
pub fn select_model_wp(
    pairs: &[[f64; 4]],
    weights: &[f64],
    crpix: [f64; 2],
    tangent: [f64; 2],
    max_order: u32,
    radial: u32,
    proj: impl Into<Projection>,
) -> Option<ModelChoice> {
    let proj = proj.into().fixed(tangent);
    let wt = |i: usize| weights.get(i).copied().unwrap_or(1.0);
    // Inliers of the held-out test. A pure TAN fit would reject the outer stars of a distorted
    // wide field -- exactly the ones that show whether distortion terms help -- so with radial
    // candidates the inliers come from the radial model (TAN as a fallback).
    let (baseline_model, baseline_fit) = if radial > 0 {
        let model = (max_order.min(3), radial);
        match robust_fit_wmp(pairs, weights, crpix, tangent, model.0, model.1, proj) {
            Some(fit) => (model, fit),
            None => (
                (1, 0),
                robust_fit_wmp(pairs, weights, crpix, tangent, 1, 0, proj)?,
            ),
        }
    } else {
        (
            (1, 0),
            robust_fit_wmp(pairs, weights, crpix, tangent, 1, 0, proj)?,
        )
    };
    let inl_idx: Vec<usize> = (0..pairs.len()).filter(|&i| baseline_fit.1[i]).collect();
    // Every model is judged on this same split. Build its arrays once, preserving order.
    let tr: Vec<usize> = inl_idx
        .iter()
        .enumerate()
        .filter(|(k, _)| k % 4 != 0)
        .map(|(_, &i)| i)
        .collect();
    let te: Vec<usize> = inl_idx
        .iter()
        .enumerate()
        .filter(|(k, _)| k % 4 == 0)
        .map(|(_, &i)| i)
        .collect();
    let train: Vec<[f64; 4]> = tr.iter().map(|&i| pairs[i]).collect();
    let trw: Vec<f64> = tr.iter().map(|&i| wt(i)).collect();
    let test: Vec<[f64; 4]> = te.iter().map(|&i| pairs[i]).collect();
    let tew: Vec<f64> = te.iter().map(|&i| wt(i)).collect();
    let test_err = |order: u32, rad: u32| -> Option<f64> {
        if test.len() < 8 {
            return None;
        }
        let (fit, _) = robust_fit_wmp(&train, &trw, crpix, tangent, order, rad, proj)?;
        Some(robust_test_rms(&residuals(&fit, &test), &tew))
    };
    let mut cand: Vec<(u32, u32)> = (2..=max_order).map(|o| (o, 0)).collect();
    if radial > 0 {
        for o in 1..=max_order {
            if !radial_degrees(o, radial).is_empty() {
                cand.push((o, radial));
            }
        }
    }
    cand.sort_by_key(|&(o, r)| (model_columns(o, r), o));
    let mut best = ((1u32, 0u32), test_err(1, 0).unwrap_or(f64::INFINITY));
    for (o, r) in cand {
        let base = match o {
            2 => 30,
            3 => 60,
            4 => 100,
            5 => 160,
            _ => 240,
        };
        let need = if r > 0 {
            base.max(6 * model_columns(o, r))
        } else {
            base
        };
        if inl_idx.len() < need {
            continue;
        }
        let e = test_err(o, r);
        if crate::solve::debug_on() {
            eprintln!(
                "    model ({o}, {r}): held-out {:?} (best {:?} {:.3}), inliers {}",
                e,
                best.0,
                best.1,
                inl_idx.len()
            );
        }
        if let Some(e) = e
            && e < 0.9 * best.1
        {
            best = ((o, r), e);
        }
    }
    // The inlier-selection fit is already the final full-data fit when its model wins.
    let (fit, keep) = if best.0 == baseline_model {
        baseline_fit
    } else {
        robust_fit_wmp(pairs, weights, crpix, tangent, best.0.0, best.0.1, proj)?
    };
    Some((fit, keep, best.1, best.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::project_p;

    // Frozen pre-cache kernel: compares every tangent iteration and coefficient, including
    // weights and radial columns, without duplicating WCS coefficient conversion.
    fn uncached_polynomial(
        pairs: &[[f64; 4]],
        weights: &[f64],
        crpix: [f64; 2],
        tangent: [f64; 2],
        order: u32,
        radial: u32,
        proj: Proj,
    ) -> Option<PolynomialFit> {
        let terms = Sip::terms(order);
        let rdeg = radial_degrees(order, radial);
        let ncol = 3 + terms.len() + 2 * rdeg.len();
        if pairs.len() < ncol + 3 {
            return None;
        }
        let scale = pairs
            .iter()
            .map(|p| (p[0] - crpix[0]).abs().max((p[1] - crpix[1]).abs()))
            .fold(1.0, f64::max);
        let mut tp = tangent;
        let mut sol: Option<(Vec<f64>, Vec<f64>)> = None;
        let mut row = vec![0.0; ncol];
        for _ in 0..3 {
            // normal equations (ncol x ncol), columns normalised by `scale`
            let mut ata = DMatrix::<f64>::zeros(ncol, ncol);
            let mut atx = DVector::<f64>::zeros(ncol);
            let mut aty = DVector::<f64>::zeros(ncol);
            for (i, p) in pairs.iter().enumerate() {
                let w = weights.get(i).copied().unwrap_or(1.0);
                let q = project_p(proj, p[2], p[3], tp)?;
                let u = (p[0] - crpix[0]) / scale;
                let v = (p[1] - crpix[1]) / scale;
                row[0] = 1.0;
                row[1] = u;
                row[2] = v;
                let (mut pu, mut pv) = ([1.0f64; 16], [1.0f64; 16]);
                for i in 1..=(order as usize).min(15) {
                    pu[i] = pu[i - 1] * u;
                    pv[i] = pv[i - 1] * v;
                }
                for (k, &(pp, qq)) in terms.iter().enumerate() {
                    row[3 + k] = pu[pp as usize] * pv[qq as usize];
                }
                let rho2 = u * u + v * v;
                for (k, &d) in rdeg.iter().enumerate() {
                    let f = rho2.powi(((d - 1) / 2) as i32);
                    row[3 + terms.len() + 2 * k] = u * f;
                    row[3 + terms.len() + 2 * k + 1] = v * f;
                }
                for r in 0..ncol {
                    let wr = w * row[r];
                    atx[r] += wr * q[0];
                    aty[r] += wr * q[1];
                    for c in r..ncol {
                        ata[(r, c)] += wr * row[c];
                    }
                }
            }
            for r in 0..ncol {
                for c in 0..r {
                    ata[(r, c)] = ata[(c, r)];
                }
            }
            let chol = ata.cholesky()?;
            let cx = chol.solve(&atx);
            let cy = chol.solve(&aty);
            let off = [cx[0], cy[0]];
            tp = unproject_p(proj, off[0], off[1], tp);
            let done = off[0].abs() < 1e-11 && off[1].abs() < 1e-11;
            sol = Some((cx.as_slice().to_vec(), cy.as_slice().to_vec()));
            if done {
                break;
            }
        }
        let (x, y) = sol?;
        Some(PolynomialFit {
            scale,
            tangent: tp,
            x,
            y,
        })
    }

    fn fixture(proj: Proj, order: u32) -> (Wcs, Vec<[f64; 4]>, Vec<f64>) {
        let terms = Sip::terms(order);
        let truth = Wcs {
            ra: 359.8,
            dec: 74.0,
            crpix: [1200.0, 900.0],
            cd: [[-0.003, 0.0002], [0.0001, 0.003]],
            sip: (order > 1).then(|| Sip {
                order,
                a: terms
                    .iter()
                    .enumerate()
                    .map(|(i, &(p, q))| {
                        2.0 * (i as f64 % 3.0 - 1.0) / 1200.0f64.powi((p + q) as i32)
                    })
                    .collect(),
                b: terms
                    .iter()
                    .enumerate()
                    .map(|(i, &(p, q))| (i as f64 % 5.0 - 2.0) / 1200.0f64.powi((p + q) as i32))
                    .collect(),
            }),
            proj,
            native_pole: None,
            local_grid: None,
        };
        let mut state = 137u64;
        let mut noise = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            0.06 * ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
        };
        let mut pairs = vec![];
        let mut weights = vec![];
        for i in 0..20 {
            for j in 0..16 {
                let (x, y) = (73.0 + 125.0 * i as f64, 61.0 + 115.0 * j as f64);
                let sky = truth.pixel_to_sky(x, y);
                pairs.push([x + noise(), y + noise(), sky[0], sky[1]]);
                weights.push(0.3 + ((i * 7 + j * 11) % 19) as f64 / 8.0);
            }
        }
        (truth, pairs, weights)
    }

    fn assert_same_wcs(a: &Wcs, b: &Wcs) {
        assert_eq!(
            serde_json::to_string(a).unwrap(),
            serde_json::to_string(b).unwrap()
        );
    }

    #[test]
    fn cached_normal_system_is_bitwise_equivalent_for_weighted_models() {
        for (proj, order, radial) in [
            (Proj::Tan, 1, 0),
            (Proj::Tan, 5, 0),
            (Proj::Tan, 3, 7),
            (Proj::Arc, 5, 0),
            (Proj::Arc, 3, 7),
        ] {
            let (truth, pairs, weights) = fixture(proj, order.min(5));
            // Exercise unit weights, full weights, and the historical default for missing weights.
            for w in [&[][..], &weights[..], &weights[..weights.len() / 2]] {
                let args = (truth.crpix, [359.6, 74.15]);
                let got = fit_polynomial(&pairs, w, args.0, args.1, order, radial, proj).unwrap();
                let old =
                    uncached_polynomial(&pairs, w, args.0, args.1, order, radial, proj).unwrap();
                assert_eq!(got.scale.to_bits(), old.scale.to_bits());
                assert_eq!(got.tangent.map(f64::to_bits), old.tangent.map(f64::to_bits));
                assert_eq!(
                    got.x.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    old.x.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(
                    got.y.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    old.y.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn prepared_residuals_are_bitwise_equivalent() {
        for proj in [Proj::Tan, Proj::Arc] {
            let (truth, pairs, _) = fixture(proj, 5);
            let scale = truth.determinant().abs().sqrt();
            let expected: Vec<f64> = pairs
                .iter()
                .map(|p| {
                    let q = project_p(proj, p[2], p[3], [truth.ra, truth.dec]).unwrap();
                    let (mut u, mut v) = (p[0] - truth.crpix[0], p[1] - truth.crpix[1]);
                    let (fa, fb) = truth.sip.as_ref().unwrap().eval(u, v);
                    u += fa;
                    v += fb;
                    let xi = truth.cd[0][0] * u + truth.cd[0][1] * v;
                    let eta = truth.cd[1][0] * u + truth.cd[1][1] * v;
                    (xi - q[0]).hypot(eta - q[1]) / scale
                })
                .collect();
            assert_eq!(
                residuals(&truth, &pairs)
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn cached_fits_preserve_projection_and_rank_failures() {
        for proj in [Proj::Tan, Proj::Arc] {
            let (truth, mut pairs, weights) = fixture(proj, 3);
            pairs[17][2] = 180.0;
            pairs[17][3] = 0.0;
            assert!(
                fit_polynomial(&pairs, &weights, truth.crpix, [0.0, 0.0], 3, 0, proj).is_none()
            );
            assert!(
                uncached_polynomial(&pairs, &weights, truth.crpix, [0.0, 0.0], 3, 0, proj)
                    .is_none()
            );
        }
        let (truth, mut pairs, weights) = fixture(Proj::Tan, 3);
        assert!(
            fit_model(
                &pairs[..5],
                &[],
                truth.crpix,
                [truth.ra, truth.dec],
                3,
                0,
                Proj::Tan
            )
            .is_none()
        );
        for p in &mut pairs {
            p[1] = truth.crpix[1];
        }
        assert!(
            fit_model(
                &pairs,
                &weights,
                truth.crpix,
                [truth.ra, truth.dec],
                3,
                0,
                Proj::Tan
            )
            .is_none()
        );
    }

    #[test]
    fn weighted_high_order_robust_fit_rejects_outliers() {
        for proj in [Proj::Tan, Proj::Arc] {
            let (truth, mut pairs, weights) = fixture(proj, 5);
            for p in pairs.iter_mut().step_by(13) {
                p[0] += 45.0;
                p[1] -= 35.0;
            }
            let (fit, keep) = robust_fit_wmp(
                &pairs,
                &weights,
                truth.crpix,
                [truth.ra, truth.dec],
                5,
                0,
                proj,
            )
            .unwrap();
            let r = residuals(&fit, &pairs);
            for (i, (&k, &error)) in keep.iter().zip(&r).enumerate() {
                assert_eq!(k, i % 13 != 0, "{proj:?}, star {i}, residual {error}");
                if k {
                    assert!(error < 0.1, "{proj:?}, star {i}, residual {error}");
                }
            }
        }
    }

    #[test]
    fn model_selection_reuses_the_exact_winning_linear_fit() {
        for proj in [Proj::Tan, Proj::Arc] {
            let (truth, pairs, weights) = fixture(proj, 1);
            let tangent = [truth.ra, truth.dec];
            let (expected, mask) =
                robust_fit_wmp(&pairs, &weights, truth.crpix, tangent, 1, 0, proj).unwrap();
            let (fit, keep, _, model) =
                select_model_wp(&pairs, &weights, truth.crpix, tangent, 5, 0, proj).unwrap();
            assert_eq!(model, (1, 0));
            assert_same_wcs(&fit, &expected);
            assert_eq!(keep, mask);
        }
    }

    #[test]
    fn model_selection_preserves_radial_baseline_and_linear_fallback() {
        let (mut truth, _, weights) = fixture(Proj::Tan, 1);
        let terms = Sip::terms(7);
        let mut sip = Sip {
            order: 7,
            a: vec![0.0; terms.len()],
            b: vec![0.0; terms.len()],
        };
        let pos = |p, q| terms.iter().position(|&t| t == (p, q)).unwrap();
        sip.a[pos(3, 0)] = 1e-8;
        sip.b[pos(0, 3)] = -1e-8;
        for (i, binom) in [1.0, 3.0, 3.0, 1.0].into_iter().enumerate() {
            let c = 0.08 * binom / 1500.0f64.powi(6);
            sip.a[pos(7 - 2 * i as u32, 2 * i as u32)] += c;
            sip.b[pos(6 - 2 * i as u32, 1 + 2 * i as u32)] += c;
        }
        truth.sip = Some(sip);
        let (_, base_pairs, _) = fixture(Proj::Tan, 1);
        let pairs: Vec<_> = base_pairs
            .iter()
            .map(|p| {
                let sky = truth.pixel_to_sky(p[0], p[1]);
                [p[0], p[1], sky[0], sky[1]]
            })
            .collect();
        let tangent = [truth.ra, truth.dec];
        let (expected, mask) =
            robust_fit_wmp(&pairs, &weights, truth.crpix, tangent, 3, 7, Proj::Tan).unwrap();
        let (fit, keep, _, model) =
            select_model_wp(&pairs, &weights, truth.crpix, tangent, 3, 7, Proj::Tan).unwrap();
        assert_eq!(model, (3, 7));
        assert_same_wcs(&fit, &expected);
        assert_eq!(keep, mask);
        // Twelve spatially spread pairs support TAN but not the radial baseline's 14 columns.
        let (linear, linear_pairs, _) = fixture(Proj::Tan, 1);
        let sparse: Vec<_> = linear_pairs.iter().step_by(27).copied().collect();
        assert_eq!(sparse.len(), 12);
        let tangent = [linear.ra, linear.dec];
        assert!(robust_fit_wmp(&sparse, &[], linear.crpix, tangent, 3, 7, Proj::Tan).is_none());
        let (expected, mask) =
            robust_fit_wmp(&sparse, &[], linear.crpix, tangent, 1, 0, Proj::Tan).unwrap();
        let (fit, keep, _, model) =
            select_model_wp(&sparse, &[], linear.crpix, tangent, 3, 7, Proj::Tan).unwrap();
        assert_eq!(model, (1, 0));
        assert_same_wcs(&fit, &expected);
        assert_eq!(keep, mask);
    }

    /// Strong barrel distortion (k1 = -0.1, k2 = 0.01 on the half diagonal, as in
    /// `lab/skyanchor_lab/synth.py`): a fit on the inner 60 % of the frame must predict the
    /// corners. Free order-3 SIP extrapolates poorly there; the radial extension does not.
    #[test]
    fn radial_extension_extrapolates_barrel_distortion() {
        let (w, h) = (6000.0f64, 4000.0f64);
        let c = [(w - 1.0) / 2.0, (h - 1.0) / 2.0];
        let rh = 0.5 * w.hypot(h);
        let tan = Wcs {
            ra: 80.0,
            dec: 20.0,
            crpix: c,
            cd: [[-1.0e-3, 0.0], [0.0, 1.0e-3]],
            sip: None,
            proj: Proj::Tan,
            native_pole: None,
            local_grid: None,
        };
        let (k1, k2) = (-0.1, 0.01);
        // ideal (undistorted) pixel -> distorted pixel, then sky of the ideal position
        let pair = |x: f64, y: f64| {
            let (dx, dy) = (x - c[0], y - c[1]);
            let r2 = (dx * dx + dy * dy) / (rh * rh);
            let f = 1.0 + k1 * r2 + k2 * r2 * r2;
            let s = tan.pixel_to_sky(x, y);
            [c[0] + dx * f, c[1] + dy * f, s[0], s[1]]
        };
        let mut inner = vec![];
        let mut outer = vec![];
        for i in 0..=60 {
            for j in 0..=40 {
                let p = pair(i as f64 * w / 60.0, j as f64 * h / 40.0);
                if p[0] < 0.0 || p[1] < 0.0 || p[0] > w - 1.0 || p[1] > h - 1.0 {
                    continue;
                }
                let r = (p[0] - c[0]).hypot(p[1] - c[1]) / rh;
                if r < 0.6 {
                    inner.push(p);
                } else if r > 0.85 {
                    outer.push(p);
                }
            }
        }
        let worst = |fit: &Wcs| residuals(fit, &outer).iter().fold(0.0f64, |a, &b| a.max(b));
        let poly = fit_tan_sip(&inner, &[], c, [80.0, 20.0], 3).unwrap();
        let rad = fit_tan_sip_model(&inner, &[], c, [80.0, 20.0], 3, 7).unwrap();
        assert_eq!(rad.sip.as_ref().unwrap().order, 7);
        let (ep, er) = (worst(&poly), worst(&rad));
        assert!(
            er < 2.0 && er < 0.2 * ep,
            "corner error radial {er:.2} px vs poly {ep:.2} px"
        );
        // the radial terms must also survive the FITS-style forward/inverse round trip
        for p in outer.iter().step_by(17) {
            let q = rad.sky_to_pixel(p[2], p[3]).unwrap();
            assert!((q[0] - p[0]).hypot(q[1] - p[1]) < 2.0);
        }
    }

    #[test]
    fn recovers_tan_sip_model() {
        let truth = Wcs {
            ra: 150.0,
            dec: 60.0,
            crpix: [2000.0, 1500.0],
            cd: [[-2.0e-4, 1.0e-5], [1.2e-5, 2.0e-4]],
            sip: Some(Sip {
                order: 3,
                a: vec![1e-7, -2e-7, 3e-8, 1e-10, -2e-11, 5e-11, 1e-11],
                b: vec![-5e-8, 1e-7, 2e-8, 2e-11, 1e-10, -1e-11, 3e-11],
            }),
            proj: Proj::Tan,
            native_pole: None,
            local_grid: None,
        };
        let mut pairs = vec![];
        for i in 0..20 {
            for j in 0..15 {
                let x = i as f64 * 200.0 + 13.0;
                let y = j as f64 * 200.0 + 7.0;
                let s = truth.pixel_to_sky(x, y);
                pairs.push([x, y, s[0], s[1]]);
            }
        }
        let fit = fit_tan_sip(&pairs, &[], [2000.0, 1500.0], [149.9, 60.1], 3).unwrap();
        for p in &pairs {
            let q = fit.sky_to_pixel(p[2], p[3]).unwrap();
            assert!(
                (q[0] - p[0]).hypot(q[1] - p[1]) < 1e-3,
                "{:?} vs {:?}",
                q,
                p
            );
        }
        let (_, _, err) = select_order(&pairs, [2000.0, 1500.0], [150.0, 60.0], 4).unwrap();
        assert!(err < 1e-3);
    }
}
