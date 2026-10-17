//! Satellite / airplane streaks and line defects (ARCHITECTURE §2.1, `lines.rs`).
//!
//! Two kinds of collinear chains are removed:
//! * **continuous streaks**: ≥ 4 collinear points whose accepted members are
//!   elongated along the chain and whose gaps are lit (the image stays above
//!   `bg + 2σ` between them) — peaks of a satellite trail or a bad column;
//! * **dotted lines**: ≥ 6 regularly spaced, collinear, similar sources
//!   (airplane strobes).
//!
//! Ridge-rejected candidates are used as gap fillers but never seed a chain, so
//! nebular structure cannot turn a group of stars into a "streak".
use crate::{
    extract::{Ctx, Det},
    types::star_flags as flag,
};
use rayon::prelude::*;

/// Members ordered along a chain `(projection, index)`, lateral RMS, origin
/// and unit direction.
type ChainGeometry = (Vec<(f32, usize)>, f32, [f32; 2], [f32; 2]);

struct Points<'a> {
    xy: Vec<[f32; 2]>,
    /// Number of accepted detections (the rest are ridge hints).
    n_det: usize,
    dets: &'a [Det],
    cell: f32,
    gw: usize,
    gh: usize,
    head: Vec<u32>,
    next: Vec<u32>,
}

impl<'a> Points<'a> {
    fn new(dets: &'a [Det], hints: &[[f32; 2]], w: usize, h: usize, cell: f32) -> Self {
        let mut xy: Vec<[f32; 2]> = dets.iter().map(|d| [d.x as f32, d.y as f32]).collect();
        xy.extend_from_slice(hints);
        let (gw, gh) = (
            (w as f32 / cell) as usize + 1,
            (h as f32 / cell) as usize + 1,
        );
        let mut head = vec![u32::MAX; gw * gh];
        let mut next = vec![u32::MAX; xy.len()];
        for (i, p) in xy.iter().enumerate() {
            let c = Self::cell_of(p, cell, gw, gh);
            next[i] = head[c];
            head[c] = i as u32;
        }
        Self {
            xy,
            n_det: dets.len(),
            dets,
            cell,
            gw,
            gh,
            head,
            next,
        }
    }

    fn cell_of(p: &[f32; 2], cell: f32, gw: usize, gh: usize) -> usize {
        let cx = ((p[0] / cell).max(0.0) as usize).min(gw - 1);
        let cy = ((p[1] / cell).max(0.0) as usize).min(gh - 1);
        cy * gw + cx
    }

    /// Visit all points inside the axis-aligned box [lo, hi].
    fn visit(&self, lo: [f32; 2], hi: [f32; 2], mut f: impl FnMut(usize)) {
        let c0 = (
            ((lo[0] / self.cell).max(0.0) as usize).min(self.gw - 1),
            ((lo[1] / self.cell).max(0.0) as usize).min(self.gh - 1),
        );
        let c1 = (
            ((hi[0] / self.cell).max(0.0) as usize).min(self.gw - 1),
            ((hi[1] / self.cell).max(0.0) as usize).min(self.gh - 1),
        );
        for cy in c0.1..=c1.1 {
            for cx in c0.0..=c1.0 {
                let mut j = self.head[cy * self.gw + cx];
                while j != u32::MAX {
                    let p = self.xy[j as usize];
                    if p[0] >= lo[0] && p[0] <= hi[0] && p[1] >= lo[1] && p[1] <= hi[1] {
                        f(j as usize);
                    }
                    j = self.next[j as usize];
                }
            }
        }
    }

    /// The `k` nearest accepted detections within `max_r` (expanding rings).
    fn nearest(&self, i: usize, max_r: f32, k_max: usize) -> Vec<(f32, usize)> {
        let p = self.xy[i];
        let (cx, cy) = (
            ((p[0] / self.cell).max(0.0) as isize).min(self.gw as isize - 1),
            ((p[1] / self.cell).max(0.0) as isize).min(self.gh as isize - 1),
        );
        let mut found: Vec<(f32, usize)> = vec![];
        let rings = (max_r / self.cell).ceil() as isize + 1;
        for k in 0..=rings {
            if found.len() >= k_max {
                found.sort_by(|a, b| a.0.total_cmp(&b.0));
                if found[k_max - 1].0 < (k - 1) as f32 * self.cell {
                    break;
                }
            }
            for gy in cy - k..=cy + k {
                for gx in cx - k..=cx + k {
                    if (gy - cy).abs().max((gx - cx).abs()) != k
                        || gx < 0
                        || gy < 0
                        || gx >= self.gw as isize
                        || gy >= self.gh as isize
                    {
                        continue;
                    }
                    let mut j = self.head[gy as usize * self.gw + gx as usize];
                    while j != u32::MAX {
                        let ju = j as usize;
                        if ju != i && ju < self.n_det {
                            let q = self.xy[ju];
                            let d = ((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2)).sqrt();
                            if d >= 1.0 && d <= max_r {
                                found.push((d, ju));
                            }
                        }
                        j = self.next[ju];
                    }
                }
            }
        }
        found.sort_by(|a, b| a.0.total_cmp(&b.0));
        found.truncate(k_max);
        found
    }

    /// Accepted detection elongated along direction `angle` (radians).
    fn aligned(&self, k: usize, angle: f32) -> bool {
        if k >= self.n_det {
            return false;
        }
        let d = &self.dets[k];
        let diff = (d.pa - (angle as f64).to_degrees()).rem_euclid(180.0);
        d.elong >= 1.4 && diff.min(180.0 - diff) < 15.0
    }

    /// Extend the chain (a, b) along its line in both directions: dotted chains
    /// look only near the next regularly spaced position; continuous chains take
    /// the nearest point in a corridor ahead (ridge hints allowed as fillers).
    fn grow(&self, a: usize, b: usize, ff: f32, dotted: bool) -> Vec<usize> {
        let mut chain = std::collections::VecDeque::from([a, b]);
        for forward in [true, false] {
            loop {
                let (p0, p1) = if forward {
                    (self.xy[chain[0]], self.xy[*chain.back().unwrap()])
                } else {
                    (self.xy[*chain.back().unwrap()], self.xy[chain[0]])
                };
                let len = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2)).sqrt();
                if len <= 0.0 || chain.len() >= 400 {
                    break;
                }
                let dir = [(p1[0] - p0[0]) / len, (p1[1] - p0[1]) / len];
                let spacing = len / (chain.len() - 1) as f32;
                if dotted {
                    let pred = [p1[0] + dir[0] * spacing, p1[1] + dir[1] * spacing];
                    let tol = 1.5 + 0.1 * spacing;
                    let lo = [pred[0] - tol, pred[1] - tol];
                    let hi = [pred[0] + tol, pred[1] + tol];
                    let mut best: Option<(f32, usize)> = None;
                    self.visit(lo, hi, |k| {
                        if k >= self.n_det || chain.contains(&k) {
                            return;
                        }
                        let p = self.xy[k];
                        let d = ((p[0] - pred[0]).powi(2) + (p[1] - pred[1]).powi(2)).sqrt();
                        if d < tol && best.is_none_or(|b| d < b.0) {
                            best = Some((d, k));
                        }
                    });
                    match best {
                        Some((_, k)) if forward => chain.push_back(k),
                        Some((_, k)) => chain.push_front(k),
                        None => break,
                    }
                    continue;
                }
                let g_max = (1.8 * spacing).max(4.0 * ff).min(150.0);
                let tol = 1.5 + 0.005 * len;
                let end = [p1[0] + dir[0] * g_max, p1[1] + dir[1] * g_max];
                let lo = [p1[0].min(end[0]) - tol, p1[1].min(end[1]) - tol];
                let hi = [p1[0].max(end[0]) + tol, p1[1].max(end[1]) + tol];
                let mut best: Option<(f32, usize)> = None;
                self.visit(lo, hi, |k| {
                    if chain.contains(&k) {
                        return;
                    }
                    let p = self.xy[k];
                    let (dx, dy) = (p[0] - p1[0], p[1] - p1[1]);
                    let t = dx * dir[0] + dy * dir[1];
                    if t <= 0.5 || t > g_max {
                        return;
                    }
                    let (ex, ey) = (p[0] - p0[0], p[1] - p0[1]);
                    let lateral = (ex * dir[1] - ey * dir[0]).abs();
                    if lateral < tol && best.is_none_or(|b| t < b.0) {
                        best = Some((t, k));
                    }
                });
                match best {
                    Some((_, k)) if forward => chain.push_back(k),
                    Some((_, k)) => chain.push_front(k),
                    None => break,
                }
            }
        }
        chain.into()
    }

    /// Members ordered along the chain, with their projections and lateral RMS.
    fn geometry(&self, chain: &[usize]) -> ChainGeometry {
        let (p0, p1) = (self.xy[chain[0]], self.xy[chain[chain.len() - 1]]);
        let len = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2))
            .sqrt()
            .max(1e-6);
        let dir = [(p1[0] - p0[0]) / len, (p1[1] - p0[1]) / len];
        let mut t: Vec<(f32, usize)> = chain
            .iter()
            .map(|&k| {
                let p = self.xy[k];
                ((p[0] - p0[0]) * dir[0] + (p[1] - p0[1]) * dir[1], k)
            })
            .collect();
        t.sort_by(|a, b| a.0.total_cmp(&b.0));
        let rms = (chain
            .iter()
            .map(|&k| {
                let p = self.xy[k];
                ((p[0] - p0[0]) * dir[1] - (p[1] - p0[1]) * dir[0]).powi(2)
            })
            .sum::<f32>()
            / chain.len() as f32)
            .sqrt();
        (t, rms, p0, dir)
    }
}

/// Fraction of the gaps between consecutive members that is lit (above
/// `bg + 2σ` on the binned plane), ignoring 1.5 FWHM around each member.
/// Fractions of the gaps between consecutive members that are lit (above
/// `bg + 2σ` on the binned plane) on the chain and at `off` px to either side,
/// ignoring 1.5 FWHM around each member.
fn lit_fraction(
    ctx: &Ctx,
    t: &[(f32, usize)],
    p0: [f32; 2],
    dir: [f32; 2],
    ff: f32,
    off: f32,
) -> (f32, f32) {
    let b = ctx.b as f32;
    let lit_at = |x: f32, y: f32| -> Option<bool> {
        let (xb, yb) = ((x + 0.5) / b - 0.5, (y + 0.5) / b - 0.5);
        let v = ctx.plane.bilinear(xb, yb);
        v.is_finite().then(|| {
            let (bg, sig, _) = ctx.grid.at(xb, yb);
            v > bg + 2.0 * sig
        })
    };
    let normal = [-dir[1], dir[0]];
    let (mut lit, mut side, mut total) = (0usize, 0usize, 0usize);
    for pair in t.windows(2) {
        let (ta, tb) = (pair[0].0 + 1.5 * ff, pair[1].0 - 1.5 * ff);
        let mut s = ta;
        while s < tb {
            let (x, y) = (p0[0] + dir[0] * s, p0[1] + dir[1] * s);
            if let Some(on) = lit_at(x, y) {
                total += 1;
                lit += on as usize;
                let a = lit_at(x + normal[0] * off, y + normal[1] * off).unwrap_or(false);
                let c = lit_at(x - normal[0] * off, y - normal[1] * off).unwrap_or(false);
                side += (a && c) as usize;
            }
            s += 1.0;
        }
    }
    if total < 3 {
        (1.0, 0.0)
    } else {
        (lit as f32 / total as f32, side as f32 / total as f32)
    }
}

/// Flag members of streaks and dotted lines with [`flag::ON_LINE`].
pub(crate) fn flag_lines(ctx: &Ctx, dets: &mut [Det], hints: &[[f32; 2]]) {
    if dets.len() < 4 {
        return;
    }
    let ff = ctx.fwhm_frame.max(2.0);
    let pts = Points::new(dets, hints, ctx.w, ctx.h, 16.0);
    let n = pts.n_det;
    let r_pair = (8.0 * ff).clamp(12.0, 96.0);
    let chains: Vec<Vec<usize>> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let mut out: Vec<Vec<usize>> = vec![];
            let p = pts.xy[i];
            let d = &pts.dets[i];
            let mut seeds: Vec<usize> = vec![];
            // Streak peaks are strongly elongated along the streak.
            if d.elong >= 1.8 && d.snr >= 8.0 {
                let lo = [p[0] - r_pair, p[1] - r_pair];
                let hi = [p[0] + r_pair, p[1] + r_pair];
                pts.visit(lo, hi, |j| {
                    let q = pts.xy[j];
                    let dist = ((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2)).sqrt();
                    if j != i
                        && (1.0..=r_pair).contains(&dist)
                        && pts.aligned(i, (q[1] - p[1]).atan2(q[0] - p[0]))
                        && (j >= n || pts.aligned(j, (q[1] - p[1]).atan2(q[0] - p[0])))
                    {
                        seeds.push(j);
                    }
                });
            }
            // The nearest accepted neighbours seed the dotted-line search.
            let nearest = pts.nearest(i, 160.0, 4);
            for j in seeds {
                let chain = pts.grow(i, j, ff, false);
                if chain.len() >= 4 && continuous(ctx, &pts, &chain, ff) {
                    // Round stars that merely lie on a trail are kept.
                    let (_, _, _, dir) = pts.geometry(&chain);
                    let angle = dir[1].atan2(dir[0]);
                    out.push(
                        chain
                            .into_iter()
                            .filter(|&k| pts.aligned(k, angle))
                            .collect(),
                    );
                }
            }
            for (_, j) in nearest {
                let (fi, fj) = (pts.dets[i].flux, pts.dets[j].flux);
                if j > i && fi < 3.0 * fj && fj < 3.0 * fi {
                    let chain = pts.grow(i, j, ff, true);
                    if chain.len() >= 6 && dotted(&pts, &chain) {
                        out.push(chain);
                    }
                }
            }
            out
        })
        .collect();
    for chain in chains {
        for k in chain {
            if k < n {
                dets[k].flags |= flag::ON_LINE;
            }
        }
    }
}

/// Continuous streak: tight line, lit gaps, members elongated along it.
fn continuous(ctx: &Ctx, pts: &Points, chain: &[usize], ff: f32) -> bool {
    let (t, rms, p0, dir) = pts.geometry(chain);
    if rms > 1.2 {
        return false;
    }
    let angle = dir[1].atan2(dir[0]);
    let accepted: Vec<usize> = chain.iter().copied().filter(|&k| k < pts.n_det).collect();
    let aligned = accepted.iter().filter(|&&k| pts.aligned(k, angle)).count();
    if aligned < 2 || aligned * 10 < accepted.len() * 6 {
        return false;
    }
    let span = t[t.len() - 1].0 - t[0].0;
    if span < 6.0 * ff {
        return false;
    }
    // A trail is narrow: lit along the chain, dark two FWHM to either side
    // (a galaxy arm or a nebular filament is lit on both sides).
    let (on, beside) = lit_fraction(ctx, &t, p0, dir, ff, 2.0 * ff + 2.0);
    on >= 0.75 && beside <= 0.35
}

/// Dotted line: regular spacing, very straight, similar brightness.
fn dotted(pts: &Points, chain: &[usize]) -> bool {
    let (t, rms, _, _) = pts.geometry(chain);
    if rms > 0.7 {
        return false;
    }
    let gaps: Vec<f32> = t.windows(2).map(|p| p[1].0 - p[0].0).collect();
    let mean = gaps.iter().sum::<f32>() / gaps.len() as f32;
    let var = gaps.iter().map(|g| (g - mean).powi(2)).sum::<f32>() / gaps.len() as f32;
    if mean < 3.0 || var.sqrt() > 0.15 * mean {
        return false;
    }
    let flux: Vec<f64> = chain.iter().map(|&k| pts.dets[k].flux).collect();
    let (lo, hi) = flux
        .iter()
        .fold((f64::INFINITY, 0.0f64), |a, &f| (a.0.min(f), a.1.max(f)));
    hi <= 5.0 * lo
}
