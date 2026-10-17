use crate::types::Star;
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub struct Pattern {
    pub ids: [usize; 4],
    pub code: [f64; 4],
    pub baseline: f64,
    pub quality: f64,
    pub features: [f64; 6],
}
pub fn distance(a: &Star, b: &Star) -> f64 {
    (a.x - b.x).hypot(a.y - b.y)
}
fn variant(stars: &[Star], ids: [usize; 4], flip: f64) -> Option<([f64; 4], f64)> {
    let (a, b, c, d) = (
        &stars[ids[0]],
        &stars[ids[1]],
        &stars[ids[2]],
        &stars[ids[3]],
    );
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let len = dx.hypot(dy);
    if len < 2.0 {
        return None;
    }
    let norm = |s: &Star| {
        let (x, y) = (s.x - a.x, s.y - a.y);
        [
            (x * dx + y * dy) / (len * len),
            0.5 + flip * (y * dx - x * dy) / (2.0 * len * len),
        ]
    };
    let u = norm(c);
    let v = norm(d);
    Some(([u[0], u[1], v[0], v[1]], len))
}
pub fn variants(stars: &[Star], ids: [usize; 4]) -> Vec<Pattern> {
    let mut edges = Vec::new();
    for i in 0..4 {
        for j in i + 1..4 {
            edges.push((distance(&stars[ids[i]], &stars[ids[j]]), i, j));
        }
    }
    edges.sort_by(|a, b| b.0.total_cmp(&a.0));
    let max = edges[0].0;
    let mut out = vec![];
    for &(len, i, j) in edges.iter().take(3) {
        if len < max * 0.98 {
            break;
        }
        let rest: Vec<_> = (0..4).filter(|&x| x != i && x != j).collect();
        for swap in [false, true] {
            for rev in [false, true] {
                for sign in [-1.0, 1.0] {
                    let (a, b) = if rev { (j, i) } else { (i, j) };
                    let (c, d) = if swap {
                        (rest[1], rest[0])
                    } else {
                        (rest[0], rest[1])
                    };
                    let ix = [ids[a], ids[b], ids[c], ids[d]];
                    if let Some((code, baseline)) = variant(stars, ix, sign) {
                        if code.iter().any(|v| *v < 0.0 || *v > 1.0) {
                            continue;
                        }
                        let area = ((code[1] - 0.5).abs() + (code[3] - 0.5).abs()).min(0.5);
                        if area < 0.015 {
                            continue;
                        }
                        let sigma = ix.iter().map(|&k| stars[k].sigma).sum::<f64>() / 4.0;
                        let snr = ix
                            .iter()
                            .map(|&k| stars[k].snr)
                            .fold(f64::INFINITY, f64::min)
                            .max(1.0);
                        let features = [
                            (snr.ln() / 5.0).min(2.0),
                            (sigma / baseline * 100.0).min(2.0),
                            area,
                            baseline.ln() / 10.0,
                            0.0,
                            ix.iter().filter(|&&k| stars[k].saturated).count() as f64 / 4.0,
                        ];
                        let quality = snr.ln() * area / (0.01 + sigma / baseline * 100.0);
                        out.push(Pattern {
                            ids: ix,
                            code,
                            baseline,
                            quality,
                            features,
                        });
                    }
                }
            }
        }
    }
    out
}
pub fn canonical(stars: &[Star], ids: [usize; 4]) -> Option<Pattern> {
    variants(stars, ids)
        .into_iter()
        .min_by(|a, b| a.code.partial_cmp(&b.code).unwrap())
}

/// Deterministic, brightness-ranked spatial round-robin. No sky/scale metadata.
pub fn balanced(stars: &[Star], width: u32, height: u32) -> Vec<usize> {
    let mut cells: Vec<Vec<usize>> = vec![vec![]; 16];
    for (i, s) in stars.iter().enumerate() {
        if !s.x.is_finite() || !s.y.is_finite() || !s.flux.is_finite() {
            continue;
        }
        let x = ((s.x / width.max(1) as f64 * 4.0).floor() as usize).min(3);
        let y = ((s.y / height.max(1) as f64 * 4.0).floor() as usize).min(3);
        cells[y * 4 + x].push(i);
    }
    for c in &mut cells {
        c.sort_by(|&a, &b| stars[b].flux.total_cmp(&stars[a].flux));
    }
    let mut out = vec![];
    let mut depth = 0;
    loop {
        let mut layer: Vec<usize> = cells.iter().filter_map(|c| c.get(depth).copied()).collect();
        if layer.is_empty() {
            break;
        }
        layer.sort_by(|&a, &b| stars[b].flux.total_cmp(&stars[a].flux));
        out.extend(layer);
        depth += 1;
    }
    out
}
pub fn enumerate(stars: &[Star], width: u32, height: u32) -> Vec<Pattern> {
    Emitter::new(stars, width, height).collect()
}

/// Iterative widening: preserve the order of the exhaustive emitter, but do not
/// construct deeper tiers until the previous tier has failed to solve the image.
pub struct Emitter<'a> {
    stars: &'a [Star],
    ranked: Vec<usize>,
    seen: HashSet<[usize; 4]>,
    tier: usize,
    pending: std::vec::IntoIter<Pattern>,
}
impl<'a> Emitter<'a> {
    pub fn new(stars: &'a [Star], width: u32, height: u32) -> Self {
        Self {
            stars,
            ranked: balanced(stars, width, height),
            seen: HashSet::new(),
            tier: 0,
            pending: vec![].into_iter(),
        }
    }
    fn expand(&mut self, tier: usize) {
        let stars = self.stars;
        let selected = &self.ranked[..self.ranked.len().min(tier)];
        let mut batch = vec![];
        for &anchor in selected {
            let mut neighbors: Vec<usize> =
                selected.iter().copied().filter(|&i| i != anchor).collect();
            neighbors.sort_by(|&a, &b| {
                distance(&stars[anchor], &stars[a]).total_cmp(&distance(&stars[anchor], &stars[b]))
            });
            neighbors.truncate(8);
            for a in 0..neighbors.len() {
                for b in a + 1..neighbors.len() {
                    for c in b + 1..neighbors.len() {
                        let mut ids = [anchor, neighbors[a], neighbors[b], neighbors[c]];
                        ids.sort();
                        if !self.seen.insert(ids) {
                            continue;
                        }
                        if let Some(p) = canonical(stars, ids) {
                            batch.push(p);
                        }
                    }
                }
            }
        }
        batch.sort_by(|a, b| b.quality.total_cmp(&a.quality));
        self.pending = batch.into_iter();
    }
}
impl Iterator for Emitter<'_> {
    type Item = Pattern;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(p) = self.pending.next() {
                return Some(p);
            }
            let &size = [12, 24, 48, 96].get(self.tier)?;
            self.tier += 1;
            self.expand(size);
        }
    }
}
pub const BINS: usize = 1 << 20;
pub fn bin(code: [f64; 4]) -> usize {
    let mut v = 0;
    for x in code {
        v = (v << 5) | ((x.clamp(0.0, 0.999999) * 32.0) as usize);
    }
    v
}
pub fn quantize(code: [f64; 4]) -> [u16; 4] {
    code.map(|v| (v.clamp(0.0, 1.0) * 65535.0).round() as u16)
}

#[cfg(test)]
mod emitter_tests {
    use super::*;
    #[test]
    fn deeper_tiers_are_not_built_before_they_are_requested() {
        let stars: Vec<_> = (0..100)
            .map(|i| Star {
                x: ((i * 137) % 997) as f64,
                y: ((i * i * 53) % 751) as f64,
                flux: 1000.0 - i as f64,
                sigma: 0.2,
                snr: 40.0,
                fwhm: 3.0,
                elongation: 1.0,
                saturated: false,
                ..Default::default()
            })
            .collect();
        let mut emitter = Emitter::new(&stars, 1000, 760);
        assert_eq!(emitter.tier, 0);
        let first = emitter.next().unwrap();
        assert_eq!(emitter.tier, 1);
        assert!(first.ids.iter().all(|id| emitter.ranked[..12].contains(id)));
        let all: Vec<_> = std::iter::once(first).chain(emitter.by_ref()).collect();
        let unique: HashSet<_> = all
            .iter()
            .map(|p| {
                let mut ids = p.ids;
                ids.sort();
                ids
            })
            .collect();
        assert_eq!(unique.len(), all.len());
        assert!(
            all.iter()
                .any(|p| p.ids.iter().any(|id| !emitter.ranked[..48].contains(id)))
        );
    }
}
