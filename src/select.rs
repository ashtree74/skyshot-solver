//! Image-side dominance: d_i = distance (px) from detection i to the nearest *brighter*
//! detection, and the rung ladder r_k = 2^(k/4) px whose pools {d >= r} reproduce catalogue
//! dominance levels without knowing the pixel scale.

/// Uniform 2-D grid of point indices for k-NN queries.
pub struct Grid2 {
    cell: f64,
    nx: usize,
    ny: usize,
    offsets: Vec<u32>,
    items: Vec<u32>,
}

impl Grid2 {
    /// Build over `ids` (indices into `pts`) with roughly `per_cell` points per cell.
    pub fn build(pts: &[[f64; 2]], ids: &[u32], width: f64, height: f64, per_cell: f64) -> Self {
        let n = ids.len().max(1) as f64;
        let cell = ((width * height * per_cell / n).sqrt()).max(1.0);
        let nx = ((width / cell).ceil() as usize).max(1);
        let ny = ((height / cell).ceil() as usize).max(1);
        let mut counts = vec![0u32; nx * ny + 1];
        let cidx = |p: [f64; 2]| -> usize {
            let x = ((p[0] / cell).floor().max(0.0) as usize).min(nx - 1);
            let y = ((p[1] / cell).floor().max(0.0) as usize).min(ny - 1);
            y * nx + x
        };
        for &i in ids {
            counts[cidx(pts[i as usize]) + 1] += 1;
        }
        for i in 0..nx * ny {
            counts[i + 1] += counts[i];
        }
        let offsets = counts.clone();
        let mut fill = counts;
        let mut items = vec![0u32; ids.len()];
        for &i in ids {
            let c = cidx(pts[i as usize]);
            items[fill[c] as usize] = i;
            fill[c] += 1;
        }
        Self {
            cell,
            nx,
            ny,
            offsets,
            items,
        }
    }

    /// Lower bound on distance to one axis of a cell. Border cells are unbounded
    /// outwards because `build` clamps off-frame detections into those cells.
    #[inline]
    fn cell_axis_distance(&self, q: f64, c: usize, count: usize) -> f64 {
        let lo = if c == 0 {
            f64::NEG_INFINITY
        } else {
            (c as f64 * self.cell).next_down()
        };
        let hi = if c + 1 == count {
            f64::INFINITY
        } else {
            ((c + 1) as f64 * self.cell).next_up()
        };
        (lo - q).max(q - hi).max(0.0)
    }

    /// Minimum distance to any cell outside the visited square. Round boundaries
    /// outwards so multiplication rounding cannot exclude a point assigned by division.
    #[inline]
    fn outside_distance(&self, q: [f64; 2], cx: i64, cy: i64, ring: i64) -> f64 {
        let mut distance = f64::INFINITY;
        for (p, c, count) in [(q[0], cx, self.nx), (q[1], cy, self.ny)] {
            if c - ring > 0 {
                let edge = ((c - ring) as f64 * self.cell).next_up();
                distance = distance.min((p - edge).max(0.0));
            }
            if c + ring + 1 < count as i64 {
                let edge = ((c + ring + 1) as f64 * self.cell).next_down();
                distance = distance.min((edge - p).max(0.0));
            }
        }
        distance
    }

    /// Nearest point within an inclusive pixel radius, excluding `skip`.
    /// Visits cells and items in the same order as `knn(..., 1, ...)`, preserving
    /// its first-visited tie rule. Cells outside the radius or farther than the
    /// incumbent are skipped; the search ends when unseen cells cannot improve it.
    pub fn nearest_within(
        &self,
        pts: &[[f64; 2]],
        q: [f64; 2],
        radius: f64,
        skip: u32,
    ) -> Option<(f64, u32)> {
        if radius.is_nan() || radius < 0.0 || !q[0].is_finite() || !q[1].is_finite() {
            return None;
        }
        let r2 = radius * radius;
        let cx = ((q[0] / self.cell).floor().max(0.0) as i64).min(self.nx as i64 - 1);
        let cy = ((q[1] / self.cell).floor().max(0.0) as i64).min(self.ny as i64 - 1);
        let mut best: Option<(f64, u32)> = None;
        for ring in 0..=self.nx.max(self.ny) as i64 {
            for y in (cy - ring)..=(cy + ring) {
                if y < 0 || y >= self.ny as i64 {
                    continue;
                }
                let dy = self.cell_axis_distance(q[1], y as usize, self.ny);
                let full = y == cy - ring || y == cy + ring;
                let mut x = cx - ring;
                while x <= cx + ring {
                    if x >= 0 && x < self.nx as i64 {
                        let dx = self.cell_axis_distance(q[0], x as usize, self.nx);
                        let limit = best.map_or(r2, |b| b.0);
                        if dx * dx + dy * dy <= limit {
                            let c = y as usize * self.nx + x as usize;
                            for &i in
                                &self.items[self.offsets[c] as usize..self.offsets[c + 1] as usize]
                            {
                                if i == skip {
                                    continue;
                                }
                                let p = pts[i as usize];
                                let d2 = (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2);
                                if d2 <= r2 && best.is_none_or(|b| d2 < b.0) {
                                    best = Some((d2, i));
                                }
                            }
                        }
                    }
                    x += if full || ring == 0 { 1 } else { 2 * ring };
                }
            }
            let outside = self.outside_distance(q, cx, cy, ring);
            let outside2 = outside * outside;
            // Strict comparison retains points on the inclusive radius boundary.
            // Once there is an incumbent, an equal-distance later point cannot replace it.
            if best.is_some_and(|b| b.0 <= outside2) || r2 < outside2 {
                return best;
            }
        }
        best
    }

    /// k nearest points to `q` (excluding `skip`), sorted by distance; writes (dist2, id).
    pub fn knn(
        &self,
        pts: &[[f64; 2]],
        q: [f64; 2],
        k: usize,
        skip: u32,
        out: &mut Vec<(f64, u32)>,
    ) {
        out.clear();
        let cx = ((q[0] / self.cell).floor().max(0.0) as i64).min(self.nx as i64 - 1);
        let cy = ((q[1] / self.cell).floor().max(0.0) as i64).min(self.ny as i64 - 1);
        let maxr = self.nx.max(self.ny) as i64;
        for ring in 0..=maxr {
            // scan cells on the square ring
            for y in (cy - ring)..=(cy + ring) {
                if y < 0 || y >= self.ny as i64 {
                    continue;
                }
                let full = y == cy - ring || y == cy + ring;
                let mut x = cx - ring;
                while x <= cx + ring {
                    if x >= 0 && x < self.nx as i64 {
                        let c = y as usize * self.nx + x as usize;
                        for &i in
                            &self.items[self.offsets[c] as usize..self.offsets[c + 1] as usize]
                        {
                            if i == skip {
                                continue;
                            }
                            let p = pts[i as usize];
                            let d2 = (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2);
                            if out.len() < k {
                                out.push((d2, i));
                                if out.len() == k {
                                    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                                }
                            } else if d2 < out[k - 1].0 {
                                out[k - 1] = (d2, i);
                                let mut j = k - 1;
                                while j > 0 && out[j].0 < out[j - 1].0 {
                                    out.swap(j, j - 1);
                                    j -= 1;
                                }
                            }
                        }
                    }
                    x += if full || ring == 0 { 1 } else { 2 * ring };
                }
            }
            // all points within `ring * cell` of q have been seen
            if out.len() == k {
                let covered = ring as f64 * self.cell;
                if out[k - 1].0 <= covered * covered {
                    return;
                }
            }
        }
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }
}

/// Dominance distances in pixels for points sorted by brightness (index 0 = brightest).
/// Returns f64::INFINITY for points without a brighter neighbour.
pub fn dominance_px(pts: &[[f64; 2]], width: f64, height: f64) -> Vec<f64> {
    let n = pts.len();
    let mut d = vec![f64::INFINITY; n];
    if n < 2 {
        return d;
    }
    // Incremental grid: cells hold already-inserted (brighter) points.
    let cell = ((width * height * 2.0 / n as f64).sqrt()).max(2.0);
    let nx = ((width / cell).ceil() as usize).max(1);
    let ny = ((height / cell).ceil() as usize).max(1);
    let mut cells: Vec<Vec<u32>> = vec![Vec::new(); nx * ny];
    let cidx = |p: [f64; 2]| -> (i64, i64) {
        (
            ((p[0] / cell).floor().max(0.0) as i64).min(nx as i64 - 1),
            ((p[1] / cell).floor().max(0.0) as i64).min(ny as i64 - 1),
        )
    };
    let maxr = nx.max(ny) as i64;
    for i in 0..n {
        let q = pts[i];
        let (cx, cy) = cidx(q);
        if i > 0 {
            let mut best = f64::INFINITY;
            for ring in 0..=maxr {
                for y in (cy - ring)..=(cy + ring) {
                    if y < 0 || y >= ny as i64 {
                        continue;
                    }
                    let full = y == cy - ring || y == cy + ring;
                    let mut x = cx - ring;
                    while x <= cx + ring {
                        if x >= 0 && x < nx as i64 {
                            for &j in &cells[y as usize * nx + x as usize] {
                                let p = pts[j as usize];
                                let d2 = (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2);
                                if d2 < best {
                                    best = d2;
                                }
                            }
                        }
                        x += if full || ring == 0 { 1 } else { 2 * ring };
                    }
                }
                let covered = ring as f64 * cell;
                if best <= covered * covered {
                    break;
                }
            }
            d[i] = best.sqrt();
        }
        cells[cy as usize * nx + cx as usize].push(i as u32);
    }
    d
}

/// Distance from each point to the nearest frame edge.
pub fn edge_px(pts: &[[f64; 2]], width: f64, height: f64) -> Vec<f64> {
    pts.iter()
        .map(|p| {
            p[0].min(p[1])
                .min(width - 1.0 - p[0])
                .min(height - 1.0 - p[1])
                .max(0.0)
        })
        .collect()
}

pub const RUNG_STEP_LOG2: f64 = 0.25;

#[inline]
pub fn rung_radius(k: i32) -> f64 {
    (k as f64 * RUNG_STEP_LOG2).exp2()
}

/// Rung indices ordered for search: start where the pool has about `target` members, then
/// alternate outward, keeping pools in [min_pool, max_pool].
pub fn rung_order(d: &[f64], target: usize, min_pool: usize, max_pool: usize) -> Vec<i32> {
    let mut finite: Vec<f64> = d.iter().copied().filter(|x| x.is_finite()).collect();
    finite.sort_by(|a, b| b.partial_cmp(a).unwrap());
    // the pool at radius r has |{d >= r}| members (+1 for the brightest star with d = inf)
    let pool_at = |r: f64| finite.partition_point(|&x| x >= r) + 1;
    if finite.len() + 1 < min_pool {
        return vec![];
    }
    let rmax = finite[0];
    let kmax = (rmax.log2() / RUNG_STEP_LOG2).floor() as i32;
    let mut valid = vec![];
    let mut k = kmax;
    while k > -8 {
        let n = pool_at(rung_radius(k));
        if n > max_pool {
            break;
        }
        if n >= min_pool {
            valid.push((k, n));
        }
        k -= 1;
    }
    if valid.is_empty() {
        return vec![];
    }
    // starting rung: pool closest to target (in log space)
    let start = valid
        .iter()
        .enumerate()
        .min_by(|a, b| {
            let fa = ((a.1.1 as f64) / target as f64).ln().abs();
            let fb = ((b.1.1 as f64) / target as f64).ln().abs();
            fa.partial_cmp(&fb).unwrap()
        })
        .map(|(i, _)| i)
        .unwrap();
    let mut out = vec![valid[start].0];
    let (mut lo, mut hi) = (start as i64 - 1, start + 1);
    while lo >= 0 || hi < valid.len() {
        if hi < valid.len() {
            out.push(valid[hi].0);
            hi += 1;
        }
        if lo >= 0 {
            out.push(valid[lo as usize].0);
            lo -= 1;
        }
    }
    // remove consecutive rungs with identical pools
    let mut seen = std::collections::HashSet::new();
    out.retain(|&k| seen.insert(pool_at(rung_radius(k))));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_nearest_matches_brute_force_and_knn_inside_and_outside_frame() {
        let mut state = 615u64;
        let mut rnd = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        // Out-of-frame detections are clamped into edge cells by Grid2::build.
        let pts: Vec<_> = (0..700)
            .map(|_| [rnd() * 1200.0 - 100.0, rnd() * 1000.0 - 100.0])
            .collect();
        let ids: Vec<_> = (0..700).step_by(3).rev().collect();
        let grid = Grid2::build(&pts, &ids, 1000.0, 800.0, 2.0);
        let mut nearest = vec![];
        for _ in 0..300 {
            let q = [rnd() * 1800.0 - 400.0, rnd() * 1600.0 - 400.0];
            let skip = (rnd() * 700.0) as u32;
            grid.knn(&pts, q, 1, skip, &mut nearest);
            let brute = ids
                .iter()
                .filter(|&&i| i != skip)
                .map(|&i| {
                    (
                        (pts[i as usize][0] - q[0]).powi(2) + (pts[i as usize][1] - q[1]).powi(2),
                        i,
                    )
                })
                .min_by(|a, b| a.0.total_cmp(&b.0));
            for radius in [0.0, 1.0, 5.0, 20.0, 80.0, 400.0, f64::INFINITY] {
                let got = grid.nearest_within(&pts, q, radius, skip);
                let expect = nearest.first().copied().filter(|p| p.0 <= radius * radius);
                assert_eq!(got, expect, "q={q:?}, radius={radius}");
                assert_eq!(got, brute.filter(|p| p.0 <= radius * radius));
            }
        }
    }

    #[test]
    fn bounded_nearest_preserves_ties_and_inclusive_boundaries() {
        let pts = [
            [19.0, 20.0],
            [21.0, 20.0],
            [20.0, 19.0],
            [20.0, 21.0],
            [21.0, 20.0],
            [0.0, 0.0],
            [79.0, 79.0],
            [30.0, 20.0],
        ];
        let ids = [4, 0, 1, 2, 3, 5, 6, 7];
        // Eight points and per_cell=0.125 give exactly ten-pixel cells.
        let grid = Grid2::build(&pts, &ids, 80.0, 80.0, 0.125);
        assert_eq!(grid.cell, 10.0);
        assert_eq!(
            grid.nearest_within(&pts, [20.0, 20.0], 1.0, u32::MAX),
            Some((1.0, 4))
        );
        assert_eq!(
            grid.nearest_within(&pts, [20.0, 20.0], 1.0, 4),
            Some((1.0, 1))
        );
        assert_eq!(
            grid.nearest_within(&pts, [20.0, 20.0], 1.0 - f64::EPSILON, u32::MAX),
            None
        );
        assert_eq!(
            grid.nearest_within(&pts, [21.0, 20.0], 0.0, u32::MAX),
            Some((0.0, 4))
        );
        let mut nearest = vec![];
        for q in [
            [20.0, 20.0],
            [20.0 - 1e-12, 20.0],
            [20.0 + 1e-12, 20.0],
            [-1.0, 0.0],
            [80.0, 79.0],
        ] {
            for skip in [u32::MAX, 4, 1, 5, 6] {
                grid.knn(&pts, q, 1, skip, &mut nearest);
                for radius in [0.0, 1.0, 10.0, 100.0] {
                    assert_eq!(
                        grid.nearest_within(&pts, q, radius, skip),
                        nearest.first().copied().filter(|p| p.0 <= radius * radius)
                    );
                }
            }
        }
        let empty = Grid2::build(&pts, &[], 80.0, 80.0, 2.0);
        assert_eq!(
            empty.nearest_within(&pts, [20.0, 20.0], 5.0, u32::MAX),
            None
        );
        assert_eq!(
            grid.nearest_within(&pts, [20.0, 20.0], f64::NAN, u32::MAX),
            None
        );
        assert_eq!(
            grid.nearest_within(&pts, [20.0, 20.0], -1.0, u32::MAX),
            None
        );
    }

    #[test]
    fn bounded_nearest_keeps_points_adjacent_to_rounded_cell_edges() {
        let mut pts = vec![];
        for edge in [0.0_f64, 10.0, 20.0, 190.0, 200.0] {
            for x in [edge.next_down(), edge, edge.next_up()] {
                pts.push([x, 20.0]);
                pts.push([20.0, x]);
            }
        }
        pts.extend((0..10).map(|i| [100.0 + i as f64, 130.0]));
        let ids: Vec<_> = (0..pts.len() as u32).collect();
        let grid = Grid2::build(&pts, &ids, 200.0, 200.0, 0.1);
        assert_eq!(grid.cell, 10.0);
        for &q in &pts {
            for radius in [0.0, 1e-12, 10.0, 20.0] {
                let got = grid.nearest_within(&pts, q, radius, u32::MAX).unwrap();
                assert_eq!(got.0, 0.0);
                let mut legacy = vec![];
                grid.knn(&pts, q, 1, u32::MAX, &mut legacy);
                assert_eq!(Some(got), legacy.first().copied());
            }
        }
    }

    #[test]
    fn dominance_matches_brute_force() {
        let mut s = 7u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let pts: Vec<[f64; 2]> = (0..3000)
            .map(|_| [rnd() * 4000.0, rnd() * 3000.0])
            .collect();
        let d = dominance_px(&pts, 4000.0, 3000.0);
        for i in (0..pts.len()).step_by(37) {
            let mut best = f64::INFINITY;
            for j in 0..i {
                best = best.min(
                    ((pts[i][0] - pts[j][0]).powi(2) + (pts[i][1] - pts[j][1]).powi(2)).sqrt(),
                );
            }
            assert!(
                (d[i] - best).abs() < 1e-9 || (d[i].is_infinite() && best.is_infinite()),
                "{i}"
            );
        }
    }

    #[test]
    fn knn_matches_brute_force() {
        let mut s = 99u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let pts: Vec<[f64; 2]> = (0..2000).map(|_| [rnd() * 1000.0, rnd() * 800.0]).collect();
        let ids: Vec<u32> = (0..2000).step_by(3).collect();
        let g = Grid2::build(&pts, &ids, 1000.0, 800.0, 2.0);
        let mut out = vec![];
        for q in 0..200u32 {
            let qp = pts[q as usize * 7 % 2000];
            g.knn(&pts, qp, 6, q * 7 % 2000, &mut out);
            let mut bf: Vec<(f64, u32)> = ids
                .iter()
                .filter(|&&i| i != q * 7 % 2000)
                .map(|&i| {
                    (
                        (pts[i as usize][0] - qp[0]).powi(2) + (pts[i as usize][1] - qp[1]).powi(2),
                        i,
                    )
                })
                .collect();
            bf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for t in 0..6 {
                assert!((out[t].0 - bf[t].0).abs() < 1e-9);
            }
        }
    }
}
