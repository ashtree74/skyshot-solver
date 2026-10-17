//! Spherical utilities: unit vectors, an equal-angle cube-face grid with CSR cell lists, and
//! robust cap queries that cross face boundaries by sampling the cap in its tangent plane.
//!
//! Cell id = face * n^2 + j * n + i, where (i, j) index the equal-angle coordinates
//! (alpha, beta) = (atan(u), atan(v)) in [-pi/4, pi/4] of the face's gnomonic plane.
//! The smallest cell width is about 0.707 * (pi/2) / n radians (mid-edge, along the edge).

use std::f64::consts::FRAC_PI_4;

pub type Vec3 = [f64; 3];

#[inline]
pub fn unit(ra_deg: f64, dec_deg: f64) -> Vec3 {
    let (r, d) = (ra_deg.to_radians(), dec_deg.to_radians());
    let cd = d.cos();
    [cd * r.cos(), cd * r.sin(), d.sin()]
}

#[inline]
pub fn radec(v: Vec3) -> (f64, f64) {
    let ra = v[1].atan2(v[0]).to_degrees().rem_euclid(360.0);
    let dec = v[2].atan2(v[0].hypot(v[1])).to_degrees();
    (ra, dec)
}

#[inline]
pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}

#[inline]
pub fn normalize(a: Vec3) -> Vec3 {
    let n = norm(a);
    [a[0] / n, a[1] / n, a[2] / n]
}

/// Angle between unit vectors (radians), numerically stable for small and large angles.
#[inline]
pub fn angle(a: Vec3, b: Vec3) -> f64 {
    norm(cross(a, b)).atan2(dot(a, b))
}

/// Chord length for an angle (radians).
#[inline]
pub fn chord(angle: f64) -> f64 {
    2.0 * (angle / 2.0).sin()
}

/// Orthonormal tangent basis (east, north) at a unit vector. At the poles east is +y.
#[inline]
pub fn tangent_basis(c: Vec3) -> (Vec3, Vec3) {
    let e = [-c[1], c[0], 0.0];
    let n = norm(e);
    let east = if n < 1e-12 {
        [0.0, 1.0, 0.0]
    } else {
        [e[0] / n, e[1] / n, 0.0]
    };
    let north = cross(c, east);
    (east, north)
}

/// Gnomonic projection of `p` onto the tangent plane at `c` in the (east, north) basis, radians.
#[inline]
pub fn gnomonic(p: Vec3, c: Vec3, east: Vec3, north: Vec3) -> Option<[f64; 2]> {
    let d = dot(p, c);
    if d <= 1e-12 {
        return None;
    }
    Some([dot(p, east) / d, dot(p, north) / d])
}

#[inline]
pub fn ungnomonic(xy: [f64; 2], c: Vec3, east: Vec3, north: Vec3) -> Vec3 {
    normalize([
        c[0] + xy[0] * east[0] + xy[1] * north[0],
        c[1] + xy[0] * east[1] + xy[1] * north[1],
        c[2] + xy[0] * east[2] + xy[1] * north[2],
    ])
}

/// Stereographic (conformal) projection of `p` about the centre `c` in the (east, north) basis:
/// `rho = 2 tan(theta / 2)` at angle `theta` from `c`, so a unit vector 90 deg away lands at
/// radius 2. Shares the basis convention of [`gnomonic`]; the two agree to second order near `c`
/// and the local scale is isotropic everywhere (`sec^2(theta / 2)`), which is what the phone's
/// virtual plane relies on (`camera::VirtualPlane`). `None` only at the antipode of `c`
/// (`dot(p, c) <= -1 + 1e-12`); the hemisphere behind `c` is projected (outside radius 2).
#[inline]
pub fn stereographic(p: Vec3, c: Vec3, east: Vec3, north: Vec3) -> Option<[f64; 2]> {
    let d = 1.0 + dot(p, c);
    if d <= 1e-12 {
        return None;
    }
    Some([2.0 * dot(p, east) / d, 2.0 * dot(p, north) / d])
}

/// Inverse of [`stereographic`]: `xy` in the (east, north) basis back to a unit vector.
/// With `t2 = |xy|^2 / 4`: `v = ((1 - t2) c + x east + y north) / (1 + t2)` (exactly unit).
#[inline]
pub fn unstereographic(xy: [f64; 2], c: Vec3, east: Vec3, north: Vec3) -> Vec3 {
    let t2 = (xy[0] * xy[0] + xy[1] * xy[1]) / 4.0;
    let a = 1.0 - t2;
    let s = 1.0 / (1.0 + t2);
    [
        (a * c[0] + xy[0] * east[0] + xy[1] * north[0]) * s,
        (a * c[1] + xy[0] * east[1] + xy[1] * north[1]) * s,
        (a * c[2] + xy[0] * east[2] + xy[1] * north[2]) * s,
    ]
}

/// Face and equal-angle coordinates (alpha, beta) in [-pi/4, pi/4].
#[inline]
pub fn face_coords(v: Vec3) -> (u32, f64, f64) {
    let (ax, ay, az) = (v[0].abs(), v[1].abs(), v[2].abs());
    let (face, u, w) = if ax >= ay && ax >= az {
        if v[0] > 0.0 {
            (0, v[1] / ax, v[2] / ax)
        } else {
            (1, -v[1] / ax, v[2] / ax)
        }
    } else if ay >= az {
        if v[1] > 0.0 {
            (2, -v[0] / ay, v[2] / ay)
        } else {
            (3, v[0] / ay, v[2] / ay)
        }
    } else if v[2] > 0.0 {
        (4, v[1] / az, -v[0] / az)
    } else {
        (5, v[1] / az, v[0] / az)
    };
    (face, u.atan(), w.atan())
}

/// Inverse of [`face_coords`].
pub fn face_vector(face: u32, alpha: f64, beta: f64) -> Vec3 {
    let (u, w) = (alpha.tan(), beta.tan());
    let v = match face {
        0 => [1.0, u, w],
        1 => [-1.0, -u, w],
        2 => [-u, 1.0, w],
        3 => [u, -1.0, w],
        4 => [-w, u, 1.0],
        _ => [w, u, -1.0],
    };
    normalize(v)
}

#[derive(Clone, Copy, Debug)]
pub struct CubeGrid {
    pub n: u32,
}

impl CubeGrid {
    pub fn new(n: u32) -> Self {
        assert!(n >= 1 && (n as u64) * (n as u64) * 6 < u32::MAX as u64);
        Self { n }
    }
    /// Choose the finest power-of-two n whose smallest cell width is >= `min_width` radians.
    pub fn with_min_width(min_width: f64) -> Self {
        let mut n = 1u32;
        while n < 16384 && Self::new(n * 2).min_cell_width() >= min_width {
            n *= 2;
        }
        Self::new(n)
    }
    #[inline]
    pub fn cells(&self) -> usize {
        6 * self.n as usize * self.n as usize
    }
    /// Lower bound of the angular width of any cell (radians).
    #[inline]
    pub fn min_cell_width(&self) -> f64 {
        0.70 * (std::f64::consts::FRAC_PI_2 / self.n as f64)
    }
    /// Upper bound of the angular diameter of any cell (radians).
    #[inline]
    pub fn max_cell_diameter(&self) -> f64 {
        1.5 * (std::f64::consts::FRAC_PI_2 / self.n as f64)
    }
    #[inline]
    pub fn cell(&self, v: Vec3) -> u32 {
        let (f, a, b) = face_coords(v);
        self.cell_of(f, a, b)
    }
    #[inline]
    pub fn cell_of(&self, face: u32, alpha: f64, beta: f64) -> u32 {
        let n = self.n as f64;
        let i = (((alpha + FRAC_PI_4) / std::f64::consts::FRAC_PI_2) * n).floor();
        let j = (((beta + FRAC_PI_4) / std::f64::consts::FRAC_PI_2) * n).floor();
        let i = (i.max(0.0) as u32).min(self.n - 1);
        let j = (j.max(0.0) as u32).min(self.n - 1);
        face * self.n * self.n + j * self.n + i
    }
    /// (face, i, j) of a cell.
    #[inline]
    pub fn split(&self, cell: u32) -> (u32, u32, u32) {
        let nn = self.n * self.n;
        (cell / nn, cell % self.n, (cell % nn) / self.n)
    }
    /// Centre vector of a cell.
    pub fn center(&self, cell: u32) -> Vec3 {
        let (f, i, j) = self.split(cell);
        let step = std::f64::consts::FRAC_PI_2 / self.n as f64;
        face_vector(
            f,
            -FRAC_PI_4 + (i as f64 + 0.5) * step,
            -FRAC_PI_4 + (j as f64 + 0.5) * step,
        )
    }
    /// All cells that may intersect the cap of angular radius `radius` around `c`, appended to
    /// `out` (deduplicated, unsorted). Uses an in-face block when the cap is far from face edges
    /// and tangent-plane sampling otherwise; both are conservative (may include extra cells).
    pub fn cap_cells(&self, c: Vec3, radius: f64, out: &mut Vec<u32>) {
        out.clear();
        let step = std::f64::consts::FRAC_PI_2 / self.n as f64;
        let (f, a, b) = face_coords(c);
        // Equal-angle stretch is >= 0.70, so `radius` covers at most radius/(0.70*step) cells.
        let reach = (radius / (0.70 * step)).ceil() as i64 + 1;
        let ci = ((a + FRAC_PI_4) / step).floor() as i64;
        let cj = ((b + FRAC_PI_4) / step).floor() as i64;
        let n = self.n as i64;
        if ci - reach >= 0 && cj - reach >= 0 && ci + reach < n && cj + reach < n && radius < 0.3 {
            for j in cj - reach..=cj + reach {
                for i in ci - reach..=ci + reach {
                    out.push(f * self.n * self.n + j as u32 * self.n + i as u32);
                }
            }
            return;
        }
        // Sample the cap enlarged by one cell diameter on a lattice finer than the narrowest cell.
        let r = (radius + self.max_cell_diameter()).min(std::f64::consts::FRAC_PI_2 * 0.98);
        let h = self.min_cell_width() / std::f64::consts::SQRT_2;
        let (east, north) = tangent_basis(c);
        let extent = r.tan();
        let k = (extent / h.tan().min(h)).ceil() as i64;
        let k = k.min(4000);
        let dh = extent / k.max(1) as f64;
        for y in -k..=k {
            for x in -k..=k {
                let (px, py) = (x as f64 * dh, y as f64 * dh);
                if px * px + py * py > extent * extent * 1.0001 {
                    continue;
                }
                let v = ungnomonic([px, py], c, east, north);
                out.push(self.cell(v));
            }
        }
        out.sort_unstable();
        out.dedup();
        let _ = f;
    }
}

/// CSR index of points by cube cell, with items inside each cell ordered by an external key.
pub struct CellIndex {
    pub grid: CubeGrid,
    pub offsets: Vec<u32>,
    pub items: Vec<u32>,
}

impl CellIndex {
    /// Build from per-point cells. `items` inside a cell keep the order of `order` (typically
    /// brightness rank), so a prefix of each cell list is "the brightest points of that cell".
    pub fn build(grid: CubeGrid, cells: &[u32], order: impl Iterator<Item = u32>) -> Self {
        let ncell = grid.cells();
        let mut counts = vec![0u32; ncell + 1];
        for &c in cells {
            counts[c as usize + 1] += 1;
        }
        for i in 0..ncell {
            counts[i + 1] += counts[i];
        }
        let offsets = counts.clone();
        let mut fill = counts;
        let mut items = vec![0u32; cells.len()];
        for p in order {
            let c = cells[p as usize] as usize;
            items[fill[c] as usize] = p;
            fill[c] += 1;
        }
        Self {
            grid,
            offsets,
            items,
        }
    }
    #[inline]
    pub fn cell_items(&self, cell: u32) -> &[u32] {
        let (a, b) = (
            self.offsets[cell as usize] as usize,
            self.offsets[cell as usize + 1] as usize,
        );
        &self.items[a..b]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_roundtrip() {
        for k in 0..2000 {
            let ra = (k as f64 * 37.13) % 360.0;
            let dec = ((k as f64 * 0.61803).fract() * 2.0 - 1.0)
                .asin()
                .to_degrees();
            let v = unit(ra, dec);
            let (f, a, b) = face_coords(v);
            let w = face_vector(f, a, b);
            assert!(angle(v, w) < 1e-12, "{ra} {dec}");
            let (r2, d2) = radec(v);
            assert!(angle(unit(r2, d2), v) < 1e-12);
        }
    }

    #[test]
    fn cap_cells_cover_brute_force() {
        let g = CubeGrid::new(64);
        let mut out = vec![];
        for k in 0..400 {
            let ra = (k as f64 * 77.7) % 360.0;
            let dec = ((k as f64 * 0.414).fract() * 2.0 - 1.0).asin().to_degrees();
            let c = unit(ra, dec);
            let radius = [0.002, 0.01, 0.05, 0.2][k % 4];
            g.cap_cells(c, radius, &mut out);
            // every random point inside the cap must fall into a listed cell
            for t in 0..200 {
                let (e, n) = tangent_basis(c);
                let ang = t as f64 * 2.399;
                let rr = radius * ((t as f64 + 0.5) / 200.0).sqrt();
                let p = ungnomonic([rr.tan() * ang.cos(), rr.tan() * ang.sin()], c, e, n);
                if angle(p, c) <= radius {
                    let cell = g.cell(p);
                    assert!(
                        out.contains(&cell),
                        "missed cell near ra={ra} dec={dec} r={radius}"
                    );
                }
            }
        }
    }

    #[test]
    fn min_cell_width_is_a_lower_bound() {
        let g = CubeGrid::new(32);
        let step = std::f64::consts::FRAC_PI_2 / 32.0;
        let mut min_w = f64::MAX;
        for f in 0..6 {
            for j in 0..32 {
                for i in 0..32 {
                    let a0 = -FRAC_PI_4 + i as f64 * step;
                    let b0 = -FRAC_PI_4 + j as f64 * step;
                    let bm = b0 + 0.5 * step;
                    let am = a0 + 0.5 * step;
                    let w1 = angle(face_vector(f, a0, bm), face_vector(f, a0 + step, bm));
                    let w2 = angle(face_vector(f, am, b0), face_vector(f, am, b0 + step));
                    min_w = min_w.min(w1).min(w2);
                }
            }
        }
        assert!(
            g.min_cell_width() <= min_w,
            "{} > {}",
            g.min_cell_width(),
            min_w
        );
    }
}

#[cfg(test)]
mod stereographic_tests {
    use super::*;

    fn pseudo_dir(k: usize) -> Vec3 {
        let ra = (k as f64 * 53.17) % 360.0;
        let dec = ((k as f64 * 0.7548).fract() * 2.0 - 1.0)
            .asin()
            .to_degrees();
        unit(ra, dec)
    }

    #[test]
    fn stereographic_roundtrip_and_known_values() {
        for k in 0..500 {
            let c = pseudo_dir(k);
            let (e, n) = tangent_basis(c);
            // points up to 150 deg from c (well into the back hemisphere)
            let th = ((k as f64 * 0.3271).fract()) * 150.0_f64.to_radians();
            let ph = (k as f64 * 1.731) % std::f64::consts::TAU;
            let p = normalize([
                c[0] * th.cos() + (e[0] * ph.cos() + n[0] * ph.sin()) * th.sin(),
                c[1] * th.cos() + (e[1] * ph.cos() + n[1] * ph.sin()) * th.sin(),
                c[2] * th.cos() + (e[2] * ph.cos() + n[2] * ph.sin()) * th.sin(),
            ]);
            let xy = stereographic(p, c, e, n).expect("not antipodal");
            let rho = xy[0].hypot(xy[1]);
            assert!(
                (rho - 2.0 * (th / 2.0).tan()).abs() < 1e-12,
                "rho {rho} vs 2tan(theta/2) at theta {th}"
            );
            let q = unstereographic(xy, c, e, n);
            assert!((norm(q) - 1.0).abs() < 1e-15, "unit length");
            assert!(angle(p, q) < 1e-12, "round trip {k}: {}", angle(p, q));
        }
        // 90 deg from the centre -> radius exactly 2 along the basis vectors
        let c = unit(10.0, 20.0);
        let (e, n) = tangent_basis(c);
        let xe = stereographic(e, c, e, n).unwrap();
        let xn = stereographic(n, c, e, n).unwrap();
        assert!((xe[0] - 2.0).abs() < 1e-14 && xe[1].abs() < 1e-14);
        assert!(xn[0].abs() < 1e-14 && (xn[1] - 2.0).abs() < 1e-14);
        // the centre maps to the origin; the antipode is rejected
        let o = stereographic(c, c, e, n).unwrap();
        assert!(o[0].abs() < 1e-15 && o[1].abs() < 1e-15);
        assert!(stereographic([-c[0], -c[1], -c[2]], c, e, n).is_none());
    }

    #[test]
    fn stereographic_matches_gnomonic_to_second_order() {
        // gnomonic radius tan(theta), stereographic 2 tan(theta/2): both ~ theta + O(theta^3)
        let c = unit(200.0, -35.0);
        let (e, n) = tangent_basis(c);
        for k in 1..50 {
            let th = k as f64 * 1e-3;
            let p = normalize([
                c[0] * th.cos() + e[0] * th.sin(),
                c[1] * th.cos() + e[1] * th.sin(),
                c[2] * th.cos() + e[2] * th.sin(),
            ]);
            let g = gnomonic(p, c, e, n).unwrap();
            let s = stereographic(p, c, e, n).unwrap();
            // tan(t) - 2tan(t/2) = t^3/4 + (1/8) t^5 + O(t^7)
            let expect = th.powi(3) / 4.0;
            let next = th.powi(5) / 8.0;
            assert!(
                ((g[0] - s[0]) - expect).abs() < 2.0 * next + 1e-16,
                "theta {th}"
            );
            assert!(g[1].abs() < 1e-15 && s[1].abs() < 1e-15);
        }
    }
}
