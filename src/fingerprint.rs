//! Per-star fingerprint (key spec v1). Bit-exact port of `lab/skyanchor_lab/fingerprint.py`.
//!
//! A key describes an anchor A, a reference neighbour R and three further neighbours in the
//! similarity-invariant basis (origin A, x-axis A->R, unit |AR|). For neighbour j:
//! `rho_j = ln(|p_j - A| / |AR|)`, `theta_j = atan2(y_j, x_j)`. Neighbours are sorted by rho;
//! the parity canonical form makes theta of the first neighbour lie in [0, pi] (flip = 1 when
//! all angles had to be negated). Quantisation into 32 bits:
//! `key = r1 | r2<<5 | r3<<10 | t1<<15 | t2<<20 | t3<<26`.

use std::f64::consts::{PI, TAU};

pub const RHO_STEP: f64 = 0.12;
pub const RHO_BINS: u32 = 20;
pub const THETA_BINS: u32 = 60;
pub const THETA1_BINS: u32 = THETA_BINS / 2;
pub const THETA_STEP: f64 = TAU / THETA_BINS as f64;
pub const RHO_TOL_FLOOR: f64 = 0.03;
pub const THETA_TOL_FLOOR: f64 = 1.5 * PI / 180.0;
pub const MAX_VARIANTS: usize = 96;

pub type P2 = [f64; 2];

#[inline]
fn rho_bin(r: f64) -> u32 {
    ((r / RHO_STEP).floor().max(0.0) as u32).min(RHO_BINS - 1)
}
#[inline]
fn t1_bin(t: f64) -> u32 {
    ((t / THETA_STEP).floor().max(0.0) as u32).min(THETA1_BINS - 1)
}
#[inline]
fn t_bin(t: f64) -> u32 {
    ((t.rem_euclid(TAU) / THETA_STEP).floor() as u32) % THETA_BINS
}

#[inline]
pub fn pack(b: [u32; 6]) -> u32 {
    b[0] | (b[1] << 5) | (b[2] << 10) | (b[3] << 15) | (b[4] << 20) | (b[5] << 26)
}

#[inline]
pub fn unpack(k: u32) -> [u32; 6] {
    [
        k & 31,
        (k >> 5) & 31,
        (k >> 10) & 31,
        (k >> 15) & 31,
        (k >> 20) & 63,
        (k >> 26) & 63,
    ]
}

/// (rho, theta) of three neighbours in the basis (a, r), and |AR|.
#[inline]
pub fn basis_coords(a: P2, r: P2, nbrs: [P2; 3]) -> Option<([(f64, f64); 3], f64)> {
    let (ux, uy) = (r[0] - a[0], r[1] - a[1]);
    let len = ux.hypot(uy);
    if len <= 0.0 || !len.is_finite() {
        return None;
    }
    let ex = (ux / len, uy / len);
    let ey = (-ex.1, ex.0);
    let mut out = [(0.0, 0.0); 3];
    for (k, p) in nbrs.iter().enumerate() {
        let (vx, vy) = (p[0] - a[0], p[1] - a[1]);
        let x = (vx * ex.0 + vy * ex.1) / len;
        let y = (vx * ey.0 + vy * ey.1) / len;
        let h = x.hypot(y);
        if h <= 0.0 {
            return None;
        }
        out[k] = (h.ln(), y.atan2(x));
    }
    Some((out, len))
}

/// Canonical catalogue key: (key, flip, |AR|).
pub fn canonical_key(a: P2, r: P2, nbrs: [P2; 3]) -> Option<(u32, bool, f64)> {
    let (mut c, len) = basis_coords(a, r, nbrs)?;
    // stable sort by rho (matches Python's list.sort with key)
    c.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    let flip = c[0].1 < 0.0;
    if flip {
        for v in &mut c {
            v.1 = -v.1;
        }
    }
    let key = pack([
        rho_bin(c[0].0),
        rho_bin(c[1].0),
        rho_bin(c[2].0),
        t1_bin(c[0].1),
        t_bin(c[1].1),
        t_bin(c[2].1),
    ]);
    Some((key, flip, len))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct KeyVariant {
    pub key: u32,
    pub flip: bool,
}

#[inline]
fn push_bins_rho(v: f64, tol: f64, out: &mut [u32; 3]) -> usize {
    let b = rho_bin(v);
    let mut n = 0;
    out[n] = b;
    n += 1;
    let lo = b as f64 * RHO_STEP;
    let hi = (b + 1) as f64 * RHO_STEP;
    if v - lo < tol && b > 0 {
        out[n] = b - 1;
        n += 1;
    }
    if hi - v < tol && b < RHO_BINS - 1 {
        out[n] = b + 1;
        n += 1;
    }
    n
}
#[inline]
fn push_bins_t1(v: f64, tol: f64, out: &mut [u32; 3]) -> usize {
    let b = t1_bin(v);
    let mut n = 0;
    out[n] = b;
    n += 1;
    let lo = b as f64 * THETA_STEP;
    let hi = (b + 1) as f64 * THETA_STEP;
    if v - lo < tol && b > 0 {
        out[n] = b - 1;
        n += 1;
    }
    if hi - v < tol && b < THETA1_BINS - 1 {
        out[n] = b + 1;
        n += 1;
    }
    n
}
#[inline]
fn push_bins_t(v: f64, tol: f64, out: &mut [u32; 3]) -> usize {
    let v = v.rem_euclid(TAU);
    let b = t_bin(v);
    let mut n = 0;
    out[n] = b;
    n += 1;
    let lo = b as f64 * THETA_STEP;
    let hi = (b + 1) as f64 * THETA_STEP;
    if v - lo < tol {
        out[n] = (b + THETA_BINS - 1) % THETA_BINS;
        n += 1;
    }
    if hi - v < tol {
        out[n] = (b + 1) % THETA_BINS;
        n += 1;
    }
    n
}

const PERMS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

/// The gate of [`image_keys`] without enumerating the variants: |AR| in pixels, or None exactly
/// when [`image_keys`] returns None (used by the learned key with a fixed probe count).
pub fn image_gate(a: P2, r: P2, nbrs: [P2; 3], sigma_px: f64) -> Option<f64> {
    let (c, len) = basis_coords(a, r, nbrs)?;
    let tr = RHO_TOL_FLOOR.max(2.8 * sigma_px / len);
    if c.iter().any(|v| v.0 < -tr) {
        return None;
    }
    Some(len)
}

/// All key variants an image configuration may correspond to (sorted, deduplicated), appended to
/// `out` (cleared first). Returns |AR| in pixels, or None if no catalogue record can match.
pub fn image_keys(
    a: P2,
    r: P2,
    nbrs: [P2; 3],
    sigma_px: f64,
    out: &mut Vec<KeyVariant>,
) -> Option<f64> {
    out.clear();
    let (c, len) = basis_coords(a, r, nbrs)?;
    let tr = RHO_TOL_FLOOR.max(2.8 * sigma_px / len);
    let tt = THETA_TOL_FLOOR.max(1.6 * sigma_px / len);
    if c.iter().any(|v| v.0 < -tr) {
        return None;
    }
    for p in PERMS {
        let cs = [c[p[0]], c[p[1]], c[p[2]]];
        if cs[0].0 > cs[1].0 + 2.0 * tr
            || cs[0].0 > cs[2].0 + 2.0 * tr
            || cs[1].0 > cs[2].0 + 2.0 * tr
        {
            continue;
        }
        let tf = cs[0].1;
        let ambiguous = tf.abs() < tt || tf.abs() > PI - tt;
        let flips: &[bool] = if ambiguous {
            &[false, true]
        } else if tf < 0.0 {
            &[true]
        } else {
            &[false]
        };
        for &flip in flips {
            let s = if flip { -1.0 } else { 1.0 };
            let t1 = (cs[0].1 * s).clamp(0.0, PI);
            let mut b = [[0u32; 3]; 6];
            let mut n = [0usize; 6];
            n[0] = push_bins_rho(cs[0].0, tr, &mut b[0]);
            n[1] = push_bins_rho(cs[1].0, tr, &mut b[1]);
            n[2] = push_bins_rho(cs[2].0, tr, &mut b[2]);
            n[3] = push_bins_t1(t1, tt, &mut b[3]);
            n[4] = push_bins_t(cs[1].1 * s, tt, &mut b[4]);
            n[5] = push_bins_t(cs[2].1 * s, tt, &mut b[5]);
            for i0 in 0..n[0] {
                for i1 in 0..n[1] {
                    for i2 in 0..n[2] {
                        for i3 in 0..n[3] {
                            for i4 in 0..n[4] {
                                for i5 in 0..n[5] {
                                    out.push(KeyVariant {
                                        key: pack([
                                            b[0][i0], b[1][i1], b[2][i2], b[3][i3], b[4][i4],
                                            b[5][i5],
                                        ]),
                                        flip,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out.truncate(MAX_VARIANTS);
    Some(len)
}

/// Bijective 32-bit mixer (murmur3 fmix32). bucket = h >> 12 (20 bits), fp = h & 0xfff.
#[inline]
pub fn mix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rot(p: P2, ang: f64, s: f64, t: P2, mirror: bool) -> P2 {
        let y = if mirror { -p[1] } else { p[1] };
        let (c, sn) = (ang.cos(), ang.sin());
        [
            s * (c * p[0] - sn * y) + t[0],
            s * (sn * p[0] + c * y) + t[1],
        ]
    }

    #[test]
    fn canonical_key_is_similarity_invariant_and_flip_tracks_mirror() {
        let a = [0.0, 0.0];
        let r = [1.0, 0.0];
        let n = [[1.31, 0.77], [-0.9, 1.7], [0.4, -2.1]];
        let (k0, f0, _) = canonical_key(a, r, n).unwrap();
        for (i, &(ang, s, mirror)) in [(0.3, 2.0, false), (2.9, 0.01, true), (-1.2, 340.0, true)]
            .iter()
            .enumerate()
        {
            let t = [10.0 * i as f64, -3.0];
            let (k, f, len) = canonical_key(
                rot(a, ang, s, t, mirror),
                rot(r, ang, s, t, mirror),
                n.map(|p| rot(p, ang, s, t, mirror)),
            )
            .unwrap();
            assert_eq!(k, k0);
            assert_eq!(f, f0 ^ mirror);
            assert!((len - s).abs() < 1e-9 * s.max(1.0));
        }
    }

    #[test]
    fn image_variants_contain_the_canonical_key_under_noise() {
        let mut out = vec![];
        let mut misses = 0;
        let mut total = 0;
        let mut seed = 12345u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..3000 {
            let a = [500.0, 500.0];
            let ang = rnd() * TAU;
            let r = [a[0] + 60.0 * ang.cos(), a[1] + 60.0 * ang.sin()];
            let nb: [P2; 3] = std::array::from_fn(|_| {
                let d = 60.0 * (1.0 + 3.0 * rnd());
                let t = rnd() * TAU;
                [a[0] + d * t.cos(), a[1] + d * t.sin()]
            });
            let Some((k, f, _)) = canonical_key(a, r, nb) else {
                continue;
            };
            let noisy =
                |p: P2, r1: f64, r2: f64| [p[0] + (r1 - 0.5) * 0.6, p[1] + (r2 - 0.5) * 0.6];
            let (x1, x2, x3, x4) = (rnd(), rnd(), rnd(), rnd());
            let na = noisy(a, x1, x2);
            let nr = noisy(r, x3, x4);
            let nn = nb.map(|p| noisy(p, rnd(), rnd()));
            total += 1;
            if image_keys(na, nr, nn, 0.3, &mut out).is_none()
                || !out.contains(&KeyVariant { key: k, flip: f })
            {
                misses += 1;
            }
        }
        assert!(misses * 100 < total, "{misses}/{total}");
    }

    #[test]
    fn mix32_is_bijective_on_a_sample() {
        let mut seen = std::collections::HashSet::new();
        for k in 0..200_000u32 {
            assert!(seen.insert(mix32(k.wrapping_mul(2654435761))));
        }
    }
}
