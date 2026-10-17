//! Learned 32-bit key LGQ-GD (spec: `docs/learned/FEATURES.md`; reference implementation:
//! `lab/skyanchor_lab/learned/{features,lgq,probes}.py`).
//!
//! A configuration (anchor A at the origin, reference R, three key neighbours S1..S3 and the
//! dominance distances of all five) is featurised in f64, embedded by a small DeepSets network in
//! f32, mapped through a per-dimension empirical CDF (piecewise-linear quantile table) to 8
//! equal-mass bins of 4/3/3/3/4/5/5/5 bits and Gray-coded into the 32-bit key. The record payload
//! bit `flip` is the parity canonicalisation (sign of the parity statistic S).
//!
//! Image side: a noise head predicts the CDF noise of every dimension; per-bit log-odds give the
//! probability that the catalogue bit differs, and the K most probable keys (cheapest flip sets
//! over the 12 least confident bits) are probed. Configurations near the parity boundary also
//! enumerate the opposite-parity branch with a prior cost.
//!
//! Determinism: the network uses exactly one summation order per output (sum over inputs in
//! index order, then + bias) for every batch size, so the builder and the solver compute
//! bit-identical embeddings for identical inputs.

use crate::fingerprint::KeyVariant;
use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256};
use std::{f64::consts::PI, path::Path};

/// Manifest `key_type` of an index built with this coder.
pub const KEY_TYPE: &str = "lgq-gd";
/// File name prefix of the coder inside an index directory (`coder.f32`, `coder.json`).
pub const INDEX_CODER: &str = "coder";
pub const N_Q: usize = 4097;
pub const SIGMA_FLOOR: f32 = 0.002;
/// Least confident bits enumerated by the multiprobe.
pub const M_BITS: usize = 12;
/// Opposite-parity branch only when its prior cost is below this (nats).
pub const PRIOR_CUT: f32 = 12.0;
/// Parity calibration (FEATURES.md section 6): t = |S| / sqrt((c sigma_S)^2 + s0^2).
pub const PARITY_C: f64 = 1.0;
pub const PARITY_S0: f64 = 0.005;

const FE: usize = 16;
const FG: usize = 2;
const FN: usize = 3;
const H1: usize = 64;
const H2: usize = 128;
const DIMS: usize = 8;
const TIN: usize = H1 + FG;
const NIN: usize = H2 + FN;

const LAYOUT: [(&str, usize, usize); 12] = [
    ("phi1.weight", H1, FE),
    ("phi1.bias", H1, 1),
    ("phi2.weight", H1, H1),
    ("phi2.bias", H1, 1),
    ("t1.weight", H2, TIN),
    ("t1.bias", H2, 1),
    ("t2.weight", H2, H2),
    ("t2.bias", H2, 1),
    ("emb.weight", DIMS, H2),
    ("emb.bias", DIMS, 1),
    ("noise.weight", DIMS, NIN),
    ("noise.bias", DIMS, 1),
];

/// Raw configuration, anchor at the origin. Image side: pixels; catalogue side: gnomonic
/// coordinates at A (radians) and dominance distances in radians.
#[derive(Clone, Copy, Debug)]
pub struct Raw {
    /// R, S1, S2, S3 relative to A
    pub xy: [[f64; 2]; 4],
    /// dominance distance of A, R, S1, S2, S3 (+inf allowed)
    pub d: [f64; 5],
    /// image only: max clipped centroid sigma of the five stars (catalogue: 0)
    pub sigma: f64,
    /// image only: distance of A to the nearest frame edge (catalogue: +inf)
    pub edge: f64,
}

/// Basis coordinates before the parity canonicalisation.
#[derive(Clone, Copy, Debug)]
pub struct Basis {
    pub len: f64,
    pub u: [f64; 3],
    pub v: [f64; 3],
    /// parity statistic S (sign decides the canonical frame)
    pub spar: f64,
}

/// Network inputs (f32) of one configuration plus the parity information.
#[derive(Clone, Copy, Debug)]
pub struct Feats {
    pub elem: [[f32; FE]; 3],
    pub glob: [f32; FG],
    pub noise: [f32; FN],
    pub flip: bool,
    pub spar: f64,
    pub spar_sigma: f64,
    pub len: f64,
}

#[inline]
fn nan0(x: f64) -> f64 {
    if x.is_nan() { 0.0 } else { x }
}

/// u, v of S1..S3 in the (A, R) basis (unit |AR|) and the parity statistic; None if invalid.
pub fn basis(raw: &Raw) -> Option<Basis> {
    let r = raw.xy[0];
    let len = r[0].hypot(r[1]);
    if len.is_nan() || len <= 0.0 {
        return None;
    }
    let ex = [r[0] / len, r[1] / len];
    let ey = [-ex[1], ex[0]];
    let mut u = [0.0; 3];
    let mut v = [0.0; 3];
    for j in 0..3 {
        let p = raw.xy[j + 1];
        u[j] = (p[0] * ex[0] + p[1] * ex[1]) / len;
        v[j] = (p[0] * ey[0] + p[1] * ey[1]) / len;
        let r2 = u[j] * u[j] + v[j] * v[j];
        if !(r2 > 0.0 && r2.is_finite()) {
            return None;
        }
    }
    let mut spar = -0.0;
    for j in 0..3 {
        spar += v[j] / (u[j] * u[j] + v[j] * v[j]);
    }
    Some(Basis { len, u, v, spar })
}

/// Element features of one key neighbour after the parity flip (`v` already signed).
#[inline]
pub fn elem_features(u: f64, v: f64, d: f64, len: f64) -> [f32; FE] {
    let rho = 0.5 * (u * u + v * v).ln();
    let th = v.atan2(u);
    let mut f = [0.0f64; FE];
    f[0] = rho;
    for k in 1..=4 {
        let a = k as f64 * th;
        f[2 * k - 1] = a.cos();
        f[2 * k] = a.sin();
    }
    // sin/cos(2 pi rho / T), T = 2, 1, 0.5
    for (i, w) in [PI, 2.0 * PI, 4.0 * PI].into_iter().enumerate() {
        f[9 + 2 * i] = (w * rho).sin();
        f[10 + 2 * i] = (w * rho).cos();
    }
    f[15] = (d / len).ln().clamp(-2.0, 3.0);
    f.map(|x| nan0(x) as f32)
}

/// Featurisation (FEATURES.md sections 2-3). `invert` builds the opposite-parity branch.
pub fn featurize(raw: &Raw, invert: bool) -> Option<Feats> {
    let b = basis(raw)?;
    Some(featurize_basis(raw, &b, invert))
}

pub fn featurize_basis(raw: &Raw, b: &Basis, invert: bool) -> Feats {
    let flip = (b.spar < 0.0) ^ invert;
    let sgn = if flip { -1.0 } else { 1.0 };
    let mut f = featurize_basis_tail(raw, b, flip);
    f.elem = std::array::from_fn(|j| elem_features(b.u[j], b.v[j] * sgn, raw.d[j + 2], b.len));
    f
}

/// Global and noise columns and the parity fields (element features left zero).
fn featurize_basis_tail(raw: &Raw, b: &Basis, flip: bool) -> Feats {
    let len = b.len;
    let elem = [[0f32; FE]; 3];
    let glob = [
        nan0((raw.d[0] / len).ln().clamp(-2.0, 4.0)) as f32,
        nan0((raw.d[1] / len).ln().clamp(-2.0, 3.0)) as f32,
    ];
    let noise = [
        (raw.sigma.max(1e-300) / len).ln().clamp(-12.0, 0.0) as f32,
        if raw.d[0] >= raw.edge { 1.0 } else { 0.0 },
        (raw.edge / len).ln().clamp(-1.0, 4.0) as f32,
    ];
    Feats {
        elem,
        glob,
        noise,
        flip,
        spar: b.spar,
        spar_sigma: spar_sigma(b, raw.sigma),
        len,
    }
}

/// First-order 1-sigma of the parity statistic under isotropic centroid noise.
pub fn spar_sigma(b: &Basis, sigma: f64) -> f64 {
    let (us, vs) = (b.u, b.v);
    let mut gu = [0.0; 3];
    let mut gv = [0.0; 3];
    for j in 0..3 {
        let r2 = us[j] * us[j] + vs[j] * vs[j];
        gu[j] = -2.0 * us[j] * vs[j] / (r2 * r2);
        gv[j] = (us[j] * us[j] - vs[j] * vs[j]) / (r2 * r2);
    }
    let sum = |f: &dyn Fn(usize) -> f64| f(0) + (f(1) + f(2));
    let var = sum(&|j| gu[j] * gu[j] + gv[j] * gv[j])
        + sum(&|j| gu[j]).powi(2)
        + sum(&|j| gv[j]).powi(2)
        + sum(&|j| -gv[j] * us[j] + gu[j] * vs[j]).powi(2)
        + b.spar * b.spar;
    sigma / b.len * var.sqrt()
}

/// ln Phi(t) - ln Phi(-t) (cost of the opposite-parity branch), t >= 0.
pub fn parity_prior(spar: f64, spar_sigma: f64) -> f32 {
    let sig = ((PARITY_C * spar_sigma).powi(2) + PARITY_S0 * PARITY_S0).sqrt();
    let t = spar.abs() / sig;
    if !t.is_finite() {
        return f32::INFINITY;
    }
    // ln Q, Q = Phi(-t) = erfc(t / sqrt 2) / 2, in log space (no underflow for large t)
    let ln_q = ln_erfc(t / std::f64::consts::SQRT_2) - std::f64::consts::LN_2;
    ((-ln_q.exp()).ln_1p() - ln_q) as f32
}

/// ln erfc(x) for x >= 0 (series below 2, continued fraction above; relative error ~1e-14).
pub fn ln_erfc(x: f64) -> f64 {
    debug_assert!(x >= 0.0);
    if x < 2.0 {
        // erf(x) = 2/sqrt(pi) exp(-x^2) sum x^(2n+1) 2^n / (2n+1)!!  (all terms positive)
        let x2 = x * x;
        let mut term = x;
        let mut sum = x;
        let mut n = 0.0;
        while term > 1e-17 * sum {
            n += 1.0;
            term *= 2.0 * x2 / (2.0 * n + 1.0);
            sum += term;
        }
        (-(2.0 / PI.sqrt()) * (-x2).exp() * sum).ln_1p()
    } else {
        // erfc(x) = exp(-x^2)/sqrt(pi) / (x + (1/2)/(x + 1/(x + (3/2)/(x + ...))))
        let mut f = x;
        for n in (1..=80).rev() {
            f = x + (n as f64 / 2.0) / f;
        }
        -x * x - (PI.sqrt() * f).ln()
    }
}

/// Complementary error function for x >= 0.
pub fn erfc(x: f64) -> f64 {
    ln_erfc(x).exp()
}

#[inline]
fn gray(n: u32) -> u32 {
    n ^ (n >> 1)
}

/// y[b] = act(W x[b] + bias) for B configurations; W stored input-major ([IN][OUT]).
/// Register-blocked: B configurations x CH outputs of accumulators. Summation order per output:
/// inputs in index order, then + bias -- identical for every B and CH, so batched and single
/// evaluations (builder and solver) give bit-identical results.
#[inline(always)]
fn dense<const IN: usize, const OUT: usize, const B: usize, const CH: usize>(
    w: &[f32],
    bias: &[f32],
    x: &[[f32; IN]; B],
    y: &mut [[f32; OUT]; B],
    relu: bool,
) {
    let (rows, rest) = w.as_chunks::<OUT>();
    assert!(rows.len() == IN && rest.is_empty() && bias.len() == OUT && OUT.is_multiple_of(CH));
    for blk in 0..OUT / CH {
        let mut acc = [[0f32; CH]; B];
        for (i, row) in rows.iter().enumerate() {
            let wr: &[f32; CH] = &row.as_chunks::<CH>().0[blk];
            for bb in 0..B {
                let xv = x[bb][i];
                for q in 0..CH {
                    acc[bb][q] += wr[q] * xv;
                }
            }
        }
        let bc: &[f32; CH] = &bias.as_chunks::<CH>().0[blk];
        for bb in 0..B {
            let yc: &mut [f32; CH] = &mut y[bb].as_chunks_mut::<CH>().0[blk];
            for q in 0..CH {
                let v = acc[bb][q] + bc[q];
                yc[q] = if relu && (v.is_nan() || v <= 0.0) {
                    0.0
                } else {
                    v
                };
            }
        }
    }
}

/// The LGQ-GD coder (weights transposed to input-major for the forward pass).
pub struct Coder {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    w3: Vec<f32>,
    b3: Vec<f32>,
    w4: Vec<f32>,
    b4: Vec<f32>,
    we: Vec<f32>,
    be: Vec<f32>,
    ws: Vec<f32>,
    bs: Vec<f32>,
    /// quantile table [DIMS][N_Q] (f32 values widened to f64, as numpy.interp sees them)
    q: Vec<f32>,
    bits: [u32; DIMS],
    shift: [u32; DIMS],
    /// SHA-256 of the .f32 blob followed by the .json bytes
    pub sha256: String,
}

/// Memo of element features and phi outputs keyed by the exact element inputs
/// (|AR|, u, signed v, D). Clear it per anchor to keep lookups short.
#[derive(Default)]
pub struct ElemCache {
    keys: Vec<[u64; 4]>,
    vals: Vec<([f32; FE], [f32; H1])>,
}

impl ElemCache {
    pub fn clear(&mut self) {
        self.keys.clear();
        self.vals.clear();
    }
    pub fn len(&self) -> usize {
        self.keys.len()
    }
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

/// Per-configuration image-side output (for tests and diagnostics).
#[derive(Clone, Debug)]
pub struct ImageCode {
    pub feats: Feats,
    pub e: [f32; DIMS],
    pub sigma: [f32; DIMS],
    pub cdf: [f64; DIMS],
    pub key: u32,
    pub logodds: [f32; 32],
}

/// One probe: key, parity flip of the frame it was computed in, and its cost (nats).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Probe {
    pub key: u32,
    pub flip: bool,
    pub cost: f64,
}

impl Coder {
    /// Load `<prefix>.f32` and `<prefix>.json`.
    pub fn load(prefix: &Path) -> Result<Self> {
        let jp = prefix.with_extension("json");
        let bp = prefix.with_extension("f32");
        let jbytes = std::fs::read(&jp).with_context(|| format!("read {}", jp.display()))?;
        let blob = std::fs::read(&bp).with_context(|| format!("read {}", bp.display()))?;
        Self::from_bytes(&blob, &jbytes)
    }

    pub fn from_bytes(blob: &[u8], json: &[u8]) -> Result<Self> {
        let meta: serde_json::Value = serde_json::from_slice(json).context("coder json")?;
        ensure!(
            meta["format"] == "skyanchor2-learned-coder-lgq" && meta["version"] == 1,
            "unsupported coder format"
        );
        let a = &meta["arch"];
        let num = |k: &str| a[k].as_u64().unwrap_or(u64::MAX) as usize;
        ensure!(
            num("fe") == FE
                && num("fg") == FG
                && num("fn") == FN
                && num("h1") == H1
                && num("h2") == H2
                && num("dims") == DIMS
                && a["use_extra"] == false,
            "coder architecture is not LGQ-GD (fe 16, fg 2, h1 64, h2 128, dims 8)"
        );
        let cols = |k: &str| -> Vec<u64> {
            a[k].as_array()
                .map(|v| v.iter().filter_map(|x| x.as_u64()).collect())
                .unwrap_or_default()
        };
        ensure!(
            cols("elem_cols") == (0..16).collect::<Vec<u64>>() && cols("glob_cols") == [0, 1],
            "coder feature columns are not the GD set"
        );
        let bits: Vec<u32> = meta["bits_per_dim"]
            .as_array()
            .context("bits_per_dim")?
            .iter()
            .map(|x| x.as_u64().unwrap_or(0) as u32)
            .collect();
        ensure!(
            bits.len() == DIMS && bits.iter().sum::<u32>() == 32 && bits.iter().all(|&b| b >= 1),
            "bits_per_dim"
        );
        ensure!(blob.len().is_multiple_of(4), "coder blob size");
        let floats: Vec<f32> = blob
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let tensors = meta["tensors"].as_array().context("tensors")?;
        let get = |name: &str, rows: usize, cols: usize| -> Result<&[f32]> {
            let t = tensors
                .iter()
                .find(|t| t["name"] == name)
                .with_context(|| format!("tensor {name}"))?;
            let shape: Vec<usize> = t["shape"]
                .as_array()
                .context("shape")?
                .iter()
                .map(|x| x.as_u64().unwrap_or(0) as usize)
                .collect();
            let n: usize = shape.iter().product();
            ensure!(n == rows * cols, "tensor {name}: shape {shape:?}");
            let off = t["offset_floats"].as_u64().context("offset")? as usize;
            ensure!(off + n <= floats.len(), "tensor {name} out of range");
            Ok(&floats[off..off + n])
        };
        // PyTorch [out, in] row-major -> input-major [in][out]
        let tr = |name: &str, out: usize, inp: usize| -> Result<Vec<f32>> {
            let w = get(name, out, inp)?;
            let mut t = vec![0f32; out * inp];
            for o in 0..out {
                for i in 0..inp {
                    t[i * out + o] = w[o * inp + i];
                }
            }
            Ok(t)
        };
        let vecf = |name: &str, n: usize| -> Result<Vec<f32>> { Ok(get(name, n, 1)?.to_vec()) };
        let _ = LAYOUT;
        let q32 = get("quantiles", DIMS, N_Q)?;
        let q: Vec<f32> = q32.to_vec();
        for j in 0..DIMS {
            let row = &q[j * N_Q..(j + 1) * N_Q];
            ensure!(
                row.windows(2).all(|w| w[0] <= w[1]),
                "quantiles of dim {j} not sorted"
            );
        }
        let mut b = [0u32; DIMS];
        b.copy_from_slice(&bits);
        let mut shift = [0u32; DIMS];
        for j in 1..DIMS {
            shift[j] = shift[j - 1] + b[j - 1];
        }
        let mut h = Sha256::new();
        h.update(blob);
        h.update(json);
        Ok(Self {
            w1: tr("phi1.weight", H1, FE)?,
            b1: vecf("phi1.bias", H1)?,
            w2: tr("phi2.weight", H1, H1)?,
            b2: vecf("phi2.bias", H1)?,
            w3: tr("t1.weight", H2, TIN)?,
            b3: vecf("t1.bias", H2)?,
            w4: tr("t2.weight", H2, H2)?,
            b4: vecf("t2.bias", H2)?,
            we: tr("emb.weight", DIMS, H2)?,
            be: vecf("emb.bias", DIMS)?,
            ws: tr("noise.weight", DIMS, NIN)?,
            bs: vecf("noise.bias", DIMS)?,
            q,
            bits: b,
            shift,
            sha256: format!("{:x}", h.finalize()),
        })
    }

    /// Copy the coder files into `dir` as `coder.f32` / `coder.json` (self-contained index).
    pub fn copy_files(prefix: &Path, dir: &Path) -> Result<()> {
        for ext in ["f32", "json"] {
            let src = prefix.with_extension(ext);
            let dst = dir.join(format!("{INDEX_CODER}.{ext}"));
            // copying a file onto itself truncates it (std::fs::copy opens dst first)
            if let (Ok(a), Ok(b)) = (src.canonicalize(), dst.canonicalize())
                && a == b
            {
                continue;
            }
            std::fs::copy(&src, &dst)
                .with_context(|| format!("copy {} -> {}", src.display(), dst.display()))?;
        }
        Ok(())
    }

    pub fn bits(&self) -> [u32; DIMS] {
        self.bits
    }

    /// phi(e) = relu(W2 relu(W1 e + b1) + b2)
    #[inline]
    pub fn phi(&self, e: &[f32; FE]) -> [f32; H1] {
        let mut h = [[0f32; H1]; 1];
        dense::<FE, H1, 1, 64>(&self.w1, &self.b1, &[*e], &mut h, true);
        let mut o = [[0f32; H1]; 1];
        dense::<H1, H1, 1, 64>(&self.w2, &self.b2, &h, &mut o, true);
        o[0]
    }

    /// Trunk input [h1 + h2 + h3, glob] from the three element embeddings.
    #[inline]
    pub fn pool(h: [&[f32; H1]; 3], glob: &[f32; FG]) -> [f32; TIN] {
        let mut p = [0f32; TIN];
        for o in 0..H1 {
            p[o] = h[0][o] + h[1][o] + h[2][o];
        }
        p[H1..].copy_from_slice(glob);
        p
    }

    /// Trunk and heads for B configurations: embedding e and (if `noise` is given) the CDF noise.
    #[inline]
    /// CH: output register block (64 suits B = 1, 16 suits B = 4); results do not depend on it.
    pub fn heads<const B: usize, const CH: usize>(
        &self,
        p: &[[f32; TIN]; B],
        noise: Option<&[[f32; FN]; B]>,
        e: &mut [[f32; DIMS]; B],
        sig: &mut [[f32; DIMS]; B],
    ) {
        let mut x1 = [[0f32; H2]; B];
        dense::<TIN, H2, B, CH>(&self.w3, &self.b3, p, &mut x1, true);
        let mut x = [[0f32; H2]; B];
        dense::<H2, H2, B, CH>(&self.w4, &self.b4, &x1, &mut x, true);
        dense::<H2, DIMS, B, DIMS>(&self.we, &self.be, &x, e, false);
        if let Some(n) = noise {
            let mut xn = [[0f32; NIN]; B];
            for bb in 0..B {
                xn[bb][..H2].copy_from_slice(&x[bb]);
                xn[bb][H2..].copy_from_slice(&n[bb]);
            }
            let mut z = [[0f32; DIMS]; B];
            dense::<NIN, DIMS, B, DIMS>(&self.ws, &self.bs, &xn, &mut z, false);
            for bb in 0..B {
                for j in 0..DIMS {
                    sig[bb][j] = SIGMA_FLOOR + z[bb][j].exp();
                }
            }
        }
    }

    /// Embedding (and noise) of one featurised configuration.
    pub fn embed(&self, f: &Feats, with_noise: bool) -> ([f32; DIMS], [f32; DIMS]) {
        let h = [
            self.phi(&f.elem[0]),
            self.phi(&f.elem[1]),
            self.phi(&f.elem[2]),
        ];
        let p = [Self::pool([&h[0], &h[1], &h[2]], &f.glob)];
        let mut e = [[0f32; DIMS]; 1];
        let mut s = [[0f32; DIMS]; 1];
        self.heads::<1, 64>(&p, with_noise.then_some(&[f.noise]), &mut e, &mut s);
        (e[0], s[0])
    }

    /// Piecewise-linear CDF of dimension j (numpy.interp semantics on the f32 quantiles).
    #[inline]
    pub fn cdf(&self, j: usize, e: f32) -> f64 {
        let x = e as f64;
        if x.is_nan() {
            return x;
        }
        // f32 comparisons are exact equivalents of numpy's f64 comparisons of widened values
        let q = &self.q[j * N_Q..(j + 1) * N_Q];
        let pp = q.partition_point(|&v| v <= e);
        if pp == 0 {
            return 0.0;
        }
        if pp == N_Q {
            return 1.0;
        }
        let k = pp - 1;
        let yk = k as f64 / (N_Q - 1) as f64;
        let (q0, q1) = (q[k] as f64, q[k + 1] as f64);
        if q0 == x {
            return yk;
        }
        let slope = (1.0 / (N_Q - 1) as f64) / (q1 - q0);
        slope * (x - q0) + yk
    }

    #[inline]
    fn bin(&self, j: usize, c: f64) -> u32 {
        let n = (1u32 << self.bits[j]) as f64;
        (c * n).floor().min(n - 1.0) as u32
    }

    /// CDF values, bins and the key of an embedding.
    #[inline]
    pub fn quantise(&self, e: &[f32; DIMS]) -> ([f64; DIMS], [u32; DIMS], u32) {
        let mut c = [0.0; DIMS];
        let mut b = [0u32; DIMS];
        let mut key = 0u32;
        for j in 0..DIMS {
            c[j] = self.cdf(j, e[j]);
            b[j] = self.bin(j, c[j]);
            key |= gray(b[j]) << self.shift[j];
        }
        (c, b, key)
    }

    /// Per-bit log-odds (sign = the image key's bit, |l| = ln((1-P)/P)).
    pub fn logodds(&self, c: &[f64; DIMS], b: &[u32; DIMS], s: &[f32; DIMS]) -> [f32; 32] {
        let mut l = [0f32; 32];
        let p_sat = 1e-12f64;
        let saturated = ((1.0 - p_sat) / p_sat).ln();
        for j in 0..DIMS {
            let nbit = self.bits[j];
            let n = 1i64 << nbit;
            let w = 1.0 / n as f64;
            let bj = b[j] as i64;
            let g = gray(b[j]);
            let sj = s[j].max(1e-4f32) as f64;
            for k in 0..nbit {
                let step = 1i64 << (k + 1);
                let first = 1i64 << k;
                let up = bj + 1 + (first - (bj + 1)).rem_euclid(step);
                let dn = bj - (bj - first).rem_euclid(step);
                let m_up = if up < n {
                    up as f64 * w - c[j]
                } else {
                    f64::INFINITY
                };
                let m_dn = if dn >= 1 {
                    c[j] - dn as f64 * w
                } else {
                    f64::INFINITY
                };
                let m = m_up.min(m_dn).max(0.0);
                let z = -m / sj;
                // 0.5 exp(z) < 1e-12 for z <= -27: P clips to 1e-12 (same value as computed)
                let mag = if z <= -27.0 {
                    saturated
                } else {
                    let p = (0.5 * z.exp()).clamp(1e-12, 0.5 - 1e-7);
                    ((1.0 - p) / p).ln()
                };
                let bit = (g >> k) & 1;
                l[(self.shift[j] + k) as usize] = if bit == 1 { mag } else { -mag } as f32;
            }
        }
        l
    }

    /// Catalogue record key and parity flip; None if the configuration is degenerate.
    pub fn catalogue_key(&self, raw: &Raw) -> Option<(u32, bool)> {
        let f = featurize(raw, false)?;
        let (e, _) = self.embed(&f, false);
        let (_, _, key) = self.quantise(&e);
        Some((key, f.flip))
    }

    /// Image-side code of one branch using `cache` for the per-neighbour element embeddings
    /// (exact-input memoisation: bit-identical to `image_code`). Configurations of one anchor
    /// share neighbours under the same basis and parity, so most phi evaluations are reused.
    pub fn image_code_cached(
        &self,
        raw: &Raw,
        invert: bool,
        cache: &mut ElemCache,
    ) -> Option<ImageCode> {
        let b = basis(raw)?;
        let len = b.len;
        let flip = (b.spar < 0.0) ^ invert;
        let sgn = if flip { -1.0 } else { 1.0 };
        let mut elem = [[0f32; FE]; 3];
        let mut h = [[0f32; H1]; 3];
        for j in 0..3 {
            let (u, v, d) = (b.u[j], b.v[j] * sgn, raw.d[j + 2]);
            let key = [len.to_bits(), u.to_bits(), v.to_bits(), d.to_bits()];
            let slot = match cache.keys.iter().position(|k| *k == key) {
                Some(i) => i,
                None => {
                    let f = elem_features(u, v, d, len);
                    cache.keys.push(key);
                    cache.vals.push((f, self.phi(&f)));
                    cache.keys.len() - 1
                }
            };
            (elem[j], h[j]) = cache.vals[slot];
        }
        let mut feats = featurize_basis_tail(raw, &b, flip);
        feats.elem = elem;
        let p = [Self::pool([&h[0], &h[1], &h[2]], &feats.glob)];
        let mut e = [[0f32; DIMS]; 1];
        let mut sigma = [[0f32; DIMS]; 1];
        self.heads::<1, 64>(&p, Some(&[feats.noise]), &mut e, &mut sigma);
        let (cdf, bins, key) = self.quantise(&e[0]);
        let logodds = self.logodds(&cdf, &bins, &sigma[0]);
        Some(ImageCode {
            feats,
            e: e[0],
            sigma: sigma[0],
            cdf,
            key,
            logodds,
        })
    }

    /// `image_probes` with the element cache (bit-identical results).
    pub fn image_probes_cached(
        &self,
        raw: &Raw,
        k: usize,
        cache: &mut ElemCache,
        out: &mut Vec<Probe>,
    ) -> Option<bool> {
        out.clear();
        let c1 = self.image_code_cached(raw, false, cache)?;
        let prior2 = parity_prior(c1.feats.spar, c1.feats.spar_sigma);
        branch_probes(&c1.logodds, k, 0.0, c1.feats.flip, out);
        let both = prior2 < PRIOR_CUT;
        if both {
            let c2 = self.image_code_cached(raw, true, cache)?;
            branch_probes(&c2.logodds, k, prior2 as f64, c2.feats.flip, out);
            out.sort_by(|a, b| a.cost.total_cmp(&b.cost));
            out.truncate(k);
        }
        Some(both)
    }

    /// Image-side code of one branch (canonical when `invert` is false).
    pub fn image_code(&self, raw: &Raw, invert: bool) -> Option<ImageCode> {
        let feats = featurize(raw, invert)?;
        let (e, sigma) = self.embed(&feats, true);
        let (cdf, b, key) = self.quantise(&e);
        let logodds = self.logodds(&cdf, &b, &sigma);
        Some(ImageCode {
            feats,
            e,
            sigma,
            cdf,
            key,
            logodds,
        })
    }

    /// The K most probable (key, flip) probes of an image configuration, by cost (FEATURES.md
    /// section 6): both parity branches merged, canonical branch first on equal cost. Keys are
    /// not de-duplicated across branches: the same key with the other parity is a different
    /// hypothesis for the voter. Returns None for a degenerate configuration, otherwise whether
    /// the opposite-parity branch was enumerated.
    pub fn image_probes(&self, raw: &Raw, k: usize, out: &mut Vec<Probe>) -> Option<bool> {
        out.clear();
        let c1 = self.image_code(raw, false)?;
        let prior2 = parity_prior(c1.feats.spar, c1.feats.spar_sigma);
        branch_probes(&c1.logodds, k, 0.0, c1.feats.flip, out);
        let both = prior2 < PRIOR_CUT;
        if both {
            let c2 = self.image_code(raw, true)?;
            branch_probes(&c2.logodds, k, prior2 as f64, c2.feats.flip, out);
            // stable sort: the canonical branch wins ties
            out.sort_by(|a, b| a.cost.total_cmp(&b.cost));
            out.truncate(k);
        }
        Some(both)
    }

    /// Solver entry point: probes as key variants (key, flip); see `image_probes`. `cache`
    /// memoises element embeddings (clear it per anchor).
    pub fn image_variants(
        &self,
        raw: &Raw,
        k: usize,
        cache: &mut ElemCache,
        out: &mut Vec<KeyVariant>,
    ) -> Option<bool> {
        let mut p = Vec::with_capacity(2 * k);
        let r = self.image_probes_cached(raw, k, cache, &mut p);
        out.clear();
        out.extend(p.iter().map(|p| KeyVariant {
            key: p.key,
            flip: p.flip,
        }));
        r
    }
}

/// The `kmax` cheapest flip sets over the M_BITS least confident bits of one branch, appended
/// to `out` in increasing cost (best-first enumeration of subset sums).
pub fn branch_probes(l: &[f32; 32], kmax: usize, prior: f64, flip: bool, out: &mut Vec<Probe>) {
    if kmax == 0 {
        return;
    }
    let mut base = 0u32;
    for (i, &x) in l.iter().enumerate() {
        if x > 0.0 {
            base |= 1 << i;
        }
    }
    let mut idx: [u8; 32] = std::array::from_fn(|i| i as u8);
    idx.sort_by(|&a, &b| {
        l[a as usize]
            .abs()
            .total_cmp(&l[b as usize].abs())
            .then(a.cmp(&b))
    });
    let a: [f64; M_BITS] = std::array::from_fn(|i| l[idx[i] as usize].abs() as f64);
    let bit: [u32; M_BITS] = std::array::from_fn(|i| 1u32 << idx[i]);
    out.push(Probe {
        key: base,
        flip,
        cost: prior,
    });
    if kmax == 1 {
        return;
    }
    // heap entries: (cost, last index, flip mask)
    #[derive(PartialEq)]
    struct E(f64, u8, u32);
    impl Eq for E {}
    impl PartialOrd for E {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for E {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            // min-heap on cost, then on mask for determinism
            o.0.total_cmp(&self.0).then(o.2.cmp(&self.2))
        }
    }
    let mut heap = std::collections::BinaryHeap::with_capacity(2 * kmax);
    heap.push(E(a[0], 0, bit[0]));
    let mut n = 1;
    while n < kmax {
        let Some(E(c, last, mask)) = heap.pop() else {
            break;
        };
        out.push(Probe {
            key: base ^ mask,
            flip,
            cost: prior + c,
        });
        n += 1;
        let nx = last as usize + 1;
        if nx < M_BITS {
            heap.push(E(c + a[nx], nx as u8, mask | bit[nx]));
            heap.push(E(
                c - a[last as usize] + a[nx],
                nx as u8,
                (mask ^ bit[last as usize]) | bit[nx],
            ));
        }
    }
}

/// Load the coder of an index directory and check it against the manifest's SHA-256.
pub fn load_index_coder(dir: &Path, sha256: Option<&str>) -> Result<Coder> {
    let c = Coder::load(&dir.join(INDEX_CODER))?;
    match sha256 {
        Some(s) if s == c.sha256 => Ok(c),
        Some(s) => bail!("coder SHA-256 mismatch: manifest {s}, files {}", c.sha256),
        None => bail!("learned index without coder_sha256"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erfc_matches_reference_values() {
        for (x, v) in [
            (0.0, 1.0),
            (0.5, 0.479_500_122_186_953_5),
            (1.0, 0.157_299_207_050_285_13),
            (2.0, 0.004_677_734_981_047_266),
            (3.0, 2.209_049_699_858_544e-5),
            (5.0, 1.537_459_794_428_034_8e-12),
        ] {
            let r = erfc(x);
            assert!(((r - v) / v).abs() < 1e-12, "erfc({x}) = {r}, want {v}");
        }
    }

    #[test]
    fn best_first_enumeration_matches_brute_force() {
        let mut s = 0x1234_5678_9abc_def0u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..200 {
            let l: [f32; 32] =
                std::array::from_fn(|_| ((rnd() - 0.5) * 2.0 * (1.0 + 30.0 * rnd())) as f32);
            let mut got = vec![];
            branch_probes(&l, 64, 0.0, false, &mut got);
            // brute force over subsets of the 12 least confident bits
            let mut idx: Vec<usize> = (0..32).collect();
            idx.sort_by(|&a, &b| l[a].abs().total_cmp(&l[b].abs()).then(a.cmp(&b)));
            let mut all: Vec<f64> = (0..1u32 << 12)
                .map(|m| {
                    (0..12)
                        .filter(|i| m >> i & 1 == 1)
                        .map(|i| l[idx[i]].abs() as f64)
                        .sum()
                })
                .collect();
            all.sort_by(f64::total_cmp);
            assert_eq!(got.len(), 64);
            for (g, w) in got.iter().zip(&all) {
                assert!((g.cost - w).abs() < 1e-9, "{} vs {}", g.cost, w);
            }
            let mut keys: Vec<u32> = got.iter().map(|p| p.key).collect();
            keys.sort_unstable();
            keys.dedup();
            assert_eq!(keys.len(), 64);
        }
    }
}
