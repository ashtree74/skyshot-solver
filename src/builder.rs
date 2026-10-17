//! Index builder: exact dominance distances over the whole D80 catalogue, dominance levels,
//! per-anchor k-nearest-neighbour fingerprints and the memory-mapped index files.
//!
//! Dominance: D_i = angle from star i to the nearest *brighter* star (brightness order is
//! (mag10, catalogue position)); level m = { D >= R_m }, R_m = R_7 * 2^(7-m). Levels nest and
//! each has a near-uniform density ~ 1/(pi R_m^2) independent of the underlying star density.

use crate::{
    d80::{D80, RawStar, decode_raw},
    fingerprint::canonical_key,
    index::{
        AR_SCALE, DEEP_BUCKET_BITS, FORMAT_VERSION, FORMAT_VERSION_DEEP, LevelInfo, Manifest,
        N_BUCKETS, N_LEVELS, Record, STAR_BYTES, STAR_CELL_N, bucket_fp_layout, level_radius,
    },
    learned::{self, Coder},
    sphere::{CellIndex, CubeGrid, Vec3, angle, chord, face_coords, gnomonic, tangent_basis, unit},
};
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    f64::consts::{FRAC_PI_2, FRAC_PI_4, TAU},
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

#[derive(Clone, Debug)]
pub struct BuildConfig {
    pub d80: PathBuf,
    pub out: PathBuf,
    pub r7_arcmin: f64,
    /// Highest pool level that gets records (8 = full, 7 = standard, 6 = lite).
    pub max_pool_level: u32,
    pub profile: String,
    pub dump_levels: bool,
    pub max_bucket: usize,
    /// magnitude caps for depth-truncated pools (descending)
    pub caps: Vec<f32>,
    pub caps_min_level: u32,
    /// learned key coder prefix (`<prefix>.f32` / `<prefix>.json`); None = key v1
    pub coder: Option<PathBuf>,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            d80: "work/astap".into(),
            out: "work/index".into(),
            r7_arcmin: 1.94,
            max_pool_level: 8,
            profile: "full".into(),
            dump_levels: true,
            max_bucket: 4096,
            caps: vec![18.0, 16.5, 15.0, 13.0, 11.0],
            caps_min_level: 1,
            coder: None,
        }
    }
}

/// Record layout per pool level: (reference neighbour index, three neighbour indices).
pub fn record_plan(m: u32, dominant: bool) -> &'static [(usize, [usize; 3])] {
    const FULL4: [(usize, [usize; 3]); 4] = [
        (0, [1, 2, 3]),
        (0, [1, 2, 4]),
        (0, [1, 3, 4]),
        (0, [2, 3, 4]),
    ];
    const ONE: [(usize, [usize; 3]); 1] = [(0, [1, 2, 3])];
    const TWO: [(usize, [usize; 3]); 2] = [(0, [1, 2, 3]), (1, [2, 3, 4])];
    match (m, dominant) {
        (..=6, true) => &FULL4,
        (..=6, false) => &ONE,
        (7, true) => &TWO,
        (8, true) => &ONE,
        _ => &[],
    }
}

fn log(t0: Instant, msg: impl AsRef<str>) {
    eprintln!("[{:7.1}s] {}", t0.elapsed().as_secs_f64(), msg.as_ref());
}

#[inline]
fn vec32(v: Vec3) -> [f32; 3] {
    [v[0] as f32, v[1] as f32, v[2] as f32]
}
#[inline]
fn chord2(a: [f32; 3], b: [f32; 3]) -> f64 {
    let dx = a[0] as f64 - b[0] as f64;
    let dy = a[1] as f64 - b[1] as f64;
    let dz = a[2] as f64 - b[2] as f64;
    dx * dx + dy * dy + dz * dz
}
#[inline]
fn chord_to_angle(c: f64) -> f64 {
    2.0 * (c / 2.0).min(1.0).asin()
}

/// Cube cell of a star at level `k` of a pyramid derived from the level-0 grid (n0 >> k).
#[inline]
fn coarsen(cell0: u32, n0: u32, k: u32) -> u32 {
    let nn0 = n0 * n0;
    let (f, i, j) = (cell0 / nn0, cell0 % n0, (cell0 % nn0) / n0);
    let n = n0 >> k;
    f * n * n + (j >> k) * n + (i >> k)
}

pub struct Catalog {
    pub raw: Vec<RawStar>,
    pub vec: Vec<[f32; 3]>,
    /// ids sorted by brightness (brightest first)
    pub order: Vec<u32>,
    /// brightness rank per id
    pub rank: Vec<u32>,
}

pub fn load_catalog(dir: &Path, t0: Instant) -> Result<(Catalog, String)> {
    let db = D80::open(dir)?;
    let paths = db.tile_paths();
    let mut hasher = Sha256::new();
    for p in &paths {
        let md = fs::metadata(p)?;
        hasher.update(p.file_name().unwrap().to_string_lossy().as_bytes());
        hasher.update(md.len().to_le_bytes());
    }
    let fingerprint = format!("{:x}", hasher.finalize());
    let tiles: Vec<Vec<RawStar>> = paths
        .par_iter()
        .map(|p| -> Result<Vec<RawStar>> {
            let data = fs::read(p).with_context(|| format!("read {}", p.display()))?;
            let mut v = Vec::with_capacity(data.len() / 5);
            decode_raw(&data, &mut v)?;
            Ok(v)
        })
        .collect::<Result<_>>()?;
    let n: usize = tiles.iter().map(|t| t.len()).sum();
    let mut raw = Vec::with_capacity(n);
    for t in tiles {
        raw.extend(t);
    }
    log(
        t0,
        format!("decoded {} stars from {} tiles", raw.len(), paths.len()),
    );
    ensure!(raw.len() < u32::MAX as usize, "too many stars");
    // Brightness order: counting sort by mag10 (stable keeps catalogue order inside a bin).
    let lo = raw.iter().map(|s| s.mag10).min().unwrap_or(0) as i32;
    let hi = raw.iter().map(|s| s.mag10).max().unwrap_or(0) as i32;
    let mut counts = vec![0usize; (hi - lo + 2) as usize];
    for s in &raw {
        counts[(s.mag10 as i32 - lo) as usize + 1] += 1;
    }
    for i in 1..counts.len() {
        counts[i] += counts[i - 1];
    }
    let mut order = vec![0u32; raw.len()];
    for (i, s) in raw.iter().enumerate() {
        let b = (s.mag10 as i32 - lo) as usize;
        order[counts[b]] = i as u32;
        counts[b] += 1;
    }
    let mut rank = vec![0u32; raw.len()];
    for (r, &i) in order.iter().enumerate() {
        rank[i as usize] = r as u32;
    }
    let vec: Vec<[f32; 3]> = raw
        .par_iter()
        .map(|s| vec32(unit(s.ra_deg(), s.dec_deg())))
        .collect();
    log(t0, "brightness order and unit vectors ready");
    Ok((
        Catalog {
            raw,
            vec,
            order,
            rank,
        },
        fingerprint,
    ))
}

/// Exact dominance distance (radians) for every star with D >= r_min; others get NaN.
pub fn dominance(cat: &Catalog, r_min: f64, t0: Instant) -> Vec<f32> {
    let n0: u32 = 4096;
    let g0 = CubeGrid::new(n0);
    let cells0: Vec<u32> = cat
        .vec
        .par_iter()
        .map(|&v| g0.cell([v[0] as f64, v[1] as f64, v[2] as f64]))
        .collect();
    let idx0 = CellIndex::build(g0, &cells0, cat.order.iter().copied());
    log(t0, "level-0 grid built");
    let c2min = chord(r_min).powi(2);
    // Phase A: survivors have no brighter star within r_min.
    let done = AtomicU64::new(0);
    let survivor: Vec<bool> = (0..cat.raw.len())
        .into_par_iter()
        .map_init(Vec::new, |cells, id| {
            let v = cat.vec[id];
            let rk = cat.rank[id];
            g0.cap_cells([v[0] as f64, v[1] as f64, v[2] as f64], r_min, cells);
            for &c in cells.iter() {
                for &j in idx0.cell_items(c) {
                    if cat.rank[j as usize] >= rk {
                        break;
                    }
                    if chord2(v, cat.vec[j as usize]) < c2min {
                        return false;
                    }
                }
            }
            let k = done.fetch_add(1, Ordering::Relaxed);
            if k.is_multiple_of(10_000_000) {
                eprintln!("  survivors so far ~{k}");
            }
            true
        })
        .collect();
    let n_surv = survivor.iter().filter(|&&s| s).count();
    log(
        t0,
        format!(
            "phase A: {n_surv} stars with D >= {:.3} arcmin",
            r_min.to_degrees() * 60.0
        ),
    );
    // Phase B: pyramid grids for exact D of survivors.
    let ks: Vec<u32> = vec![2, 4, 6, 8, 10, 12];
    let pyramid: Vec<(u32, CubeGrid, CellIndex)> = ks
        .par_iter()
        .map(|&k| {
            let g = CubeGrid::new(n0 >> k);
            let cells: Vec<u32> = cells0.iter().map(|&c| coarsen(c, n0, k)).collect();
            let idx = CellIndex::build(g, &cells, cat.order.iter().copied());
            (k, g, idx)
        })
        .collect();
    drop(cells0);
    log(t0, "pyramid grids built");
    let mut d = vec![f32::NAN; cat.raw.len()];
    d.par_iter_mut()
        .enumerate()
        .for_each_init(Vec::new, |cells, (id, out)| {
            if !survivor[id] {
                return;
            }
            let v = cat.vec[id];
            let vd = [v[0] as f64, v[1] as f64, v[2] as f64];
            let rk = cat.rank[id];
            let mut r = 2.0 * r_min;
            loop {
                if r >= std::f64::consts::PI {
                    *out = std::f32::consts::PI;
                    return;
                }
                // coarsest pyramid level whose cells are still narrower than r/2
                let (_, g, idx) = pyramid
                    .iter()
                    .rev()
                    .find(|(_, g, _)| g.min_cell_width() <= r / 2.0)
                    .unwrap_or(&pyramid[0]);
                if r >= 1.0 {
                    cells.clear();
                    cells.extend(0..g.cells() as u32);
                } else {
                    g.cap_cells(vd, r, cells);
                }
                let mut best = f64::MAX;
                for &c in cells.iter() {
                    let items = idx.cell_items(c);
                    let end = items.partition_point(|&j| cat.rank[j as usize] < rk);
                    for &j in &items[..end] {
                        best = best.min(chord2(v, cat.vec[j as usize]));
                    }
                }
                let ang = if best == f64::MAX {
                    f64::MAX
                } else {
                    chord_to_angle(best.sqrt())
                };
                if ang <= r {
                    *out = ang as f32;
                    return;
                }
                r *= 2.0;
            }
        });
    log(t0, "phase B: exact dominance distances");
    d
}

#[derive(Serialize)]
struct LevelDumpHeader {
    count: u64,
    fields: &'static str,
}

/// k nearest members (angle-sorted) of `members` around `v`, excluding `self_id`.
#[allow(clippy::too_many_arguments)]
fn knn(
    grid: &CubeGrid,
    idx: &CellIndex,
    members: &[u32],
    exact: &[Vec3],
    v: Vec3,
    self_pos: u32,
    k: usize,
    r0: f64,
    cells: &mut Vec<u32>,
    cand: &mut Vec<(f64, u32)>,
) -> bool {
    let mut r = r0;
    loop {
        cand.clear();
        grid.cap_cells(v, r, cells);
        for &c in cells.iter() {
            for &p in idx.cell_items(c) {
                if p == self_pos {
                    continue;
                }
                let a = angle(v, exact[p as usize]);
                if a <= r {
                    cand.push((a, p));
                }
            }
        }
        if cand.len() >= k {
            cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
            cand.truncate(k);
            return true;
        }
        r *= 2.0;
        if r > 0.6 {
            return false;
        }
        let _ = members;
    }
}

pub fn build(cfg: &BuildConfig) -> Result<Manifest> {
    let t0 = Instant::now();
    ensure!(
        cfg.profile != "deep" || !cfg.out.join("manifest.json").exists(),
        "DEEP index already exists at {}; build a replacement in a new directory",
        cfg.out.display()
    );
    fs::create_dir_all(&cfg.out)?;
    let (cat, fingerprint) = load_catalog(&cfg.d80, t0)?;
    let r7 = cfg.r7_arcmin.to_radians() / 60.0;
    let r8 = level_radius(r7, 8);
    let d = dominance(&cat, r8, t0);
    // Survivors (level 8) in brightness order.
    let surv: Vec<u32> = cat
        .order
        .iter()
        .copied()
        .filter(|&i| d[i as usize].is_finite())
        .collect();
    let n_cat = cat.raw.len() as u64;
    log(t0, format!("L8 = {} stars", surv.len()));
    // exact f64 unit vectors for survivors (position in `surv` = survivor index)
    let exact: Vec<Vec3> = surv
        .par_iter()
        .map(|&i| {
            let s = cat.raw[i as usize];
            unit(s.ra_deg(), s.dec_deg())
        })
        .collect();
    let dist: Vec<f64> = surv.iter().map(|&i| d[i as usize] as f64).collect();
    let mags: Vec<f32> = surv.iter().map(|&i| cat.raw[i as usize].mag()).collect();
    drop(d);
    if cfg.dump_levels {
        let path = cfg.out.join("levels8.bin");
        let mut w = BufWriter::new(File::create(&path)?);
        for (k, &i) in surv.iter().enumerate() {
            let s = cat.raw[i as usize];
            w.write_all(&s.ra_deg().to_le_bytes())?;
            w.write_all(&s.dec_deg().to_le_bytes())?;
            w.write_all(&(dist[k].to_degrees() as f32).to_le_bytes())?;
            w.write_all(&s.mag().to_le_bytes())?;
        }
        w.flush()?;
        serde_json::to_writer(
            File::create(cfg.out.join("levels8.json"))?,
            &LevelDumpHeader {
                count: surv.len() as u64,
                fields: "ra f64 deg, dec f64 deg, D f32 deg, mag f32 (BP); brightness order",
            },
        )?;
        log(t0, "levels8.bin written");
    }
    drop(cat);
    write_index(cfg, t0, &exact, &dist, &mags, n_cat, fingerprint, None)
}

/// Write the index files from level-8 survivors (exact unit vectors, dominance distances in
/// radians, magnitudes; any order). `anchor_cone` restricts record anchors to a cap (regional
/// builds); stars outside it are still written for verification.
#[allow(clippy::too_many_arguments)]
pub fn write_index(
    cfg: &BuildConfig,
    t0: Instant,
    exact: &[Vec3],
    dist: &[f64],
    mags: &[f32],
    n_cat: u64,
    fingerprint: String,
    anchor_cone: Option<(Vec3, f64)>,
) -> Result<Manifest> {
    let deep_layout = cfg.profile == "deep";
    let r7 = cfg.r7_arcmin.to_radians() / 60.0;
    let r8 = level_radius(r7, 8);
    let surv_len = exact.len();
    let coder = match &cfg.coder {
        Some(p) => {
            let c = Coder::load(p)?;
            Coder::copy_files(p, &cfg.out)?;
            log(
                t0,
                format!("learned key coder {} ({})", p.display(), c.sha256),
            );
            Some(c)
        }
        None => None,
    };
    // Level-7 stars -> stars.bin ids, sorted by star cell then brightness.
    let sg = CubeGrid::new(STAR_CELL_N);
    let mut l7: Vec<(u32, u32)> = (0..surv_len as u32)
        .filter(|&k| dist[k as usize] >= r7)
        .map(|k| (sg.cell(exact[k as usize]), k))
        .collect();
    l7.par_sort_unstable();
    ensure!(
        l7.len() < (1 << if deep_layout { 28 } else { 24 }),
        "level 7 too large for this record format: {}",
        l7.len()
    );
    let mut star_id = vec![u32::MAX; surv_len];
    for (sid, &(_, k)) in l7.iter().enumerate() {
        star_id[k as usize] = sid as u32;
    }
    write_stars(
        &cfg.out,
        "stars.bin",
        "cells.bin",
        &sg,
        &l7,
        exact,
        dist,
        mags,
        r8,
    )?;
    log(t0, format!("stars.bin: {} level-7 stars", l7.len()));
    // deep.bin (level-8-only stars) serves verification/refinement of fields narrower than
    // ~0.5 deg; the lite profile (pool levels <= 6, fields >= ~1 deg) does without it
    let with_deep = cfg.max_pool_level >= 7;
    for f in ["deep.bin", "deep_cells.bin"] {
        let _ = fs::remove_file(cfg.out.join(f));
    }
    if with_deep {
        let mut deep: Vec<(u32, u32)> = (0..surv_len as u32)
            .filter(|&k| dist[k as usize] < r7)
            .map(|k| (sg.cell(exact[k as usize]), k))
            .collect();
        deep.par_sort_unstable();
        write_stars(
            &cfg.out,
            "deep.bin",
            "deep_cells.bin",
            &sg,
            &deep,
            exact,
            dist,
            mags,
            r8,
        )?;
        log(t0, format!("deep.bin: {} level-8-only stars", deep.len()));
    }
    // Records per pool level.
    let mut all: Vec<(u32, u64)> = Vec::new();
    let mut levels = Vec::new();
    for m in (if deep_layout { 7 } else { 1 })..=cfg.max_pool_level.min(8) {
        let rm = level_radius(r7, m);
        let members: Vec<u32> = (0..surv_len as u32)
            .filter(|&k| dist[k as usize] >= rm)
            .collect();
        let grid = CubeGrid::with_min_width(rm);
        let mcells: Vec<u32> = members
            .iter()
            .map(|&k| grid.cell(exact[k as usize]))
            .collect();
        let midx = CellIndex::build(grid, &mcells, 0..members.len() as u32);
        let mexact: Vec<Vec3> = members.iter().map(|&k| exact[k as usize]).collect();
        let anchors: Vec<u32> = (0..members.len() as u32)
            .filter(|&p| !record_plan(m, dist[members[p as usize] as usize] >= 2.0 * rm).is_empty())
            .filter(|&p| {
                anchor_cone.is_none_or(|(c, r)| angle(exact[members[p as usize] as usize], c) <= r)
            })
            .collect();
        // depth-truncated variants: pools restricted to mag <= cap (D is unchanged by the cap)
        let caps: Vec<f32> = if m >= cfg.caps_min_level {
            cfg.caps.clone()
        } else {
            vec![]
        };
        let mmag: Vec<f32> = members.iter().map(|&k| mags[k as usize]).collect();
        let capped: Vec<(f32, CellIndex, Vec<u32>)> = caps
            .par_iter()
            .map(|&c| {
                let sub: Vec<u32> = (0..members.len() as u32)
                    .filter(|&p| mmag[p as usize] <= c)
                    .collect();
                let cells: Vec<u32> = sub.iter().map(|&p| grid.cell(mexact[p as usize])).collect();
                // CellIndex over positions in `sub` (items are indices into `sub`)
                let ci = CellIndex::build(grid, &cells, 0..sub.len() as u32);
                (c, ci, sub)
            })
            .collect();
        let capped_exact: Vec<Vec<Vec3>> = capped
            .iter()
            .map(|(_, _, sub)| sub.iter().map(|&p| mexact[p as usize]).collect())
            .collect();
        // position of member p inside each capped subset (u32::MAX if absent)
        let capped_pos: Vec<Vec<u32>> = capped
            .iter()
            .map(|(_, _, sub)| {
                let mut v = vec![u32::MAX; members.len()];
                for (i, &p) in sub.iter().enumerate() {
                    v[p as usize] = i as u32;
                }
                v
            })
            .collect();
        let n_capped = AtomicU64::new(0);
        let recs: Vec<(u32, u64)> = anchors
            .par_iter()
            .map_init(
                || (Vec::new(), Vec::new()),
                |(cells, cand), &p| {
                    let k = members[p as usize] as usize;
                    let dominant = dist[k] >= 2.0 * rm;
                    let plan = if deep_layout {
                        record_plan(8, dominant)
                    } else {
                        record_plan(m, dominant)
                    };
                    let a = exact[k];
                    let mut out = Vec::with_capacity(plan.len());
                    if star_id[k] == u32::MAX {
                        return out;
                    }
                    let (east, north) = tangent_basis(a);
                    let dcode = ((2.0 * (dist[k] / rm).log2()).floor().max(0.0) as u32).min(7);
                    // nbv: neighbour unit vectors, nbk: their survivor indices (for D)
                    let emit = |nbv: &[Vec3], nbk: &[u32], out: &mut Vec<(u32, u64)>| -> bool {
                        let mut xy = [[0.0f64; 2]; 5];
                        for (t, v) in nbv.iter().enumerate() {
                            match gnomonic(*v, a, east, north) {
                                Some(p2) => xy[t] = p2,
                                None => return false,
                            }
                        }
                        for &(ri, sub) in plan {
                            let Some((key, flip, len)) = canonical_key(
                                [0.0, 0.0],
                                xy[ri],
                                [xy[sub[0]], xy[sub[1]], xy[sub[2]]],
                            ) else {
                                continue;
                            };
                            // same record plan and payload; only the key function and the
                            // meaning of the parity bit change with the learned coder
                            let (key, flip) = match &coder {
                                None => (key, flip),
                                Some(c) => {
                                    let raw = learned::Raw {
                                        xy: [xy[ri], xy[sub[0]], xy[sub[1]], xy[sub[2]]],
                                        d: [
                                            dist[k],
                                            dist[nbk[ri] as usize],
                                            dist[nbk[sub[0]] as usize],
                                            dist[nbk[sub[1]] as usize],
                                            dist[nbk[sub[2]] as usize],
                                        ],
                                        sigma: 0.0,
                                        edge: f64::INFINITY,
                                    };
                                    let Some(kf) = c.catalogue_key(&raw) else {
                                        continue;
                                    };
                                    kf
                                }
                            };
                            let ar = len.atan();
                            let lr = (ar / rm).log2();
                            if !(0.0..4.0).contains(&lr) {
                                continue;
                            }
                            let ar_code = ((lr * AR_SCALE).floor() as u32).min(1023);
                            let pa = xy[ri][0].atan2(xy[ri][1]).rem_euclid(TAU);
                            let pa_code = ((pa / TAU * 1024.0).floor() as u32) % 1024;
                            let (bucket, fp) = bucket_fp_layout(key, deep_layout);
                            out.push((
                                bucket as u32,
                                (if deep_layout {
                                    Record::new_deep
                                } else {
                                    Record::new
                                })(
                                    fp, star_id[k], ar_code, pa_code, flip, m, dominant, dcode
                                )
                                .0,
                            ));
                        }
                        true
                    };
                    if !knn(
                        &grid,
                        &midx,
                        &members,
                        &mexact,
                        a,
                        p,
                        5,
                        3.0 * rm,
                        cells,
                        cand,
                    ) {
                        return out;
                    }
                    let mut prev: Vec<u32> = cand.iter().map(|x| members[x.1 as usize]).collect();
                    let nbv: Vec<Vec3> = cand.iter().map(|x| mexact[x.1 as usize]).collect();
                    emit(&nbv, &prev, &mut out);
                    let mut maxmag = cand
                        .iter()
                        .map(|x| mmag[x.1 as usize])
                        .fold(f32::MIN, f32::max);
                    let amag = mmag[p as usize];
                    for (ci, (c, cidx, sub)) in capped.iter().enumerate() {
                        if *c >= maxmag || amag > *c {
                            continue;
                        }
                        let self_pos = capped_pos[ci][p as usize];
                        if !knn(
                            &grid,
                            cidx,
                            sub,
                            &capped_exact[ci],
                            a,
                            self_pos,
                            5,
                            3.0 * rm,
                            cells,
                            cand,
                        ) {
                            break;
                        }
                        let ids: Vec<u32> = cand
                            .iter()
                            .map(|x| members[sub[x.1 as usize] as usize])
                            .collect();
                        if ids == prev {
                            continue;
                        }
                        let nbv: Vec<Vec3> = cand
                            .iter()
                            .map(|x| capped_exact[ci][x.1 as usize])
                            .collect();
                        let before = out.len();
                        emit(&nbv, &ids, &mut out);
                        n_capped.fetch_add((out.len() - before) as u64, Ordering::Relaxed);
                        maxmag = cand
                            .iter()
                            .map(|x| mmag[sub[x.1 as usize] as usize])
                            .fold(f32::MIN, f32::max);
                        prev = ids;
                    }
                    out
                },
            )
            .flatten()
            .collect();
        log(
            t0,
            format!(
                "level {m}: {} depth-truncated records",
                n_capped.load(Ordering::Relaxed)
            ),
        );
        log(
            t0,
            format!(
                "level {m}: R={:.3}' members={} anchors={} records={}",
                rm.to_degrees() * 60.0,
                members.len(),
                anchors.len(),
                recs.len()
            ),
        );
        levels.push(LevelInfo {
            m,
            radius_arcmin: rm.to_degrees() * 60.0,
            stars: members.len() as u64,
            anchors: anchors.len() as u64,
            records: recs.len() as u64,
        });
        all.extend(recs);
    }
    // level 0 info (no records; pools start at m = 1)
    let n0 = dist.iter().filter(|&&x| x >= level_radius(r7, 0)).count() as u64;
    levels.insert(
        0,
        LevelInfo {
            m: 0,
            radius_arcmin: level_radius(r7, 0).to_degrees() * 60.0,
            stars: n0,
            anchors: 0,
            records: 0,
        },
    );
    let fp_bits = if deep_layout { 10 } else { 12 };
    let fp_mask = (1u64 << fp_bits) - 1;
    all.par_sort_unstable_by_key(|&(b, r)| ((b as u64) << fp_bits) | (r & fp_mask));
    // stop words: drop (bucket, fp) runs longer than max_bucket
    let mut kept: Vec<(u32, u64)> = Vec::with_capacity(all.len());
    let mut stop = 0u64;
    let mut i = 0;
    while i < all.len() {
        let key = ((all[i].0 as u64) << fp_bits) | (all[i].1 & fp_mask);
        let mut j = i;
        while j < all.len() && (((all[j].0 as u64) << fp_bits) | (all[j].1 & fp_mask)) == key {
            j += 1;
        }
        if j - i <= cfg.max_bucket {
            kept.extend_from_slice(&all[i..j]);
        } else {
            stop += (j - i) as u64;
        }
        i = j;
    }
    drop(all);
    let n_records = kept.len() as u64;
    write_codes(
        &cfg.out,
        &kept,
        if deep_layout {
            1 << DEEP_BUCKET_BITS
        } else {
            N_BUCKETS
        },
    )?;
    log(
        t0,
        format!("codes.bin: {n_records} records ({stop} dropped as stop words)"),
    );
    let mut files = vec![];
    for f in [
        "codes.bin",
        "keys.bin",
        "stars.bin",
        "cells.bin",
        "deep.bin",
        "deep_cells.bin",
    ] {
        if let Ok(md) = fs::metadata(cfg.out.join(f)) {
            files.push((f.to_string(), md.len()));
        }
    }
    let manifest = Manifest {
        format: "skyanchor2-index".into(),
        version: if deep_layout {
            FORMAT_VERSION_DEEP
        } else if coder.is_some() {
            crate::index::FORMAT_VERSION_LEARNED
        } else {
            FORMAT_VERSION
        },
        profile: cfg.profile.clone(),
        r7_arcmin: cfg.r7_arcmin,
        key_spec: match &coder {
            None => format!(
                "v1 rho_step={} rho_bins={} theta_bins={} bucket_bits={} fp_bits={}",
                crate::fingerprint::RHO_STEP,
                crate::fingerprint::RHO_BINS,
                crate::fingerprint::THETA_BINS,
                if deep_layout {
                    DEEP_BUCKET_BITS
                } else {
                    crate::index::BUCKET_BITS
                },
                fp_bits
            ),
            Some(c) => format!(
                "{} bits={:?} bucket_bits={} fp_bits={}",
                learned::KEY_TYPE,
                c.bits(),
                if deep_layout {
                    DEEP_BUCKET_BITS
                } else {
                    crate::index::BUCKET_BITS
                },
                fp_bits
            ),
        },
        levels,
        n_records,
        n_stars: l7.len() as u64,
        n_catalog: n_cat,
        star_cell_n: STAR_CELL_N,
        catalog: cfg.d80.display().to_string(),
        catalog_fingerprint: fingerprint,
        build_seconds: t0.elapsed().as_secs_f64(),
        files,
        stop_words: stop,
        key_type: if coder.is_some() {
            learned::KEY_TYPE.into()
        } else {
            "v1".into()
        },
        coder_sha256: coder.as_ref().map(|c| c.sha256.clone()),
    };
    serde_json::to_writer_pretty(File::create(cfg.out.join("manifest.json"))?, &manifest)?;
    log(t0, "manifest written");
    let _ = N_LEVELS;
    Ok(manifest)
}

#[allow(clippy::too_many_arguments)]
fn write_stars(
    out: &Path,
    stars_name: &str,
    cells_name: &str,
    sg: &CubeGrid,
    l7: &[(u32, u32)],
    exact: &[Vec3],
    dist: &[f64],
    mags: &[f32],
    r8: f64,
) -> Result<()> {
    let step = FRAC_PI_2 / sg.n as f64;
    let mut w = BufWriter::new(File::create(out.join(stars_name))?);
    let mut offsets = vec![0u32; sg.cells() + 1];
    for &(cell, _) in l7 {
        offsets[cell as usize + 1] += 1;
    }
    for i in 0..sg.cells() {
        offsets[i + 1] += offsets[i];
    }
    for &(cell, k) in l7 {
        let v = exact[k as usize];
        let (f, a, b) = face_coords(v);
        let (cf, ci, cj) = sg.split(cell);
        debug_assert_eq!(f, cf);
        let fa = ((a + FRAC_PI_4) / step - ci as f64).clamp(0.0, 0.999_999_99);
        let fb = ((b + FRAC_PI_4) / step - cj as f64).clamp(0.0, 0.999_999_99);
        let qa = (fa * 65536.0).floor() as u16;
        let qb = (fb * 65536.0).floor() as u16;
        let dc = ((dist[k as usize] / r8).log2() * 16.0).clamp(0.0, 255.0) as u8;
        let mc = ((mags[k as usize] + 2.0) * 10.0).round().clamp(0.0, 255.0) as u8;
        let mut rec = [0u8; STAR_BYTES];
        rec[0..2].copy_from_slice(&qa.to_le_bytes());
        rec[2..4].copy_from_slice(&qb.to_le_bytes());
        rec[4] = dc;
        rec[5] = mc;
        w.write_all(&rec)?;
    }
    w.flush()?;
    let mut wc = BufWriter::new(File::create(out.join(cells_name))?);
    for o in offsets {
        wc.write_all(&o.to_le_bytes())?;
    }
    wc.flush()?;
    Ok(())
}

fn write_codes(out: &Path, recs: &[(u32, u64)], buckets: usize) -> Result<()> {
    ensure!(recs.len() <= u32::MAX as usize, "too many index records");
    let mut offsets = vec![0u32; buckets + 1];
    for &(b, _) in recs {
        offsets[b as usize + 1] += 1;
    }
    for i in 0..buckets {
        offsets[i + 1] += offsets[i];
    }
    let mut w = BufWriter::new(File::create(out.join("codes.bin"))?);
    for &(_, r) in recs {
        w.write_all(&r.to_le_bytes())?;
    }
    w.flush()?;
    let mut wk = BufWriter::new(File::create(out.join("keys.bin"))?);
    for o in offsets {
        wk.write_all(&o.to_le_bytes())?;
    }
    wk.flush()?;
    Ok(())
}

/// Regional build (hinted deep mode): stars of D80 within `radius_deg` (+ margin) of a pointing,
/// dominance computed exactly on the tangent plane, index written with `cfg.r7_arcmin` (use a
/// finer value, e.g. 0.485' = R_9, for very narrow fields). Record anchors are limited to the cone.
pub fn build_region(cfg: &BuildConfig, ra: f64, dec: f64, radius_deg: f64) -> Result<Manifest> {
    ensure!(
        cfg.profile != "deep" || !cfg.out.join("manifest.json").exists(),
        "DEEP index already exists at {}; use a new output directory",
        cfg.out.display()
    );
    let t0 = Instant::now();
    fs::create_dir_all(&cfg.out)?;
    let r7 = cfg.r7_arcmin.to_radians() / 60.0;
    let margin = 2.5 * level_radius(r7, 1).to_degrees();
    let total = (radius_deg + margin).min(30.0);
    let db = D80::open(&cfg.d80)?;
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let deadline = Instant::now() + std::time::Duration::from_secs(600);
    let mut stars: Vec<crate::types::CatalogStar> = db
        .nearby(ra, dec, total, &cancel, deadline)?
        .into_iter()
        .map(|(_, s)| s)
        .collect();
    // brightness order with a deterministic tie break
    stars.sort_by(|a, b| {
        a.mag
            .partial_cmp(&b.mag)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.source_id.cmp(&b.source_id))
    });
    let n_cat = stars.len() as u64;
    log(
        t0,
        format!("region: {} stars within {total:.2} deg", stars.len()),
    );
    let c = unit(ra, dec);
    let (east, north) = tangent_basis(c);
    let vecs: Vec<Vec3> = stars.iter().map(|s| unit(s.ra, s.dec)).collect();
    // tangent-plane coordinates in arcseconds, shifted to be positive for the grid
    let k = 180.0 / std::f64::consts::PI * 3600.0;
    let ext = (total.to_radians()).tan() * k;
    let pts: Vec<[f64; 2]> = vecs
        .iter()
        .map(|&v| {
            let g = gnomonic(v, c, east, north).unwrap_or([f64::NAN, f64::NAN]);
            [g[0] * k + ext, g[1] * k + ext]
        })
        .collect();
    let d_arcsec = crate::select::dominance_px(&pts, 2.0 * ext, 2.0 * ext);
    let r8 = level_radius(r7, 8);
    let keep: Vec<usize> = (0..stars.len())
        .filter(|&i| pts[i][0].is_finite() && (d_arcsec[i] / k) >= r8)
        .collect();
    let exact: Vec<Vec3> = keep.iter().map(|&i| vecs[i]).collect();
    let dist: Vec<f64> = keep
        .iter()
        .map(|&i| (d_arcsec[i] / k).min(std::f64::consts::PI))
        .collect();
    let mags: Vec<f32> = keep.iter().map(|&i| stars[i].mag).collect();
    log(t0, format!("region: {} stars with D >= R_8", exact.len()));
    let fingerprint = format!(
        "region ra={ra:.5} dec={dec:.5} r={radius_deg:.3} d80={}",
        cfg.d80.display()
    );
    write_index(
        cfg,
        t0,
        &exact,
        &dist,
        &mags,
        n_cat,
        fingerprint,
        Some((c, radius_deg.to_radians())),
    )
}
