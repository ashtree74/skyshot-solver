//! On-disk index format (little-endian, memory-mapped).
//!
//! * `codes.bin`  — u64 records sorted by (bucket, fp); bucket/fp = split of `mix32(key)`.
//! * `keys.bin`   — (2^20 + 1) u32 offsets into `codes.bin` per bucket.
//! * `stars.bin`  — level-7 stars (anchor ids index this file), 6 bytes each, sorted by cube cell
//!   (n = 256) then brightness: u16 alpha fraction, u16 beta fraction within the cell, u8 dominance
//!   code (16 * log2(D / R_8), clamped), u8 magnitude code ((mag + 2) * 10).
//! * `cells.bin`  — (6 * 256^2 + 1) u32 offsets into `stars.bin` per cell.
//! * `manifest.json`.
//!
//! Record bits: fp 0..12 | anchor 12..36 | ar 36..46 (log2(|AR| / R_m) * 256) |
//! pa 46..56 (position angle A->R * 1024 / 2pi) | flip 56 | level 57..60 (m - 1) | dominant 60 |
//! dcode 61..64 (min(7, floor(2 log2(D_A / R_m)))).
//! Version 3 (`deep`): fp 0..10, anchor 10..38, ar 38..48, pa 48..58, flip 58,
//! level 59 (0 = 7, 1 = 8), dominant 60, dcode 61..64; 2^22 key buckets.

use crate::{
    fingerprint::mix32,
    sphere::{CubeGrid, Vec3, dot, face_vector},
};
use anyhow::{Context, Result, ensure};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::{
    f64::consts::{PI, TAU},
    fs::File,
    path::Path,
    sync::OnceLock,
};

/// Index format version of key-v1 indexes.
pub const FORMAT_VERSION: u32 = 1;
/// Format version of learned-key indexes: binaries that predate the learned key check
/// `version == 1` and refuse such an index instead of probing it with key v1.
pub const FORMAT_VERSION_LEARNED: u32 = 2;
/// Two-scale all-sky DEEP index: 28-bit anchor ids, 22-bit buckets and 10-bit fingerprints.
/// The complete 32-bit key and the 8-byte record are preserved.
pub const FORMAT_VERSION_DEEP: u32 = 3;
pub const DEEP_BUCKET_BITS: u32 = 22;
pub const BUCKET_BITS: u32 = 20;
pub const N_BUCKETS: usize = 1 << BUCKET_BITS;
pub const FP_MASK: u32 = 0xfff;
pub const STAR_CELL_N: u32 = 256;
pub const N_LEVELS: usize = 9;
pub const AR_SCALE: f64 = 256.0; // ar code = log2(|AR|/R_m) * 256, 10 bits -> [0, 4)
pub const STAR_BYTES: usize = 6;
/// Faintest magnitude kept in the decoded bright-star table ([`Index::bright_stars`]); BP 7.5 is
/// 23,272 stars of the phone index (`r7` 7.76', old L5), 40 bytes each, about 0.9 MB. Change it
/// before the first use with [`Index::set_bright_cut`].
pub const BRIGHT_CUT_MAG: f32 = 7.5;
/// Cone radius (rad) above which [`Index::cone_stars`] scans the bright table with dot products
/// instead of the cells: `CubeGrid::cap_cells` leaves its fast in-face block at 0.3 rad and
/// samples a tangent-plane lattice, which costs more than a scan of the table.
pub const BRIGHT_PATH_RADIUS: f64 = 0.3;

#[cfg(not(target_endian = "little"))]
compile_error!("Skyshot Solver index files are little-endian and mapped directly");

/// Level radius (radians) for level m, given R_7.
#[inline]
pub fn level_radius(r7: f64, m: u32) -> f64 {
    r7 * 2f64.powi(7 - m as i32)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Record(pub u64);

impl Record {
    #[allow(clippy::too_many_arguments)]
    pub fn new_deep(
        fp: u32,
        anchor: u32,
        ar: u32,
        pa: u32,
        flip: bool,
        level: u32,
        dominant: bool,
        dcode: u32,
    ) -> Self {
        assert!(fp < (1 << 10) && anchor < (1 << 28) && ar < 1024 && pa < 1024);
        assert!((7..=8).contains(&level) && dcode < 8);
        Self(
            fp as u64
                | (anchor as u64) << 10
                | (ar as u64) << 38
                | (pa as u64) << 48
                | (flip as u64) << 58
                | ((level - 7) as u64) << 59
                | (dominant as u64) << 60
                | (dcode as u64) << 61,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        fp: u32,
        anchor: u32,
        ar: u32,
        pa: u32,
        flip: bool,
        level: u32,
        dominant: bool,
        dcode: u32,
    ) -> Self {
        debug_assert!(fp <= 0xfff && anchor < (1 << 24) && ar < 1024 && pa < 1024);
        debug_assert!((1..=8).contains(&level) && dcode < 8);
        Self(
            fp as u64
                | (anchor as u64) << 12
                | (ar as u64) << 36
                | (pa as u64) << 46
                | (flip as u64) << 56
                | ((level - 1) as u64) << 57
                | (dominant as u64) << 60
                | (dcode as u64) << 61,
        )
    }
    #[inline]
    pub fn fp(self) -> u32 {
        (self.0 & 0xfff) as u32
    }
    #[inline]
    pub fn anchor(self) -> u32 {
        ((self.0 >> 12) & 0xff_ffff) as u32
    }
    #[inline]
    pub fn ar_code(self) -> u32 {
        ((self.0 >> 36) & 0x3ff) as u32
    }
    #[inline]
    pub fn pa_code(self) -> u32 {
        ((self.0 >> 46) & 0x3ff) as u32
    }
    #[inline]
    pub fn flip(self) -> bool {
        (self.0 >> 56) & 1 == 1
    }
    #[inline]
    pub fn level(self) -> u32 {
        ((self.0 >> 57) & 7) as u32 + 1
    }
    #[inline]
    pub fn dominant(self) -> bool {
        (self.0 >> 60) & 1 == 1
    }
    #[inline]
    pub fn dcode(self) -> u32 {
        (self.0 >> 61) as u32
    }
    /// |AR| in radians (bin centre).
    #[inline]
    pub fn ar_rad(self, r7: f64) -> f64 {
        level_radius(r7, self.level()) * ((self.ar_code() as f64 + 0.5) / AR_SCALE).exp2()
    }
    /// Position angle of A->R (radians, north through east, bin centre).
    #[inline]
    pub fn pa_rad(self) -> f64 {
        (self.pa_code() as f64 + 0.5) * TAU / 1024.0
    }
}

/// Decoded probe payload; legacy and DEEP records expose the same geometry to the solver.
#[derive(Clone, Copy, Debug)]
pub struct ProbeRecord {
    raw: u64,
    deep: bool,
}

impl ProbeRecord {
    pub fn anchor(self) -> u32 {
        if self.deep {
            ((self.raw >> 10) & 0x0fff_ffff) as u32
        } else {
            Record(self.raw).anchor()
        }
    }
    pub fn level(self) -> u32 {
        if self.deep {
            7 + ((self.raw >> 59) & 1) as u32
        } else {
            Record(self.raw).level()
        }
    }
    pub fn pa_code(self) -> u32 {
        if self.deep {
            ((self.raw >> 48) & 1023) as u32
        } else {
            Record(self.raw).pa_code()
        }
    }
    pub fn flip(self) -> bool {
        self.raw & (1 << if self.deep { 58 } else { 56 }) != 0
    }
    pub fn dcode(self) -> u32 {
        (self.raw >> 61) as u32
    }
    pub fn ar_rad(self, r7: f64) -> f64 {
        let ar = (self.raw >> if self.deep { 38 } else { 36 }) & 1023;
        level_radius(r7, self.level()) * ((ar as f64 + 0.5) / AR_SCALE).exp2()
    }
}

#[inline]
pub fn bucket_fp(key: u32) -> (usize, u32) {
    let h = mix32(key);
    ((h >> 12) as usize, h & FP_MASK)
}

#[inline]
pub fn bucket_fp_layout(key: u32, deep: bool) -> (usize, u32) {
    if deep {
        let h = mix32(key);
        ((h >> 10) as usize, h & 1023)
    } else {
        bucket_fp(key)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LevelInfo {
    pub m: u32,
    pub radius_arcmin: f64,
    pub stars: u64,
    pub anchors: u64,
    pub records: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub version: u32,
    pub profile: String,
    pub r7_arcmin: f64,
    pub key_spec: String,
    pub levels: Vec<LevelInfo>,
    pub n_records: u64,
    pub n_stars: u64,
    pub n_catalog: u64,
    pub star_cell_n: u32,
    pub catalog: String,
    pub catalog_fingerprint: String,
    pub build_seconds: f64,
    pub files: Vec<(String, u64)>,
    #[serde(default)]
    pub stop_words: u64,
    /// record key function: "v1" (hand-crafted, `fingerprint.rs`; default when absent) or
    /// "lgq-gd" (learned coder, `learned.rs`; files `coder.f32` / `coder.json` in the index)
    #[serde(default = "key_type_v1", skip_serializing_if = "is_key_v1")]
    pub key_type: String,
    /// SHA-256 of the learned coder (`coder.f32` bytes followed by `coder.json` bytes)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coder_sha256: Option<String>,
}

fn key_type_v1() -> String {
    "v1".into()
}

fn is_key_v1(k: &String) -> bool {
    k == "v1"
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StarInfo {
    pub v: Vec3,
    pub d_rad: f64,
    pub mag: f32,
}

/// The first cone at a given catalogue depth usually contains subsequent verification cones.
/// Store its cells in catalogue order, with one contiguous star allocation. Queries walk the
/// sorted cells once; missed cells use the ordinary decoder. No per-cell hash/heap overhead.
#[derive(Default)]
struct CachedCells {
    cells: Vec<u32>,
    ranges: Vec<std::ops::Range<usize>>,
    stars: Vec<StarInfo>,
}

/// Per-hypothesis decoded-cell reuse, bounded to 8192 cells and 65536 stars in total.
#[derive(Default)]
pub(crate) struct ConeCache {
    cuts: Option<(u8, u8)>,
    base: CachedCells,
    deep: CachedCells,
}

impl ConeCache {
    pub(crate) fn clear(&mut self) {
        self.cuts = None;
        for part in [&mut self.base, &mut self.deep] {
            part.cells.clear();
            part.ranges.clear();
            part.stars.clear();
        }
    }
}

/// A catalogue star decoded once from `stars.bin` (see [`Index::bright_stars`]).
///
/// Same frame and decoding as [`StarInfo`]: `v` is the equatorial (ICRS, catalogue epoch) unit
/// vector of the cell-quantised position, `mag` the BP magnitude from the 8-bit code
/// (`code / 10 - 2`), `d_rad` the dominance distance from the 8-bit code (`R_8 * 2^(code / 16)`,
/// saturating at code 255). `d_rad` stays `f64` so that comparisons against level radii
/// (`d_rad >= level_radius(m)`) give the same answer as on `StarInfo`. `id` is the row in
/// `stars.bin`, the anchor id of index records. 40 bytes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrightStar {
    pub v: Vec3,
    pub mag: f32,
    pub d_rad: f64,
    pub id: u32,
}

impl From<&BrightStar> for StarInfo {
    fn from(s: &BrightStar) -> Self {
        StarInfo {
            v: s.v,
            d_rad: s.d_rad,
            mag: s.mag,
        }
    }
}

/// Stars with magnitude code `<= cut_code`, sorted by (magnitude, id); decoded lazily.
struct BrightTable {
    cut: f32,
    /// largest magnitude code in the table (None: the cut is brighter than code 0)
    cut_code: Option<u8>,
    stars: Vec<BrightStar>,
}

pub struct Index {
    pub manifest: Manifest,
    pub r7: f64,
    codes: Mmap,
    keys: Mmap,
    stars: Mmap,
    cells: Mmap,
    pub star_grid: CubeGrid,
    /// cell index of star id (k << HINT_SHIFT), to start the forward scan in `star_cell`
    cell_hint: Vec<u32>,
    /// optional level-8-only stars (deep verification / refinement)
    deep: Option<(Mmap, Mmap)>,
    /// learned key coder (manifest key_type "lgq-gd"); None for key v1
    pub coder: Option<crate::learned::Coder>,
    /// bright stars decoded once on first use (`bright_stars` / `cone_stars`)
    bright: OnceLock<BrightTable>,
    bright_cut: f32,
}

const HINT_SHIFT: u32 = 6;

fn map(path: &Path) -> Result<Mmap> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    // SAFETY: index files are immutable once built; we only read them.
    Ok(unsafe { Mmap::map(&f)? })
}

/// The companion index for wide fields of a main index directory: the sibling `index-phone`
/// (`index-phone-v1` next to a key-v1 index), when built. See `solve::SolveConfig::wide_index`.
pub fn companion_wide_dir(main: &Path) -> Option<std::path::PathBuf> {
    let parent = main.parent()?;
    let name = main.file_name()?.to_string_lossy();
    let wide = if name.ends_with("-v1") {
        "index-phone-v1"
    } else {
        "index-phone"
    };
    let dir = parent.join(wide);
    (name != wide && dir.join("manifest.json").exists()).then_some(dir)
}

pub fn companion_deep_dir(main: &Path) -> Option<std::path::PathBuf> {
    let dir = main.parent()?.join("index-deep");
    (dir != main && dir.join("manifest.json").exists()).then_some(dir)
}

impl Index {
    pub fn open(dir: &Path) -> Result<Self> {
        let manifest: Manifest = serde_json::from_reader(File::open(dir.join("manifest.json"))?)
            .context("manifest.json")?;
        let expected = if manifest.key_type == "v1" {
            FORMAT_VERSION
        } else {
            FORMAT_VERSION_LEARNED
        };
        // learned indexes written before the version bump carry version 1 and a key_type
        ensure!(
            manifest.format == "skyanchor2-index"
                && (manifest.version == expected
                    || manifest.version == FORMAT_VERSION
                    || (manifest.version == FORMAT_VERSION_DEEP && manifest.profile == "deep")),
            "unsupported index (format {} version {}, key type {})",
            manifest.format,
            manifest.version,
            manifest.key_type
        );
        let coder = match manifest.key_type.as_str() {
            "v1" => None,
            crate::learned::KEY_TYPE => Some(crate::learned::load_index_coder(
                dir,
                manifest.coder_sha256.as_deref(),
            )?),
            k => anyhow::bail!("unsupported key type {k}"),
        };
        let codes = map(&dir.join("codes.bin"))?;
        let keys = map(&dir.join("keys.bin"))?;
        let stars = map(&dir.join("stars.bin"))?;
        let cells = map(&dir.join("cells.bin"))?;
        let buckets = if manifest.version == FORMAT_VERSION_DEEP {
            1 << DEEP_BUCKET_BITS
        } else {
            N_BUCKETS
        };
        ensure!(keys.len() == (buckets + 1) * 4, "keys.bin size");
        ensure!(
            codes.len() % 8 == 0 && codes.len() as u64 / 8 == manifest.n_records,
            "codes.bin size"
        );
        ensure!(
            stars.len() == manifest.n_stars as usize * STAR_BYTES,
            "stars.bin size"
        );
        let star_grid = CubeGrid::new(manifest.star_cell_n);
        ensure!(cells.len() == (star_grid.cells() + 1) * 4, "cells.bin size");
        ensure!(
            (codes.as_ptr() as usize).is_multiple_of(8)
                && (keys.as_ptr() as usize).is_multiple_of(4),
            "alignment"
        );
        #[cfg(unix)]
        {
            let _ = codes.advise(memmap2::Advice::Random);
        }
        // id -> cell hints
        let off: &[u32] =
            unsafe { std::slice::from_raw_parts(cells.as_ptr() as *const u32, cells.len() / 4) };
        let n_stars = manifest.n_stars as u32;
        let mut cell_hint = Vec::with_capacity((n_stars >> HINT_SHIFT) as usize + 1);
        let mut c = 0usize;
        let mut id = 0u32;
        while id < n_stars {
            while off[c + 1] <= id {
                c += 1;
            }
            cell_hint.push(c as u32);
            id += 1 << HINT_SHIFT;
        }
        let deep = if dir.join("deep.bin").exists() && dir.join("deep_cells.bin").exists() {
            let ds = map(&dir.join("deep.bin"))?;
            let dc = map(&dir.join("deep_cells.bin"))?;
            ensure!(
                dc.len() == (star_grid.cells() + 1) * 4 && ds.len() % STAR_BYTES == 0,
                "deep files"
            );
            Some((ds, dc))
        } else {
            None
        };
        Ok(Self {
            r7: manifest.r7_arcmin.to_radians() / 60.0,
            manifest,
            codes,
            keys,
            stars,
            cells,
            star_grid,
            cell_hint,
            deep,
            coder,
            bright: OnceLock::new(),
            bright_cut: BRIGHT_CUT_MAG,
        })
    }

    /// The memory-mapped files.
    fn maps(&self) -> Vec<&Mmap> {
        let mut m = vec![&self.codes, &self.keys, &self.stars, &self.cells];
        if let Some((a, b)) = &self.deep {
            m.extend([a, b]);
        }
        m
    }

    /// Bytes of the memory-mapped files.
    pub fn mapped_bytes(&self) -> u64 {
        self.maps().iter().map(|m| m.len() as u64).sum()
    }

    /// Read one byte of every page so the files sit in the page cache before the first solve.
    /// Returns the bytes touched.
    pub fn touch(&self) -> u64 {
        let mut sum = 0u8;
        for m in self.maps() {
            for i in (0..m.len()).step_by(4096) {
                sum = sum.wrapping_add(m[i]);
            }
        }
        std::hint::black_box(sum);
        self.mapped_bytes()
    }

    /// Lock the mapped files in memory (`mlock`): the kernel never evicts them, so no solve
    /// waits for the disk. Needs a memlock limit of at least [`Index::mapped_bytes`].
    pub fn lock_in_memory(&self) -> std::io::Result<u64> {
        for m in self.maps() {
            m.lock()?;
        }
        Ok(self.mapped_bytes())
    }

    /// An opaque identifier of the build (results can differ between builds); reveals nothing
    /// about the index format or the key.
    pub fn build_id(&self) -> String {
        use sha2::{Digest, Sha256};
        let m = &self.manifest;
        let mut h = Sha256::new();
        h.update(m.catalog_fingerprint.as_bytes());
        h.update(m.coder_sha256.as_deref().unwrap_or("").as_bytes());
        h.update(m.profile.as_bytes());
        h.update(m.n_records.to_le_bytes());
        let d = h.finalize();
        format!(
            "idx-{}",
            d[..6]
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        )
    }

    #[inline]
    pub fn codes(&self) -> &[u64] {
        // SAFETY: length checked in open(), mmap is page-aligned, file is little-endian u64.
        unsafe {
            std::slice::from_raw_parts(self.codes.as_ptr() as *const u64, self.codes.len() / 8)
        }
    }
    #[inline]
    fn offsets(&self) -> &[u32] {
        unsafe { std::slice::from_raw_parts(self.keys.as_ptr() as *const u32, self.keys.len() / 4) }
    }
    #[inline]
    fn cell_offsets(&self) -> &[u32] {
        unsafe {
            std::slice::from_raw_parts(self.cells.as_ptr() as *const u32, self.cells.len() / 4)
        }
    }

    /// Records whose key equals `key` exactly.
    #[inline]
    pub fn probe(&self, key: u32) -> impl Iterator<Item = ProbeRecord> + '_ {
        let deep = self.manifest.version == FORMAT_VERSION_DEEP;
        let mask = if deep { 1023 } else { 0xfff };
        let (b, fp) = bucket_fp_layout(key, deep);
        let off = self.offsets();
        let (s, e) = (off[b] as usize, off[b + 1] as usize);
        let codes = &self.codes()[s..e];
        // records are sorted by fp inside the bucket
        let start = codes.partition_point(|&c| ((c & mask) as u32) < fp);
        codes[start..]
            .iter()
            .take_while(move |&&c| (c & mask) as u32 == fp)
            .map(move |&raw| ProbeRecord { raw, deep })
    }

    /// Bucket length for diagnostics.
    pub fn n_buckets(&self) -> usize {
        self.keys.len() / 4 - 1
    }

    pub fn bucket_len(&self, b: usize) -> usize {
        let off = self.offsets();
        (off[b + 1] - off[b]) as usize
    }

    #[inline]
    pub fn n_stars(&self) -> usize {
        self.stars.len() / STAR_BYTES
    }

    /// Cell containing star `id` (stars are sorted by cell).
    #[inline]
    pub fn star_cell(&self, id: u32) -> u32 {
        let off = self.cell_offsets();
        let mut c = self.cell_hint[(id >> HINT_SHIFT) as usize] as usize;
        while off[c + 1] <= id {
            c += 1;
        }
        c as u32
    }

    #[inline]
    pub fn star_in_cell(&self, id: u32, cell: u32) -> StarInfo {
        self.decode_star(&self.stars, id, cell)
    }

    pub fn has_deep(&self) -> bool {
        self.deep.is_some()
    }

    /// Deep (level-8-only) stars in the cells that may intersect the cap: decoded StarInfo.
    pub fn deep_cone(&self, c: Vec3, radius: f64, cells: &mut Vec<u32>, out: &mut Vec<StarInfo>) {
        out.clear();
        let Some((ds, dc)) = &self.deep else { return };
        let off: &[u32] =
            unsafe { std::slice::from_raw_parts(dc.as_ptr() as *const u32, dc.len() / 4) };
        if radius >= 1.0 {
            cells.clear();
            cells.extend(0..self.star_grid.cells() as u32);
        } else {
            self.star_grid.cap_cells(c, radius, cells);
        }
        for &cell in cells.iter() {
            for id in off[cell as usize]..off[cell as usize + 1] {
                out.push(self.decode_star(ds, id, cell));
            }
        }
    }

    #[inline]
    fn decode_star(&self, data: &[u8], id: u32, cell: u32) -> StarInfo {
        let b = &data[id as usize * STAR_BYTES..id as usize * STAR_BYTES + STAR_BYTES];
        let fa = u16::from_le_bytes([b[0], b[1]]) as f64;
        let fb = u16::from_le_bytes([b[2], b[3]]) as f64;
        let (f, i, j) = self.star_grid.split(cell);
        let step = std::f64::consts::FRAC_PI_2 / self.star_grid.n as f64;
        let a = -std::f64::consts::FRAC_PI_4 + (i as f64 + (fa + 0.5) / 65536.0) * step;
        let bb = -std::f64::consts::FRAC_PI_4 + (j as f64 + (fb + 0.5) / 65536.0) * step;
        StarInfo {
            v: face_vector(f, a, bb),
            d_rad: level_radius(self.r7, 8) * (b[4] as f64 / 16.0).exp2(),
            mag: b[5] as f32 / 10.0 - 2.0,
        }
    }

    #[inline]
    pub fn star(&self, id: u32) -> StarInfo {
        self.star_in_cell(id, self.star_cell(id))
    }

    /// Star ids in the cells that may intersect the cap (caller filters by exact distance).
    pub fn cone_candidates(
        &self,
        c: Vec3,
        radius: f64,
        cells: &mut Vec<u32>,
        out: &mut Vec<(u32, u32)>,
    ) {
        out.clear();
        if radius >= 1.0 {
            cells.clear();
            cells.extend(0..self.star_grid.cells() as u32);
        } else {
            self.star_grid.cap_cells(c, radius, cells);
        }
        let off = self.cell_offsets();
        for &cell in cells.iter() {
            for id in off[cell as usize]..off[cell as usize + 1] {
                out.push((id, cell));
            }
        }
    }

    /// Decode only stars passing the solver's magnitude/dominance cuts, from cells that
    /// may intersect the cap. Deliberately no exact angular test: the solver's historical
    /// cell path projects these candidates into the frame. Preserve its order (base then
    /// deep, cell then id) because nearest-neighbour matching is order-sensitive.
    pub(crate) fn filtered_cone_candidates(
        &self,
        c: Vec3,
        radius: f64,
        d_min: f64,
        mag_limit: Option<f32>,
        include_deep: bool,
    ) -> Vec<StarInfo> {
        let Some(dc) = self.d_code_min(d_min) else {
            return vec![];
        };
        let mc = match mag_limit {
            Some(m) => match mag_code_max(m) {
                Some(code) => code,
                None => return vec![],
            },
            None => u8::MAX,
        };
        let mut cells = vec![];
        if radius >= 1.0 {
            cells.extend(0..self.star_grid.cells() as u32);
        } else {
            self.star_grid.cap_cells(c, radius, &mut cells);
        }
        let mut out = vec![];
        let mut append = |data: &[u8], offsets: &[u32]| {
            for &cell in &cells {
                for id in offsets[cell as usize]..offsets[cell as usize + 1] {
                    let b = &data[id as usize * STAR_BYTES..(id as usize + 1) * STAR_BYTES];
                    if b[4] >= dc && b[5] <= mc {
                        out.push(self.decode_star(data, id, cell));
                    }
                }
            }
        };
        append(&self.stars, self.cell_offsets());
        if include_deep && let Some((data, offsets)) = &self.deep {
            let offsets = unsafe {
                std::slice::from_raw_parts(offsets.as_ptr() as *const u32, offsets.len() / 4)
            };
            append(data, offsets);
        }
        out
    }

    /// Same candidates and order as `filtered_cone_candidates`, reusing overlapping cells.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cached_cone_candidates(
        &self,
        c: Vec3,
        radius: f64,
        d_min: f64,
        mag_limit: Option<f32>,
        include_deep: bool,
        cache: &mut ConeCache,
    ) -> Vec<StarInfo> {
        let Some(dc) = self.d_code_min(d_min) else {
            return vec![];
        };
        let mc = match mag_limit {
            Some(m) => match mag_code_max(m) {
                Some(code) => code,
                None => return vec![],
            },
            None => u8::MAX,
        };
        if cache.cuts != Some((dc, mc)) {
            cache.clear();
            cache.cuts = Some((dc, mc));
        }
        // All-sky candidates would mostly be used once and exceed the bounded cache.
        if radius >= 1.0 {
            return self.filtered_cone_candidates(c, radius, d_min, mag_limit, include_deep);
        }
        let mut cells = vec![];
        self.star_grid.cap_cells(c, radius, &mut cells);
        let mut out = vec![];
        let mut append = |data: &[u8], offsets: &[u32], part: &mut CachedCells| {
            let fill = part.cells.is_empty();
            let mut cursor = 0;
            for &cell in &cells {
                if !fill {
                    while cursor < part.cells.len() && part.cells[cursor] < cell {
                        cursor += 1;
                    }
                    if cursor < part.cells.len() && part.cells[cursor] == cell {
                        out.extend_from_slice(&part.stars[part.ranges[cursor].clone()]);
                        continue;
                    }
                }
                let start = out.len();
                for id in offsets[cell as usize]..offsets[cell as usize + 1] {
                    let b = &data[id as usize * STAR_BYTES..(id as usize + 1) * STAR_BYTES];
                    if b[4] >= dc && b[5] <= mc {
                        out.push(self.decode_star(data, id, cell));
                    }
                }
                let n = out.len() - start;
                if fill && part.cells.len() < 4096 && part.stars.len() + n <= 32768 {
                    part.cells.push(cell);
                    part.ranges.push(part.stars.len()..part.stars.len() + n);
                    part.stars.extend_from_slice(&out[start..]);
                }
            }
        };
        append(&self.stars, self.cell_offsets(), &mut cache.base);
        if include_deep && let Some((data, offsets)) = &self.deep {
            let offsets = unsafe {
                std::slice::from_raw_parts(offsets.as_ptr() as *const u32, offsets.len() / 4)
            };
            append(data, offsets, &mut cache.deep);
        }
        out
    }

    pub fn level_radius(&self, m: u32) -> f64 {
        level_radius(self.r7, m)
    }

    // ----- bright-star table and cone queries -------------------------------------------------

    /// Magnitude limit of the bright-star table (default [`BRIGHT_CUT_MAG`]).
    pub fn bright_cut(&self) -> f32 {
        self.bright_cut
    }

    /// Change the magnitude limit of the bright-star table; a table decoded earlier is dropped
    /// and rebuilt on the next use. Needs `&mut`, so call it right after `open`.
    pub fn set_bright_cut(&mut self, mag: f32) {
        self.bright_cut = mag;
        self.bright = OnceLock::new();
    }

    fn bright_table(&self) -> &BrightTable {
        self.bright
            .get_or_init(|| self.decode_bright(self.bright_cut))
    }

    /// Decode every star whose magnitude code is within `cut` (one pass over `stars.bin` with a
    /// raw-byte test, decode only the survivors), sorted by (mag, id).
    fn decode_bright(&self, cut: f32) -> BrightTable {
        let cut_code = mag_code_max(cut);
        let mut stars = Vec::new();
        if let Some(mc) = cut_code {
            let off = self.cell_offsets();
            let data = &self.stars[..];
            for cell in 0..self.star_grid.cells() {
                for id in off[cell]..off[cell + 1] {
                    if data[id as usize * STAR_BYTES + 5] > mc {
                        continue;
                    }
                    let st = self.decode_star(data, id, cell as u32);
                    stars.push(BrightStar {
                        v: st.v,
                        mag: st.mag,
                        d_rad: st.d_rad,
                        id,
                    });
                }
            }
            stars.sort_by(|a, b| a.mag.total_cmp(&b.mag).then(a.id.cmp(&b.id)));
        }
        BrightTable {
            cut,
            cut_code,
            stars,
        }
    }

    /// Stars with `mag <= min(mag_max, bright_cut())`, sorted by (magnitude, id). The table is
    /// decoded from `stars.bin` on the first call (one pass over the file, about 1 ms for the
    /// phone index) and shared by all threads afterwards; the slice is a prefix of it, so this
    /// call is O(log n). Stars fainter than the cut are never in the slice: use [`Index::cone_stars`]
    /// when `mag_max` may exceed the cut.
    pub fn bright_stars(&self, mag_max: f32) -> &[BrightStar] {
        let t = self.bright_table();
        let n = t.stars.partition_point(|s| s.mag <= mag_max);
        &t.stars[..n]
    }

    /// Catalogue stars within `radius` (rad) of the unit vector `c` with `mag <= mag_max` and
    /// `d_rad >= d_min` (both inclusive, on the decoded values of [`StarInfo`]).
    ///
    /// The angular test is `dot(v, c) >= cos(min(radius, pi))`. The result equals the cell path
    /// (`cone_candidates` + `star_in_cell`) filtered by the same three tests; only the order of
    /// `out` differs between paths (unspecified; sort by `id` when it matters). Path choice:
    /// * `mag_max <= bright_cut()`: scan of the bright table (dot product per star, no cells);
    /// * else if `radius > BRIGHT_PATH_RADIUS`: bright table plus one raw-byte pass over
    ///   `stars.bin` for the stars fainter than the cut (no `cap_cells` sampling);
    /// * else: [`Index::cone_stars_cells`] (cap cells, raw-byte prefilter, decode, exact angle).
    ///
    /// `out` is cleared first. Nothing is returned for a non-finite or negative radius.
    pub fn cone_stars(
        &self,
        c: Vec3,
        radius: f64,
        mag_max: f32,
        d_min: f64,
        out: &mut Vec<BrightStar>,
    ) {
        out.clear();
        if !radius.is_finite() || radius < 0.0 {
            return;
        }
        let t = self.bright_table();
        if mag_max > t.cut && radius <= BRIGHT_PATH_RADIUS {
            self.cone_stars_cells(c, radius, mag_max, d_min, out);
            return;
        }
        let cos_r = radius.min(PI).cos();
        for s in self.bright_stars(mag_max) {
            if s.d_rad >= d_min && dot(s.v, c) >= cos_r {
                out.push(*s);
            }
        }
        if mag_max > t.cut {
            // stars fainter than the table: magnitude codes in (cut_code, mag_code_max(mag_max)]
            let lo = match t.cut_code {
                Some(cc) if cc == u8::MAX => return,
                Some(cc) => cc + 1,
                None => 0,
            };
            let Some(hi) = mag_code_max(mag_max) else {
                return;
            };
            let Some(dc) = self.d_code_min(d_min) else {
                return;
            };
            let off = self.cell_offsets();
            let data = &self.stars[..];
            for cell in 0..self.star_grid.cells() {
                for id in off[cell]..off[cell + 1] {
                    let b = &data[id as usize * STAR_BYTES..id as usize * STAR_BYTES + STAR_BYTES];
                    if b[5] < lo || b[5] > hi || b[4] < dc {
                        continue;
                    }
                    let st = self.decode_star(data, id, cell as u32);
                    if dot(st.v, c) >= cos_r {
                        out.push(BrightStar {
                            v: st.v,
                            mag: st.mag,
                            d_rad: st.d_rad,
                            id,
                        });
                    }
                }
            }
        }
    }

    /// The cell path of [`Index::cone_stars`] for any radius: cap cells (`CubeGrid::cap_cells`,
    /// all cells at `radius >= 1`), a raw-byte prefilter on the magnitude and dominance codes,
    /// decode of the survivors, exact angular test. Same result and conventions as `cone_stars`;
    /// exposed for benchmarks and tests. Allocates a scratch cell list per call.
    pub fn cone_stars_cells(
        &self,
        c: Vec3,
        radius: f64,
        mag_max: f32,
        d_min: f64,
        out: &mut Vec<BrightStar>,
    ) {
        out.clear();
        if !radius.is_finite() || radius < 0.0 {
            return;
        }
        let Some(mc) = mag_code_max(mag_max) else {
            return;
        };
        let Some(dc) = self.d_code_min(d_min) else {
            return;
        };
        let cos_r = radius.min(PI).cos();
        let mut cells = Vec::new();
        if radius >= 1.0 {
            cells.extend(0..self.star_grid.cells() as u32);
        } else {
            self.star_grid.cap_cells(c, radius, &mut cells);
        }
        let off = self.cell_offsets();
        let data = &self.stars[..];
        for &cell in &cells {
            for id in off[cell as usize]..off[cell as usize + 1] {
                let b = &data[id as usize * STAR_BYTES..id as usize * STAR_BYTES + STAR_BYTES];
                if b[5] > mc || b[4] < dc {
                    continue;
                }
                let st = self.decode_star(data, id, cell);
                if dot(st.v, c) >= cos_r {
                    out.push(BrightStar {
                        v: st.v,
                        mag: st.mag,
                        d_rad: st.d_rad,
                        id,
                    });
                }
            }
        }
    }

    /// Smallest dominance code whose decoded distance is `>= d_min` (None: even code 255 is too
    /// small, i.e. no star qualifies). Same decoding as `StarInfo::d_rad`.
    fn d_code_min(&self, d_min: f64) -> Option<u8> {
        let r8 = level_radius(self.r7, 8);
        (0..=u8::MAX).find(|&c| decode_d(r8, c) >= d_min)
    }
}

/// Decoded magnitude of an 8-bit code, exactly as `Index::star_in_cell` computes it.
#[inline]
fn decode_mag(code: u8) -> f32 {
    code as f32 / 10.0 - 2.0
}

/// Decoded dominance distance (rad) of an 8-bit code given R_8, exactly as `Index::star_in_cell`.
#[inline]
fn decode_d(r8: f64, code: u8) -> f64 {
    r8 * (code as f64 / 16.0).exp2()
}

/// Largest magnitude code whose decoded magnitude is `<= mag_max` (None: code 0 is already too
/// faint). Decoding is strictly increasing in the code, so `code <= mag_code_max(m)` is exactly
/// `decode_mag(code) <= m`.
fn mag_code_max(mag_max: f32) -> Option<u8> {
    (0..=u8::MAX).rev().find(|&c| decode_mag(c) <= mag_max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sphere::{angle, unit};
    use rand::{Rng, SeedableRng};
    use std::path::PathBuf;

    #[test]
    fn deep_records_preserve_large_ids_and_geometric_extremes() {
        for anchor in [0, (1 << 24) - 1, 1 << 24, 140_875_367, (1 << 28) - 1] {
            for level in [7, 8] {
                for (ar, pa, flip, dcode) in
                    [(0, 1023, false, 0), (1023, 0, true, 7), (511, 512, true, 4)]
                {
                    let rec = Record::new_deep(1023, anchor, ar, pa, flip, level, true, dcode);
                    let decoded = ProbeRecord {
                        raw: rec.0,
                        deep: true,
                    };
                    assert_eq!(decoded.anchor(), anchor);
                    assert_eq!(decoded.level(), level);
                    assert_eq!(decoded.pa_code(), pa);
                    assert_eq!(decoded.flip(), flip);
                    assert_eq!(decoded.dcode(), dcode);
                    let legacy = Record::new(0, 0, ar, pa, flip, level, true, dcode);
                    assert_eq!(
                        decoded.ar_rad(0.485).to_bits(),
                        legacy.ar_rad(0.485).to_bits()
                    );
                    assert_eq!(rec.0 & 1023, 1023);
                }
            }
        }
    }

    #[test]
    fn both_bucket_layouts_preserve_the_complete_key() {
        for key in [0, 1, 1023, 0x1234_5678, u32::MAX] {
            for deep in [false, true] {
                let (bucket, fp) = bucket_fp_layout(key, deep);
                let bits = if deep { 10 } else { 12 };
                assert_eq!(((bucket as u32) << bits) | fp, mix32(key));
            }
        }
    }

    #[test]
    fn deep_and_legacy_files_probe_exact_keys_and_route_modes() {
        use crate::{
            api::SolveMode,
            solve::{DeepIndex, SolveConfig, solve_field},
            types::StarField,
        };
        use std::{
            fs,
            time::{SystemTime, UNIX_EPOCH},
        };

        struct Fixture(PathBuf);
        impl Drop for Fixture {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let fixture = Fixture(std::env::temp_dir().join(format!(
            "skyshot-index-layout-{}-{}", std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        )));
        let key = 0x1234_5678;
        let mut indexes = vec![];
        for deep in [false, true] {
            let dir = fixture.0.join(if deep { "index-deep" } else { "index" });
            fs::create_dir_all(&dir).unwrap();
            let buckets = 1 << if deep { DEEP_BUCKET_BITS } else { BUCKET_BITS };
            let (bucket, fp) = bucket_fp_layout(key, deep);
            assert!(fp > 0 && fp < if deep { 1023 } else { 4095 });
            let record = if deep { Record::new_deep } else { Record::new };
            let records = [
                record(fp - 1, 0, 0, 0, false, 7, true, 0),
                record(fp, 0, 511, 123, false, 7, true, 3),
                record(fp, 1, 1023, 999, true, 8, false, 7),
                record(fp + 1, 1, 0, 0, false, 8, false, 0),
            ];
            fs::write(
                dir.join("codes.bin"),
                records
                    .iter()
                    .flat_map(|r| r.0.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            fs::write(
                dir.join("keys.bin"),
                (0..=buckets)
                    .flat_map(|b| if b <= bucket { 0u32 } else { 4u32 }.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            fs::write(dir.join("stars.bin"), [0u8; 2 * STAR_BYTES]).unwrap();
            fs::write(
                dir.join("cells.bin"),
                (0..=6)
                    .flat_map(|c| if c == 0 { 0u32 } else { 2u32 }.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let manifest = Manifest {
                format: "skyanchor2-index".into(),
                version: if deep {
                    FORMAT_VERSION_DEEP
                } else {
                    FORMAT_VERSION
                },
                profile: if deep { "deep" } else { "full" }.into(),
                r7_arcmin: if deep { 0.485 } else { 1.94 },
                key_spec: String::new(),
                levels: vec![],
                n_records: 4,
                n_stars: 2,
                n_catalog: 2,
                star_cell_n: 1,
                catalog: "synthetic format fixture".into(),
                catalog_fingerprint: "fixture".into(),
                build_seconds: 0.0,
                files: vec![],
                stop_words: 0,
                key_type: "v1".into(),
                coder_sha256: None,
            };
            fs::write(
                dir.join("manifest.json"),
                serde_json::to_vec(&manifest).unwrap(),
            )
            .unwrap();
            let idx = Index::open(&dir).unwrap();
            assert_eq!(idx.n_buckets(), buckets);
            assert_eq!(idx.bucket_len(bucket), 4);
            let found: Vec<_> = idx.probe(key).collect();
            assert_eq!(found.len(), 2, "adjacent fingerprints must not match");
            for (r, (anchor, level, pa, flip, dcode)) in found
                .iter()
                .zip([(0, 7, 123, false, 3), (1, 8, 999, true, 7)])
            {
                assert_eq!(
                    (r.anchor(), r.level(), r.pa_code(), r.flip(), r.dcode()),
                    (anchor, level, pa, flip, dcode)
                );
            }
            assert_eq!(idx.probe(0).count(), 0);
            indexes.push(idx);
        }
        assert!(companion_deep_dir(&fixture.0.join("index-deep")).is_none());
        let companion = DeepIndex::companion(&fixture.0.join("index"))
            .unwrap()
            .unwrap();
        let field = StarField {
            width: 1000,
            height: 750,
            stars: vec![],
            epoch: None,
        };
        for mode in [SolveMode::Standard, SolveMode::Auto, SolveMode::Deep] {
            let sol = solve_field(
                &indexes[0],
                &field,
                &SolveConfig {
                    mode,
                    deep_index: Some(companion.clone()),
                    deadline_ms: 1000.0,
                    ..Default::default()
                },
            );
            assert_eq!(sol.status, "unsolved");
            assert_eq!(sol.stats.deep_search, mode != SolveMode::Standard);
        }
        let direct = solve_field(&indexes[1], &field, &SolveConfig::default());
        assert!(direct.stats.deep_search);
        drop(indexes);
        drop(companion);
    }

    #[test]
    fn code_bounds_match_decoding_exactly() {
        let r8 = (7.76f64 / 2.0).to_radians() / 60.0;
        let idx_like = |d_min: f64| (0..=u8::MAX).find(|&c| decode_d(r8, c) >= d_min);
        // every threshold between two codes, at a code and beyond both ends
        for c in 0..=u8::MAX {
            let m = decode_mag(c);
            assert_eq!(mag_code_max(m), Some(c), "mag threshold at code {c}");
            assert_eq!(
                mag_code_max(m.next_down()),
                if c == 0 { None } else { Some(c - 1) },
                "mag threshold just below code {c}"
            );
            let d = decode_d(r8, c);
            assert_eq!(idx_like(d), Some(c), "d threshold at code {c}");
            let d_above = d.next_up();
            assert_eq!(
                idx_like(d_above),
                if c == u8::MAX { None } else { Some(c + 1) },
                "d threshold just above code {c}"
            );
        }
        assert_eq!(mag_code_max(-2.1), None);
        assert_eq!(mag_code_max(30.0), Some(255));
        assert_eq!(idx_like(0.0), Some(0));
        assert_eq!(idx_like(100.0), None);
    }

    /// A built index to test against, if any: `$SKYSHOT_TEST_INDEX`, then `work/index-phone`,
    /// `work/index-lite`, `work/index` under this crate root. No parent checkout is searched.
    fn test_index_dir() -> Option<PathBuf> {
        if let Some(p) = std::env::var_os("SKYSHOT_TEST_INDEX") {
            return Some(PathBuf::from(p));
        }
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let bases = [root];
        for b in &bases {
            for name in ["work/index-phone", "work/index-lite", "work/index"] {
                let p = b.join(name);
                if p.join("manifest.json").exists() {
                    return Some(p);
                }
            }
        }
        None
    }

    /// Reference: the existing cell path filtered by (mag, d_rad, angle), sorted by id.
    fn reference(idx: &Index, c: Vec3, radius: f64, mag_max: f32, d_min: f64) -> Vec<BrightStar> {
        let (mut cells, mut cand) = (vec![], vec![]);
        idx.cone_candidates(c, radius, &mut cells, &mut cand);
        let cos_r = radius.min(PI).cos();
        let mut out: Vec<BrightStar> = cand
            .iter()
            .map(|&(id, cell)| (id, idx.star_in_cell(id, cell)))
            .filter(|(_, s)| s.mag <= mag_max && s.d_rad >= d_min && dot(s.v, c) >= cos_r)
            .map(|(id, s)| BrightStar {
                v: s.v,
                mag: s.mag,
                d_rad: s.d_rad,
                id,
            })
            .collect();
        out.sort_by_key(|s| s.id);
        out
    }

    #[test]
    fn solver_prefilter_preserves_decoded_candidates_and_order() {
        let Some(dir) = test_index_dir() else {
            eprintln!("skipping: no index built");
            return;
        };
        let idx = Index::open(&dir).expect("open index");
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(20261011);
        let mut deep_seen = false;
        for k in 0..24 {
            let c = unit(rng.gen_range(0.0..360.0), rng.gen_range(-89.0..89.0));
            let radius = [0.008, 0.03, 0.12][k % 3];
            let threshold = decode_d(level_radius(idx.r7, 8), [0, 16, 64, 128][k % 4]);
            let d_min = match k % 3 {
                0 => threshold.next_down(),
                1 => threshold,
                _ => threshold.next_up(),
            };
            let mag_limit = [None, Some(8.5), Some(12.0), Some(-2.1)][k % 4];
            let include_deep = k % 2 == 0;
            let (mut cells, mut ids) = (vec![], vec![]);
            idx.cone_candidates(c, radius, &mut cells, &mut ids);
            let mut want: Vec<_> = ids
                .iter()
                .map(|&(id, cell)| idx.star_in_cell(id, cell))
                .collect();
            if include_deep {
                let mut deep = vec![];
                idx.deep_cone(c, radius, &mut cells, &mut deep);
                deep_seen |= !deep.is_empty();
                want.extend(deep);
            }
            want.retain(|s| s.d_rad >= d_min && mag_limit.is_none_or(|m| s.mag <= m));
            let got = idx.filtered_cone_candidates(c, radius, d_min, mag_limit, include_deep);
            let bits = |s: &StarInfo| (s.v.map(f64::to_bits), s.d_rad.to_bits(), s.mag.to_bits());
            assert_eq!(
                got.iter().map(bits).collect::<Vec<_>>(),
                want.iter().map(bits).collect::<Vec<_>>(),
                "case {k}"
            );
        }
        if idx.has_deep() {
            assert!(deep_seen, "deep catalogue must be exercised");
        }
    }

    #[test]
    fn cached_cones_preserve_values_order_cuts_and_deep_membership() {
        let Some(dir) = test_index_dir() else {
            return;
        };
        let idx = Index::open(&dir).unwrap();
        let mut cache = ConeCache::default();
        for deep in [false, true, false] {
            for mag in [None, Some(8.5), Some(-2.1), None] {
                for d in [idx.r7, idx.r7.next_up(), idx.r7 * 0.49] {
                    for (ra, radius) in [(30.0, 0.06), (30.0, 0.06), (31.0, 0.08), (240.0, 0.4)] {
                        let c = unit(ra, -25.0);
                        let want = idx.filtered_cone_candidates(c, radius, d, mag, deep);
                        let got = idx.cached_cone_candidates(c, radius, d, mag, deep, &mut cache);
                        assert_eq!(got, want);
                        for part in [&cache.base, &cache.deep] {
                            assert!(part.cells.len() <= 4096 && part.stars.len() <= 32768);
                        }
                    }
                }
            }
        }
        cache.clear();
        assert!(cache.cuts.is_none());
        for part in [&cache.base, &cache.deep] {
            assert!(part.cells.is_empty() && part.ranges.is_empty() && part.stars.is_empty());
        }
    }

    #[test]
    fn cone_stars_equals_the_cell_path_on_random_cones() {
        let Some(dir) = test_index_dir() else {
            eprintln!("skipping: no index built");
            return;
        };
        let idx = Index::open(&dir).expect("open index");
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(20261009);
        let mut got = vec![];
        let mut got_cells = vec![];
        let mut nonempty = 0usize;
        for k in 0..50 {
            let ra = rng.gen_range(0.0..360.0);
            let dec = rng.gen_range(-1.0f64..1.0).asin().to_degrees();
            let c = unit(ra, dec);
            // radii across both paths, including the switch point and whole-sky cones
            let radius = match k % 5 {
                0 => rng.gen_range(0.01..0.1),
                1 => rng.gen_range(0.1..BRIGHT_PATH_RADIUS),
                2 => BRIGHT_PATH_RADIUS,
                3 => rng.gen_range(0.3..1.0),
                _ => rng.gen_range(1.0..1.6),
            };
            let mag_max = [4.5f32, 6.0, 7.5, 8.5, 9.5][k % 5];
            let d_min = match k % 4 {
                0 => 0.0,
                1 => idx.level_radius(7),
                2 => idx.level_radius(5),
                _ => idx.level_radius(3) * 1.001,
            };
            let want = reference(&idx, c, radius, mag_max, d_min);
            idx.cone_stars(c, radius, mag_max, d_min, &mut got);
            got.sort_by_key(|s| s.id);
            assert_eq!(
                got, want,
                "cone_stars ra={ra:.2} dec={dec:.2} r={radius:.3} mag<={mag_max} d>={d_min:.5}"
            );
            idx.cone_stars_cells(c, radius, mag_max, d_min, &mut got_cells);
            got_cells.sort_by_key(|s| s.id);
            assert_eq!(
                got_cells, want,
                "cone_stars_cells r={radius:.3} mag<={mag_max}"
            );
            nonempty += usize::from(!want.is_empty());
            for s in &want {
                assert!(angle(s.v, c) <= radius + 1e-9);
            }
        }
        assert!(nonempty >= 40, "only {nonempty} of 50 cones had stars");
        // bright table: sorted by magnitude, within the cut, prefix semantics
        let all = idx.bright_stars(f32::MAX);
        assert!(all.windows(2).all(|w| w[0].mag <= w[1].mag));
        assert!(all.iter().all(|s| s.mag <= idx.bright_cut()));
        let six = idx.bright_stars(6.0);
        assert_eq!(six.len(), all.iter().filter(|s| s.mag <= 6.0).count());
        assert!(six.iter().all(|s| s.mag <= 6.0));
    }
}
