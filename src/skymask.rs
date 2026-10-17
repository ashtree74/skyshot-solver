//! Sky mask: the parts of a frame where catalogue stars cannot be seen (ground, buildings, trees,
//! the black corners outside a fisheye's image circle). Client side, so the WebAssembly detector
//! builds the same mask as the native pipeline and sends it with the star list.
//!
//! The mask is a coarse grid of blocked cells. The solver drops detections and catalogue stars
//! that fall into blocked cells and uses the open area, not the frame area, for the chance
//! expectation of its tests, so a landscape with a lit village in the foreground is verified on
//! the sky alone. A wrong mask can cost a solve but cannot make a false one more likely: it only
//! removes evidence from both sides of the test.

use crate::{
    image_io::GrayImage,
    types::{StarField, star_flags},
};
use serde::{Deserialize, Serialize};

/// Blocked cells of a frame, row-major. In JSON the cells are packed into a hexadecimal string
/// (`cols * rows` bits, cell `i` is bit `i % 8` of byte `i / 8`), which keeps a 64 x 43 mask
/// under 700 characters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SkyMask {
    pub cols: u32,
    pub rows: u32,
    /// Cell size in pixels (x, y).
    pub cell: [f64; 2],
    /// Pixel position of the grid's top-left corner (0, 0 for a whole frame; a window of the
    /// frame shifts it).
    #[serde(default)]
    pub origin: [f64; 2],
    #[serde(with = "hex_bits")]
    pub blocked: Vec<bool>,
}

/// Largest grid accepted from a client.
pub const MAX_CELLS: u32 = 128 * 128;
/// Hidden fraction of the frame from which it is treated as a landscape (see
/// [`SkyMask::is_landscape`]).
pub const LANDSCAPE_FRACTION: f64 = 0.08;

impl SkyMask {
    /// An all-open grid of `cols` x `rows` cells over a `width` x `height` frame.
    pub fn open(width: u32, height: u32, cols: u32, rows: u32) -> Self {
        let (cols, rows) = (cols.max(1), rows.max(1));
        Self {
            cols,
            rows,
            cell: [width as f64 / cols as f64, height as f64 / rows as f64],
            origin: [0.0, 0.0],
            blocked: vec![false; (cols * rows) as usize],
        }
    }

    /// Structural check of a mask received from a client.
    pub fn validate(&self) -> Result<(), String> {
        if self.cols == 0 || self.rows == 0 || self.cols.saturating_mul(self.rows) > MAX_CELLS {
            return Err(format!("mask grid must have 1..{MAX_CELLS} cells"));
        }
        // the hex form pads the cells to whole bytes
        let n = (self.cols * self.rows) as usize;
        if self.blocked.len() < n || self.blocked.len() >= n + 8 {
            return Err("mask size does not match cols x rows".into());
        }
        if !(self.cell.iter().all(|c| c.is_finite() && *c > 0.0)
            && self.origin.iter().all(|o| o.is_finite()))
        {
            return Err("mask cell size and origin must be finite and positive".into());
        }
        Ok(())
    }

    #[inline]
    fn index(&self, x: f64, y: f64) -> Option<usize> {
        let i = ((x - self.origin[0]) / self.cell[0]).floor();
        let j = ((y - self.origin[1]) / self.cell[1]).floor();
        if i < 0.0 || j < 0.0 || i >= self.cols as f64 || j >= self.rows as f64 {
            return None;
        }
        Some(j as usize * self.cols as usize + i as usize)
    }

    /// Whether stars at pixel (x, y) are hidden. Pixels outside the grid are open.
    #[inline]
    pub fn is_blocked(&self, x: f64, y: f64) -> bool {
        self.index(x, y).is_some_and(|k| self.blocked[k])
    }

    pub fn n_blocked(&self) -> usize {
        self.blocked.iter().filter(|b| **b).count()
    }

    /// A landscape: at least [`LANDSCAPE_FRACTION`] of the grid hidden (ground, buildings).
    /// The black borders of a registered stack are a few per cent and do not count.
    pub fn is_landscape(&self) -> bool {
        let n = (self.cols * self.rows) as usize;
        n > 0
            && self.blocked[..n.min(self.blocked.len())]
                .iter()
                .filter(|b| **b)
                .count() as f64
                >= LANDSCAPE_FRACTION * n as f64
    }

    /// Blocked area (px^2) inside the rectangle [x0, x1) x [y0, y1).
    pub fn blocked_area(&self, x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
        let mut a = 0.0;
        for j in 0..self.rows as usize {
            let cy0 = self.origin[1] + j as f64 * self.cell[1];
            let oy = (cy0 + self.cell[1]).min(y1) - cy0.max(y0);
            if oy <= 0.0 {
                continue;
            }
            for i in 0..self.cols as usize {
                if !self.blocked[j * self.cols as usize + i] {
                    continue;
                }
                let cx0 = self.origin[0] + i as f64 * self.cell[0];
                let ox = (cx0 + self.cell[0]).min(x1) - cx0.max(x0);
                if ox > 0.0 {
                    a += ox * oy;
                }
            }
        }
        a
    }

    /// Bounding box (x0, y0, x1, y1) of the open cells within a `width` x `height` frame (the
    /// frame when nothing is open).
    pub fn open_bbox(&self, width: f64, height: f64) -> [f64; 4] {
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for j in 0..self.rows as usize {
            for i in 0..self.cols as usize {
                if self.blocked[j * self.cols as usize + i] {
                    continue;
                }
                let cx = self.origin[0] + i as f64 * self.cell[0];
                let cy = self.origin[1] + j as f64 * self.cell[1];
                x0 = x0.min(cx);
                y0 = y0.min(cy);
                x1 = x1.max(cx + self.cell[0]);
                y1 = y1.max(cy + self.cell[1]);
            }
        }
        if x0 > x1 {
            return [0.0, 0.0, width, height];
        }
        [x0.max(0.0), y0.max(0.0), x1.min(width), y1.min(height)]
    }

    /// Open (unblocked) fraction of the rectangle [x0, x1) x [y0, y1).
    pub fn open_fraction(&self, x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
        let area = (x1 - x0) * (y1 - y0);
        if area <= 0.0 {
            return 0.0;
        }
        (1.0 - self.blocked_area(x0, y0, x1, y1) / area).clamp(0.0, 1.0)
    }

    /// The same mask in the coordinates of a window whose top-left pixel is (x0, y0).
    pub fn shifted(&self, x0: f64, y0: f64) -> Self {
        Self {
            origin: [self.origin[0] - x0, self.origin[1] - y0],
            ..self.clone()
        }
    }
}

/// Cells along the longer side of the frame.
const GRID: f64 = 64.0;
/// Fewest blocked cells (fraction of the grid) for a mask to be reported at all.
const MIN_BLOCKED: f64 = 0.02;
/// A mask blocking more than this is taken as a failed estimate (none is reported).
const MAX_BLOCKED: f64 = 0.85;
/// Least gain (misplaced evidence explained, about one unit per clearly hidden cell) for a
/// skyline to count.
const MIN_GAIN: f64 = 8.0;

/// `SKYANCHOR_PROFILE` (as the extractor): mask diagnostics on stderr.
fn debug() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("SKYANCHOR_PROFILE").is_some())
}

/// Per-cell measurements of [`estimate`].
struct Cells {
    cols: usize,
    rows: usize,
    /// median level
    med: Vec<f32>,
    /// median absolute difference of pixels two apart along a row (texture plus noise)
    tex: Vec<f32>,
    /// star-shaped detections (frame PSF, round, unflagged)
    stars: Vec<f32>,
}

impl Cells {
    fn dead(&self, k: usize, sigma: f32, black: f32) -> bool {
        // The darkest percentile is not necessarily missing image data. A
        // denoised dust cloud can be flat and reach the frame edge too. Require
        // actual near-zero samples as well as flatness and no stellar PSF.
        self.med[k].abs() <= sigma
            && self.tex[k] < 0.25 * sigma
            && (self.med[k] - black) <= sigma
            && self.stars[k] == 0.0
    }
}

fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    let k = v.len() / 2;
    *v.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1
}

fn percentile(v: &[f32], q: f64) -> f32 {
    let mut w: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if w.is_empty() {
        return f32::NAN;
    }
    let k = ((w.len() - 1) as f64 * q).round() as usize;
    *w.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1
}

fn measure(gray: &GrayImage, field: &StarField, cell: f64, cols: usize, rows: usize) -> Cells {
    use rayon::prelude::*;
    let (w, h) = (gray.width as usize, gray.height as usize);
    // at most ~14 x 14 samples per cell (the medians need no more)
    let step = ((cell / 14.0).floor() as usize).max(1);
    let n = cols * rows;
    let per_row: Vec<(Vec<f32>, Vec<f32>)> = (0..rows)
        .into_par_iter()
        .map(|j| {
            let y0 = (j as f64 * cell) as usize;
            let y1 = (((j + 1) as f64 * cell) as usize).min(h);
            let (mut med, mut tex) = (vec![f32::NAN; cols], vec![f32::NAN; cols]);
            let mut vals = Vec::with_capacity(256);
            let mut diffs = Vec::with_capacity(256);
            for i in 0..cols {
                let x0 = (i as f64 * cell) as usize;
                let x1 = (((i + 1) as f64 * cell) as usize).min(w);
                vals.clear();
                diffs.clear();
                for y in (y0..y1).step_by(step) {
                    let row = &gray.pixels[y * w..y * w + w];
                    for x in (x0..x1).step_by(step) {
                        let p = row[x];
                        if !p.is_finite() {
                            continue;
                        }
                        vals.push(p);
                        if x + 2 < x1 && row[x + 2].is_finite() {
                            diffs.push((row[x + 2] - p).abs());
                        }
                    }
                }
                med[i] = median(&mut vals);
                tex[i] = median(&mut diffs);
            }
            (med, tex)
        })
        .collect();
    let (mut med, mut tex) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for (m, t) in per_row {
        med.extend(m);
        tex.extend(t);
    }
    // Bright landscape features can dominate aperture flux. Use the lower
    // quartile of the broad detection population, rather than its brightest
    // few hundred, so foliage does not define the stellar PSF.
    let mid: Vec<f32> = field
        .stars
        .iter()
        .skip(field.stars.len().min(30))
        .take(2000)
        .filter(|s| s.fwhm.is_finite() && s.fwhm > 0.0)
        .map(|s| s.fwhm as f32)
        .collect();
    let psf = percentile(&mid, 0.25);
    let mut stars = vec![0.0f32; n];
    for s in &field.stars {
        let round = s.elongation.is_finite() && s.elongation <= 1.5;
        let sized = !psf.is_finite() || (s.fwhm as f32 >= 0.5 * psf && s.fwhm as f32 <= 1.8 * psf);
        if !(round && sized && s.flags & star_flags::REJECT == 0) {
            continue;
        }
        let (i, j) = ((s.x / cell).floor(), (s.y / cell).floor());
        if i >= 0.0 && j >= 0.0 && (i as usize) < cols && (j as usize) < rows {
            stars[j as usize * cols + i as usize] += 1.0;
        }
    }
    Cells {
        cols,
        rows,
        med,
        tex,
        stars,
    }
}

/// Least-squares quadratic surface z(x, y) over the cells in `use_`, robust (two passes with
/// 3-sigma clipping). Returns the six coefficients of 1, x, y, x^2, xy, y^2 (x, y in [-1, 1]).
fn fit_surface(c: &Cells, use_: &[bool]) -> Option<[f64; 6]> {
    let basis = |i: usize, j: usize| {
        let x = 2.0 * (i as f64 + 0.5) / c.cols as f64 - 1.0;
        let y = 2.0 * (j as f64 + 0.5) / c.rows as f64 - 1.0;
        [1.0, x, y, x * x, x * y, y * y]
    };
    let mut keep = use_.to_vec();
    let mut coef = None;
    for _ in 0..3 {
        let mut ata = [[0.0f64; 6]; 6];
        let mut atb = [0.0f64; 6];
        let mut n = 0;
        for j in 0..c.rows {
            for i in 0..c.cols {
                let k = j * c.cols + i;
                if !keep[k] || !c.med[k].is_finite() {
                    continue;
                }
                let b = basis(i, j);
                for r in 0..6 {
                    atb[r] += b[r] * c.med[k] as f64;
                    for s in 0..6 {
                        ata[r][s] += b[r] * b[s];
                    }
                }
                n += 1;
            }
        }
        if n < 12 {
            return coef;
        }
        let a = nalgebra::Matrix6::from_fn(|r, s| ata[r][s]);
        let b = nalgebra::Vector6::from_fn(|r, _| atb[r]);
        let x = a.cholesky()?.solve(&b);
        let cf = [x[0], x[1], x[2], x[3], x[4], x[5]];
        coef = Some(cf);
        // clip outliers for the next pass
        let res: Vec<f32> = (0..c.rows * c.cols)
            .filter(|&k| keep[k] && c.med[k].is_finite())
            .map(|k| {
                let b = basis(k % c.cols, k / c.cols);
                (c.med[k] as f64 - (0..6).map(|r| cf[r] * b[r]).sum::<f64>()).abs() as f32
            })
            .collect();
        let sig = 1.4826 * percentile(&res, 0.5).max(1e-12);
        for (k, kp) in keep.iter_mut().enumerate() {
            if *kp && c.med[k].is_finite() {
                let b = basis(k % c.cols, k / c.cols);
                let r = (c.med[k] as f64 - (0..6).map(|r| cf[r] * b[r]).sum::<f64>()).abs();
                if r > 3.0 * sig as f64 {
                    *kp = false;
                }
            }
        }
    }
    coef
}

/// Estimate where stars are hidden: the ground and everything standing on it (a skyline found
/// column by column from the bottom of the frame) and the black outside of a fisheye's image
/// circle. Three cues per cell, each relative to the frame's own sky: texture (stone, grass,
/// foliage) well above the sky noise, a level well below a smooth model of the sky
/// (silhouettes), and the absence of star-shaped detections where the sky has them. `field`
/// is the default detection of the frame (its PSF and stars). `None` when nothing is hidden or
/// the estimate is implausible.
pub fn estimate(gray: &GrayImage, field: &StarField) -> Option<SkyMask> {
    let (w, h) = (gray.width as f64, gray.height as f64);
    if w < 64.0 || h < 64.0 || gray.pixels.len() < (gray.width * gray.height) as usize {
        return None;
    }
    let cell = (w.max(h) / GRID).ceil().max(4.0);
    let (cols, rows) = ((w / cell).ceil() as usize, (h / cell).ceil() as usize);
    let c = measure(gray, field, cell, cols, rows);
    let n = cols * rows;
    // sky noise: a low percentile of the texture over cells that have any
    let nz: Vec<f32> = c.tex.iter().copied().filter(|t| *t > 0.0).collect();
    let sigma = percentile(&nz, 0.2);
    if !sigma.is_finite() || sigma <= 0.0 {
        return None;
    }
    // star density of the sky: 3x3 sums, the median over cells with any star-shaped detection
    let dens: Vec<f32> = (0..n)
        .map(|k| {
            let (i, j) = ((k % cols) as isize, (k / cols) as isize);
            let mut s = 0.0;
            for dj in -1..=1 {
                for di in -1..=1 {
                    let (ii, jj) = (i + di, j + dj);
                    if ii >= 0 && jj >= 0 && (ii as usize) < cols && (jj as usize) < rows {
                        s += c.stars[jj as usize * cols + ii as usize];
                    }
                }
            }
            s
        })
        .collect();
    let starry: Vec<f32> = dens.iter().copied().filter(|d| *d > 0.0).collect();
    let dens_sky = percentile(&starry, 0.5).max(1.0);
    // a smooth model of the sky level from cells that look like sky
    let sky_like: Vec<bool> = (0..n)
        .map(|k| c.tex[k] <= 2.0 * sigma && dens[k] >= 0.5 * dens_sky)
        .collect();
    let surf = fit_surface(&c, &sky_like);
    let model = |k: usize| -> f64 {
        let Some(cf) = surf else {
            return f64::NAN;
        };
        let x = 2.0 * ((k % cols) as f64 + 0.5) / cols as f64 - 1.0;
        let y = 2.0 * ((k / cols) as f64 + 0.5) / rows as f64 - 1.0;
        let b = [1.0, x, y, x * x, x * y, y * y];
        (0..6).map(|r| cf[r] * b[r]).sum()
    };
    let resid: Vec<f32> = (0..n)
        .filter(|&k| sky_like[k])
        .map(|k| (c.med[k] as f64 - model(k)).abs() as f32)
        .collect();
    let sig_level = (1.4826 * percentile(&resid, 0.5)).max(sigma);
    let black = percentile(&c.med, 0.01);
    // signed evidence per cell, positive = hidden
    let g: Vec<f64> = (0..n)
        .map(|k| {
            let t = c.tex[k] / sigma;
            let mut s = ((t as f64 - 2.0) / 2.0).clamp(0.0, 2.0);
            if surf.is_some() && c.med[k].is_finite() {
                let dark = (model(k) - c.med[k] as f64) / sig_level as f64;
                s += ((dark - 3.0) / 3.0).clamp(0.0, 2.0);
            }
            // black and flat: outside the image circle, crushed shadows
            if c.dead(k, sigma, black) {
                s += 1.5;
            }
            // Nebulae, dust lanes and the Milky Way can be both textured and
            // darker than the fitted background. Those cues are not foreground
            // evidence where the stellar PSF is still visible. Broad lamps have
            // already been excluded from this density in `measure`.
            let stellar = (dens[k] / dens_sky).clamp(0.0, 1.5) as f64;
            s *= (1.0 - stellar).max(0.0);
            s -= stellar;
            s - 0.15
        })
        .collect();
    let mut blocked = skylines(&g, cols, rows);
    flood_dead(&c, sigma, black, &mut blocked);
    // one cell of margin around the ground where the sky is not clear: lamps and lit trees at
    // the horizon
    let grown: Vec<bool> = (0..n)
        .map(|k| {
            let (i, j) = (k % cols, k / cols);
            let near = (i > 0 && blocked[k - 1])
                || (i + 1 < cols && blocked[k + 1])
                || (j > 0 && blocked[k - cols])
                || (j + 1 < rows && blocked[k + cols]);
            blocked[k] || (near && g[k] > 0.0)
        })
        .collect();
    if debug() {
        eprintln!(
            "skymask: {cols}x{rows} cells, sigma {sigma:.4}, sky density {dens_sky:.1}, blocked {}",
            grown.iter().filter(|b| **b).count()
        );
    }
    let nb = grown.iter().filter(|b| **b).count() as f64;
    if nb < MIN_BLOCKED * n as f64 || nb > MAX_BLOCKED * n as f64 {
        return None;
    }
    Some(SkyMask {
        cols: cols as u32,
        rows: rows as u32,
        cell: [cell, cell],
        origin: [0.0, 0.0],
        blocked: grown,
    })
}

/// Ground from the bottom of the frame: per column a boundary row b (rows >= b hidden)
/// minimising the misplaced evidence, with a capped penalty on jumps between columns (a tower
/// is one jump up and one down). Dynamic programming over columns.
fn skyline(g: &[f64], cols: usize, rows: usize) -> (Vec<bool>, f64) {
    const JUMP: f64 = 0.35;
    const JUMP_CAP: f64 = 2.0;
    // cost of boundary b in column i: positive evidence above it, negative evidence below it
    let col_cost = |i: usize| -> Vec<f64> {
        let mut above = vec![0.0; rows + 1];
        for j in 0..rows {
            above[j + 1] = above[j] + g[j * cols + i].max(0.0);
        }
        let mut below = vec![0.0; rows + 1];
        for j in (0..rows).rev() {
            below[j] = below[j + 1] + (-g[j * cols + i]).max(0.0);
        }
        (0..=rows).map(|b| above[b] + below[b]).collect()
    };
    let mut cost = col_cost(0);
    let mut back: Vec<Vec<usize>> = vec![vec![0; rows + 1]; cols];
    for (i, bk) in back.iter_mut().enumerate().skip(1) {
        let cc = col_cost(i);
        let mut next = vec![f64::INFINITY; rows + 1];
        for b in 0..=rows {
            for (p, &pc) in cost.iter().enumerate() {
                let jump = (JUMP * (b as f64 - p as f64).abs()).min(JUMP_CAP);
                let v = pc + jump;
                if v < next[b] {
                    next[b] = v;
                    bk[b] = p;
                }
            }
            next[b] += cc[b];
        }
        cost = next;
    }
    let mut b = (0..=rows)
        .min_by(|&x, &y| cost[x].total_cmp(&cost[y]))
        .unwrap_or(rows);
    let total = cost[b];
    let mut blocked = vec![false; cols * rows];
    for i in (0..cols).rev() {
        for j in b..rows {
            blocked[j * cols + i] = true;
        }
        b = back[i][b];
    }
    (blocked, total)
}

/// The grid seen with border `o` at the bottom (0 bottom, 1 left, 2 top, 3 right): its size
/// and, per rotated cell (row-major), the original cell index.
fn rotation(cols: usize, rows: usize, o: usize) -> (usize, usize, Vec<usize>) {
    let (c2, r2) = if o.is_multiple_of(2) {
        (cols, rows)
    } else {
        (rows, cols)
    };
    let map = (0..c2 * r2)
        .map(|k| {
            let (i2, j2) = (k % c2, k / c2);
            let (i, j) = match o {
                0 => (i2, j2),
                1 => (cols - 1 - j2, i2),
                2 => (cols - 1 - i2, rows - 1 - j2),
                _ => (j2, rows - 1 - i2),
            };
            j * cols + i
        })
        .collect();
    (c2, r2, map)
}

/// Ground from one border: the skyline of the orientation that explains the most evidence (a
/// camera rolled by 90 deg has the ground on a side), when it explains clearly more than no
/// ground at all. One orientation only: ground along the bottom also "explains" evidence when
/// seen from a side, so a union would hide sky.
fn skylines(g: &[f64], cols: usize, rows: usize) -> Vec<bool> {
    // gain of an orientation: the misplaced evidence of "all sky" minus that of its skyline
    let none: f64 = g.iter().map(|v| v.max(0.0)).sum();
    let mut found: Vec<(f64, Vec<bool>)> = (0..4)
        .map(|o| {
            let (c2, r2, map) = rotation(cols, rows, o);
            let gr: Vec<f64> = map.iter().map(|&k| g[k]).collect();
            let (b, cost) = skyline(&gr, c2, r2);
            // A foreground skyline separates two sides of the frame. An
            // edge-attached nebula or galaxy is not enough; nor is a dark rim
            // surrounding a star field. Check every orientation before choosing
            // the winner, otherwise an implausible winner hides a valid skyline.
            let ground_edge = b[(r2 - 1) * c2..].iter().filter(|v| **v).count();
            let sky_edge = b[..c2].iter().filter(|v| **v).count();
            let ground_edge_hidden = gr[(r2 - 1) * c2..].iter().filter(|v| **v > 0.0).count();
            let sky_edge_hidden = gr[..c2].iter().filter(|v| **v > 0.0).count();
            let coherent = ground_edge * 5 >= c2 * 3
                && sky_edge * 10 <= c2
                && ground_edge_hidden * 2 >= c2
                && sky_edge_hidden * 2 <= c2;
            if debug() {
                eprintln!("skymask: border {o}, ground {ground_edge}/{c2}, sky {sky_edge}/{c2}, hidden sky {sky_edge_hidden}/{c2}, gain {:.1}, coherent {coherent}", none - cost);
            }
            let mut blocked = vec![false; cols * rows];
            for (k2, &k) in map.iter().enumerate() {
                blocked[k] = b[k2];
            }
            (if coherent { none - cost } else { 0.0 }, blocked)
        })
        .collect();
    if debug() {
        let gains: Vec<String> = found.iter().map(|f| format!("{:.1}", f.0)).collect();
        eprintln!(
            "skymask: skyline gains bottom/left/top/right {}",
            gains.join(" / ")
        );
    }
    // the upright orientation wins ties (bottom first in `found`, stable sort)
    found.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (gain, best) = found.swap_remove(0);
    if gain >= MIN_GAIN {
        best
    } else {
        vec![false; cols * rows]
    }
}

/// Black, flat cells connected to the border (outside a fisheye's image circle, shadows at the
/// edge of the frame) are hidden too.
fn flood_dead(c: &Cells, sigma: f32, black: f32, blocked: &mut [bool]) {
    let (cols, rows) = (c.cols, c.rows);
    let dead = |k: usize| c.dead(k, sigma, black);
    let mut stack: Vec<usize> = (0..cols * rows)
        .filter(|&k| {
            let (i, j) = (k % cols, k / cols);
            (i == 0 || j == 0 || i + 1 == cols || j + 1 == rows) && dead(k)
        })
        .collect();
    let mut seen = vec![false; cols * rows];
    while let Some(k) = stack.pop() {
        if seen[k] {
            continue;
        }
        seen[k] = true;
        blocked[k] = true;
        let (i, j) = (k % cols, k / cols);
        let mut push = |ii: usize, jj: usize| {
            let kk = jj * cols + ii;
            if !seen[kk] && dead(kk) {
                stack.push(kk);
            }
        };
        if i > 0 {
            push(i - 1, j);
        }
        if i + 1 < cols {
            push(i + 1, j);
        }
        if j > 0 {
            push(i, j - 1);
        }
        if j + 1 < rows {
            push(i, j + 1);
        }
    }
}

mod hex_bits {
    use serde::{Deserialize, Deserializer, Serializer, de::Error};

    pub fn serialize<S: Serializer>(bits: &[bool], s: S) -> Result<S::Ok, S::Error> {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(bits.len().div_ceil(8) * 2);
        for chunk in bits.chunks(8) {
            let byte = chunk
                .iter()
                .enumerate()
                .fold(0u8, |b, (i, &on)| b | ((on as u8) << i));
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 15) as usize] as char);
        }
        s.serialize_str(&out)
    }

    /// Bits beyond `cols * rows` are dropped by the caller's length check; here every hex
    /// digit becomes four bits.
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<bool>, D::Error> {
        let s = String::deserialize(d)?;
        if s.len() % 2 != 0 || s.len() > 2 * (super::MAX_CELLS as usize).div_ceil(8) {
            return Err(D::Error::custom("bad mask length"));
        }
        let mut bits = Vec::with_capacity(s.len() * 4);
        for pair in s.as_bytes().chunks(2) {
            let v = |c: u8| match c {
                b'0'..=b'9' => Ok(c - b'0'),
                b'a'..=b'f' => Ok(c - b'a' + 10),
                b'A'..=b'F' => Ok(c - b'A' + 10),
                _ => Err(D::Error::custom("bad mask digit")),
            };
            let byte = (v(pair[0])? << 4) | v(pair[1])?;
            bits.extend((0..8).map(|i| byte >> i & 1 == 1));
        }
        Ok(bits)
    }
}

impl SkyMask {
    /// The hex form pads the cells to whole bytes; drop the padding.
    pub fn trimmed(mut self) -> Self {
        self.blocked.truncate((self.cols * self.rows) as usize);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::{ExtractOptions, extract};

    /// A night photograph: sky with a gradient, noise and stars; optionally a ground below a
    /// hilly skyline (dark, textured, with a tower and bright lamps).
    fn photo(w: usize, h: usize, ground: bool) -> GrayImage {
        let mut q = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = move || {
            q ^= q << 13;
            q ^= q >> 7;
            q ^= q << 17;
            (q >> 11) as f32 / (1u64 << 53) as f32
        };
        let skyline = |x: usize| {
            let t = x as f32 / w as f32;
            let hill = 0.62 + 0.08 * (t * 9.0).sin() + 0.04 * (t * 23.0).cos();
            // a tower on the right
            if (0.7..0.8).contains(&t) { 0.35 } else { hill }
        };
        let mut px = vec![0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let sky = 0.18 + 0.06 * y as f32 / h as f32;
                let below = ground && (y as f32) > skyline(x) * h as f32;
                px[y * w + x] = if below {
                    0.06 + 0.06 * rnd()
                } else {
                    sky + 0.008 * (rnd() - 0.5)
                };
            }
        }
        let mut blob = |cx: f32, cy: f32, a: f32, s: f32| {
            for y in (cy - 6.0).max(0.0) as usize..((cy + 7.0) as usize).min(h) {
                for x in (cx - 6.0).max(0.0) as usize..((cx + 7.0) as usize).min(w) {
                    let r2 = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
                    px[y * w + x] += a * (-r2 / (2.0 * s * s)).exp();
                }
            }
        };
        for _ in 0..900 {
            let (x, y) = (rnd() * w as f32, rnd() * h as f32);
            if !ground || y < skyline(x as usize) * h as f32 - 3.0 {
                blob(x, y, 0.05 + 0.4 * rnd().powi(3), 1.1);
            }
        }
        if ground {
            for _ in 0..40 {
                let x = rnd() * w as f32;
                let y = (skyline(x as usize) + 0.05 + 0.25 * rnd()) * h as f32;
                blob(x, y.min(h as f32 - 4.0), 0.9, 2.5);
            }
        }
        GrayImage {
            width: w as u32,
            height: h as u32,
            pixels: px,
            epoch: None,
        }
    }

    #[test]
    fn finds_the_ground_and_leaves_the_sky_open() {
        let (w, h) = (900, 600);
        let img = photo(w, h, true);
        let field = extract(&img, &ExtractOptions::default());
        let m = estimate(&img, &field).expect("a landscape has a mask");
        assert!(m.is_landscape());
        // cells well inside the ground blocked, cells well inside the sky open
        let (mut ground_open, mut ground_n, mut sky_blocked, mut sky_n) = (0, 0, 0, 0);
        for j in 0..m.rows as usize {
            for i in 0..m.cols as usize {
                let (x, y) = ((i as f64 + 0.5) * m.cell[0], (j as f64 + 0.5) * m.cell[1]);
                let t = x / w as f64;
                let hill = 0.62 + 0.08 * (t * 9.0).sin() + 0.04 * (t * 23.0).cos();
                let top = if (0.7..0.8).contains(&t) { 0.35 } else { hill } * h as f64;
                if y > top + 2.0 * m.cell[1] && !(0.68..0.82).contains(&t) {
                    ground_n += 1;
                    ground_open += !m.blocked[j * m.cols as usize + i] as usize;
                } else if y < top - 3.0 * m.cell[1] {
                    sky_n += 1;
                    sky_blocked += m.blocked[j * m.cols as usize + i] as usize;
                }
            }
        }
        assert!(
            ground_open * 20 <= ground_n,
            "ground open {ground_open}/{ground_n}"
        );
        assert!(
            sky_blocked * 50 <= sky_n,
            "sky blocked {sky_blocked}/{sky_n}"
        );
        // the lamps are hidden
        let lamps = field.stars.iter().filter(|s| s.flux > 0.0).take(10);
        assert!(lamps.filter(|s| !m.is_blocked(s.x, s.y)).count() <= 3);
    }

    #[test]
    fn a_sky_without_ground_has_no_mask() {
        let img = photo(900, 600, false);
        let field = extract(&img, &ExtractOptions::default());
        assert!(field.stars.len() > 300);
        assert_eq!(estimate(&img, &field), None);
    }

    #[test]
    fn structured_nebulosity_with_stars_is_not_foreground() {
        let mut img = photo(900, 600, false);
        let field = extract(&img, &ExtractOptions::default());
        // A bright, textured nebula reaching the lower edge, with the same
        // stellar PSF visible through it. Texture alone used to outweigh stars.
        for y in 300..600 {
            for x in 0..900 {
                img.pixels[y * 900 + x] += 0.12 + 0.04 * ((x * 17 + y * 13) % 7) as f32;
            }
        }
        assert_eq!(estimate(&img, &field), None);
    }

    #[test]
    fn an_edge_attached_dark_cloud_is_not_a_landscape() {
        let (cols, rows) = (64, 40);
        let mut evidence = vec![-1.0; cols * rows];
        for y in 15..rows {
            for x in 20..40 {
                evidence[y * cols + x] = 2.0;
            }
        }
        assert!(skylines(&evidence, cols, rows).iter().all(|v| !v));
    }

    #[test]
    fn foreground_with_some_starry_edge_columns_remains_detectable() {
        let (cols, rows) = (64, 40);
        let mut evidence = vec![-1.0; cols * rows];
        // Reflected stars and small lamps can leave part of the terrain edge
        // unconvincing, but a majority of real foreground still anchors it.
        for y in 25..rows {
            for x in 0..40 {
                evidence[y * cols + x] = 2.0;
            }
        }
        let mask = skylines(&evidence, cols, rows);
        assert!(mask[35 * cols + 20]);
        assert!(!mask[5 * cols + 20]);
    }

    #[test]
    fn flat_nonzero_dust_is_not_dead_image_data() {
        let mut img = photo(900, 600, false);
        // Denoised dark sky around the frame, not missing pixels. No stars in
        // this rim: absence of stars and flatness must not be enough to mask it.
        for y in 0..600 {
            for x in 0..900 {
                if !(30..870).contains(&x) || !(30..570).contains(&y) {
                    img.pixels[y * 900 + x] = 0.17;
                }
            }
        }
        let field = extract(&img, &ExtractOptions::default());
        assert_eq!(estimate(&img, &field), None);
    }

    #[test]
    fn black_fisheye_corners_still_block_without_a_landscape() {
        let mut img = photo(600, 600, false);
        for y in 0..600 {
            for x in 0..600 {
                if (x as f32 - 300.).hypot(y as f32 - 300.) > 290. {
                    img.pixels[y * 600 + x] = 0.;
                }
            }
        }
        let field = extract(&img, &ExtractOptions::default());
        let mask = estimate(&img, &field).expect("the image circle has a mask");
        for (x, y) in [(10., 10.), (590., 10.), (10., 590.), (590., 590.)] {
            assert!(mask.is_blocked(x, y));
        }
        assert!(!mask.is_blocked(300., 300.));
    }

    #[test]
    fn panoramic_and_rolled_landscapes_preserve_starry_sky() {
        let img = photo(2400, 480, true);
        for o in 0..4 {
            let (width, height, map) = rotation(2400, 480, o);
            let rotated = GrayImage {
                width: width as u32,
                height: height as u32,
                pixels: map.iter().map(|&k| img.pixels[k]).collect(),
                epoch: None,
            };
            let field = extract(&rotated, &ExtractOptions::default());
            let mask = estimate(&rotated, &field).expect("a panorama has a mask");
            let at = |x: usize, y: usize| map.iter().position(|&k| k == y * 2400 + x).unwrap();
            for (x, y, blocked) in [(400, 430, true), (400, 60, false)] {
                let k = at(x, y);
                assert_eq!(
                    mask.is_blocked((k % width) as f64, (k / width) as f64),
                    blocked,
                    "orientation {o}"
                );
            }
        }
    }

    #[test]
    fn bright_broad_foreground_does_not_define_the_stellar_psf() {
        let img = photo(900, 600, true);
        let mut field = extract(&img, &ExtractOptions::default());
        for i in 0..600 {
            field.stars.push(crate::types::Star {
                x: (i * 127 % 900) as f64,
                y: 460. + (i * 37 % 130) as f64,
                flux: 100.,
                fwhm: 9.,
                elongation: 1.,
                ..Default::default()
            });
        }
        field.stars.sort_by(|a, b| b.flux.total_cmp(&a.flux));
        let mask =
            estimate(&img, &field).expect("foreground must not erase the sky/ground distinction");
        assert!(mask.is_blocked(400., 570.));
        assert!(!mask.is_blocked(400., 100.));
    }

    #[test]
    fn json_round_trip_and_geometry() {
        let mut m = SkyMask::open(2000, 1333, 64, 43);
        for j in 30..43 {
            for i in 0..64 {
                m.blocked[j * 64 + i] = true;
            }
        }
        let s = serde_json::to_string(&m).unwrap();
        assert!(s.len() < 800, "{}", s.len());
        let back: SkyMask = serde_json::from_str::<SkyMask>(&s).unwrap().trimmed();
        assert_eq!(back, m);
        assert!(back.validate().is_ok());
        assert!(m.is_blocked(10.0, 1300.0));
        assert!(!m.is_blocked(10.0, 100.0));
        assert!(!m.is_blocked(-5.0, 1300.0));
        let open = m.open_fraction(0.0, 0.0, 2000.0, 1333.0);
        assert!((open - 30.0 / 43.0).abs() < 1e-9, "{open}");
        // a window from (500, 900): its (10, 400) is the frame's (510, 1300)
        let w = m.shifted(500.0, 900.0);
        assert!(w.is_blocked(10.0, 400.0));
        assert!(!w.is_blocked(10.0, -800.0));
    }

    #[test]
    fn rejects_malformed_masks() {
        let mut m = SkyMask::open(100, 100, 4, 4);
        m.blocked.pop();
        assert!(m.validate().is_err());
        assert!(
            serde_json::from_str::<SkyMask>(r#"{"cols":1,"rows":1,"cell":[1,1],"blocked":"zz"}"#)
                .is_err()
        );
        let big = SkyMask::open(100, 100, 1000, 1000);
        assert!(big.validate().is_err());
    }
}
