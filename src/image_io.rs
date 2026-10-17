use anyhow::{Context, Result, bail, ensure};
use image::{DynamicImage, ImageDecoder, ImageReader};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    io::{BufRead, Cursor, Seek},
    path::Path,
};

pub const MAX_PIXELS: usize = 100_000_000;
/// Longest side of [`GrayImage::preview`] in pixels.
pub const PREVIEW_MAX_SIDE: u32 = 1600;
/// Grey level (0–1) the sky background is stretched to in [`GrayImage::preview`].
const PREVIEW_BACKGROUND: f32 = 0.15;

pub struct GrayImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<f32>,
    pub epoch: Option<f64>,
}
impl GrayImage {
    /// Size of [`GrayImage::preview`]: the frame scaled down (never up) so that its longer side
    /// is at most [`PREVIEW_MAX_SIDE`]. Preview pixel `i` covers source pixels
    /// `[i * width / w, (i + 1) * width / w)`.
    pub fn preview_size(&self) -> (u32, u32) {
        let factor = (self.width.max(self.height) as f64 / PREVIEW_MAX_SIDE as f64).max(1.0);
        (
            ((self.width as f64 / factor).round() as u32).max(1),
            ((self.height as f64 / factor).round() as u32).max(1),
        )
    }

    /// 8-bit greyscale PNG preview of at most [`PREVIEW_MAX_SIDE`] px. Downscaling averages
    /// whole pixel blocks (so stars survive it), then an automatic asinh stretch maps the
    /// sky background (median) to ~15 % grey (dark enough for night use), with the black point
    /// ~2.8 sigma of the sky noise below it and the white point at the star cores (99.99th
    /// percentile, at least 25 sigma above the sky).
    pub fn preview(&self) -> Result<Vec<u8>> {
        let (w, h) = self.preview_size();
        let small = self.box_downsample(w, h);
        let step = (small.len() / 200_000).max(1);
        let mut sample: Vec<f32> = small
            .iter()
            .step_by(step)
            .copied()
            .filter(|x| x.is_finite())
            .collect();
        ensure!(!sample.is_empty(), "No finite pixels");
        let med = quantile(&mut sample, 0.5);
        let floor = quantile(&mut sample, 0.001);
        let top = quantile(&mut sample, 0.9999);
        // Sky noise: the spread of the frame (MAD) is inflated by gradients (vignetting, light
        // pollution), the pixel-to-pixel difference underestimates smoothed (noise-reduced)
        // data; their geometric mean behaves for both and equals either for white noise.
        let mut dev: Vec<f32> = sample.iter().map(|x| (x - med).abs()).collect();
        let spread = 1.4826 * quantile(&mut dev, 0.5);
        let mut diff: Vec<f32> = small
            .chunks_exact(w as usize)
            .step_by((h as usize * w as usize / 200_000).max(1))
            .flat_map(|row| row.windows(2).map(|p| (p[1] - p[0]).abs()))
            .filter(|d| d.is_finite())
            .collect();
        let local = if diff.is_empty() {
            0.0
        } else {
            1.4826 * quantile(&mut diff, 0.5) / std::f32::consts::SQRT_2
        };
        let sigma = if local > 0.0 && local < spread {
            (local * spread).sqrt()
        } else {
            spread
        };
        let lo = (med - 2.8 * sigma).max(floor);
        // white point: star cores, but never inside the noise of a nearly empty frame
        let hi = top.max(med + 25.0 * sigma).max(lo + 1e-6);
        let k = asinh_strength((med - lo) / (hi - lo), PREVIEW_BACKGROUND);
        let norm = 255.0 / k.asinh();
        let mut gray = vec![0u8; small.len()];
        gray.par_chunks_mut(w as usize)
            .zip(small.par_chunks(w as usize))
            .for_each(|(out, row)| {
                for (o, &p) in out.iter_mut().zip(row) {
                    let t = if p.is_finite() {
                        ((p - lo) / (hi - lo)).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    *o = ((k * t).asinh() * norm).round().clamp(0.0, 255.0) as u8;
                }
            });
        let img = image::GrayImage::from_raw(w, h, gray).context("preview buffer")?;
        let mut buf = Cursor::new(Vec::with_capacity(small.len()));
        img.write_to(&mut buf, image::ImageFormat::Png)?;
        Ok(buf.into_inner())
    }

    /// Area-average downscale to `w` x `h` (NaN pixels are skipped; all-NaN blocks stay NaN).
    fn box_downsample(&self, w: u32, h: u32) -> Vec<f32> {
        let (sw, sh) = (self.width as usize, self.height as usize);
        let (w, h) = (w as usize, h as usize);
        let bounds = |i: usize, src: usize, dst: usize| {
            let a = i * src / dst;
            (a, ((i + 1) * src / dst).clamp(a + 1, src))
        };
        let mut out = vec![0f32; w * h];
        out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let (y0, y1) = bounds(y, sh, h);
            for (x, o) in row.iter_mut().enumerate() {
                let (x0, x1) = bounds(x, sw, w);
                let (mut sum, mut n) = (0f64, 0u32);
                for yy in y0..y1 {
                    for &p in &self.pixels[yy * sw + x0..yy * sw + x1] {
                        if p.is_finite() {
                            sum += p as f64;
                            n += 1;
                        }
                    }
                }
                *o = if n > 0 {
                    (sum / n as f64) as f32
                } else {
                    f32::NAN
                };
            }
        });
        out
    }
}

/// `f`-quantile of `v` (reorders `v`; `v` must be non-empty).
fn quantile(v: &mut [f32], f: f64) -> f32 {
    let k = ((v.len() - 1) as f64 * f).round() as usize;
    *v.select_nth_unstable_by(k, f32::total_cmp).1
}

/// Strength `k` of `asinh(k t) / asinh(k)` that maps `t` to `target` (k -> 0 is linear).
fn asinh_strength(t: f32, target: f32) -> f32 {
    let y = |k: f32| (k * t).asinh() / k.asinh();
    let (mut lo, mut hi) = (1e-3f32.ln(), 1e5f32.ln());
    if t.is_nan() || t <= 0.0 || y(hi.exp()) <= target {
        return hi.exp();
    }
    if y(lo.exp()) >= target {
        return lo.exp();
    }
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if y(mid.exp()) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (0.5 * (lo + hi)).exp()
}

/// Container of a decoded image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceFormat {
    Fits,
    Png,
    Jpeg,
    Tiff,
    Other,
}

/// What the decoder knows about the source samples.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct SourceInfo {
    pub format: SourceFormat,
    /// FITS-style sample type: 8, 16, 32, 64 (integers) or -32, -64 (floats).
    pub bitpix: i32,
    /// Quantization step of [`GrayImage::pixels`] (1/255 for 8-bit sources,
    /// |BSCALE| for integer FITS, 0 for floating-point samples).
    pub lsb: f32,
    /// Optional FITS astrometry; never trusted as a solution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hint: Option<crate::hints::SolveHint>,
}

pub fn load(path: &Path) -> Result<GrayImage> {
    load_with_info(path).map(|(image, _)| image)
}

/// Decode an image file and report its sample format. The format matters to the
/// extractor: posterized 8-bit skies need a quantization-aware noise floor.
#[cfg(unix)]
pub fn load_with_info(path: &Path) -> Result<(GrayImage, SourceInfo)> {
    let f = std::fs::File::open(path)?;
    load_file_with_info(&f)
}

/// Decode an already opened file. Desktop file grants retain the verified handle so a
/// path/symlink replacement cannot change the input between authorization and decoding.
#[cfg(unix)]
pub fn load_file_with_info(f: &std::fs::File) -> Result<(GrayImage, SourceInfo)> {
    let mut magic = [0u8; 8];
    Source::read_at(f, &mut magic, 0)?;
    if &magic == b"SIMPLE  " {
        return load_fits(f);
    }
    use std::io::{Seek, SeekFrom};
    let mut file = f.try_clone()?;
    file.seek(SeekFrom::Start(0))?;
    decode_image(
        ImageReader::new(std::io::BufReader::new(file)).with_guessed_format()?,
        format_of(&magic),
    )
}

#[cfg(not(unix))]
pub fn load_file_with_info(f: &std::fs::File) -> Result<(GrayImage, SourceInfo)> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = f.try_clone()?;
    file.seek(SeekFrom::Start(0))?;
    let mut bytes = Vec::new();
    file.take(300 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
    anyhow::ensure!(
        bytes.len() <= 300 * 1024 * 1024,
        "File exceeds desktop upload limit"
    );
    decode_bytes(&bytes)
}

/// Decode an image file (read whole: no positional reads on this platform).
#[cfg(not(unix))]
pub fn load_with_info(path: &Path) -> Result<(GrayImage, SourceInfo)> {
    decode_bytes(&std::fs::read(path)?)
}

/// Decode an image held in memory (the browser detector): the same decoders and conversions
/// as [`load_with_info`], so the same file gives the same pixels.
pub fn decode_bytes(bytes: &[u8]) -> Result<(GrayImage, SourceInfo)> {
    if bytes.starts_with(b"SIMPLE  ") {
        return load_fits(bytes);
    }
    decode_image(
        ImageReader::new(Cursor::new(bytes)).with_guessed_format()?,
        format_of(bytes),
    )
}

/// Container by magic number.
fn format_of(magic: &[u8]) -> SourceFormat {
    if magic.starts_with(&[0x89, b'P', b'N', b'G']) {
        SourceFormat::Png
    } else if magic.starts_with(&[0xFF, 0xD8, 0xFF]) {
        SourceFormat::Jpeg
    } else if magic.starts_with(b"II*\0") || magic.starts_with(b"MM\0*") {
        SourceFormat::Tiff
    } else {
        SourceFormat::Other
    }
}

/// PNG, JPEG, TIFF (whatever `image` recognises) to luma, EXIF orientation applied.
fn decode_image<R: BufRead + Seek>(
    mut reader: ImageReader<R>,
    format: SourceFormat,
) -> Result<(GrayImage, SourceInfo)> {
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(50000);
    limits.max_image_height = Some(50000);
    limits.max_alloc = Some(1_600_000_000);
    reader.limits(limits);
    let mut decoder = reader.into_decoder()?;
    let (w, h) = decoder.dimensions();
    ensure!(
        w as usize * h as usize <= MAX_PIXELS,
        "Image exceeds 100 MP limit"
    );
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(decoder)?;
    img.apply_orientation(orientation);
    let (width, height) = (img.width(), img.height());
    let (pixels, bitpix, lsb) = luma_plane(img);
    Ok((
        GrayImage {
            width,
            height,
            pixels,
            epoch: None,
        },
        SourceInfo {
            format,
            bitpix,
            lsb,
            hint: None,
        },
    ))
}

/// Value of the `ROWORDER` card (`TOP-DOWN` / `BOTTOM-UP`) in the primary header of a FITS
/// file, upper-cased; `src` holds the start of the file (64 header blocks are read at most).
pub fn fits_row_order(src: &[u8]) -> Option<String> {
    if !src.starts_with(b"SIMPLE  ") {
        return None;
    }
    for block in src.chunks_exact(2880).take(64) {
        for card in block.chunks_exact(80) {
            match &card[..8] {
                b"END     " => return None,
                b"ROWORDER" if card[8] == b'=' => {
                    let value = String::from_utf8_lossy(&card[10..]);
                    let value = value.split('/').next()?.trim().trim_matches('\'').trim();
                    return Some(value.to_ascii_uppercase());
                }
                _ => {}
            }
        }
    }
    None
}

/// Positional reads of an image source: a file (pread, no shared cursor, so bands decode in
/// parallel) or bytes in memory.
trait Source: Sync {
    fn size(&self) -> Result<u64>;
    fn read_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<()>;
}

#[cfg(unix)]
impl Source for std::fs::File {
    fn size(&self) -> Result<u64> {
        Ok(self.metadata()?.len())
    }
    fn read_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
        std::os::unix::fs::FileExt::read_exact_at(self, buf, offset)
    }
}

impl Source for [u8] {
    fn size(&self) -> Result<u64> {
        Ok(self.len() as u64)
    }
    fn read_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
        let src = usize::try_from(offset)
            .ok()
            .and_then(|a| self.get(a..a.checked_add(buf.len())?))
            .ok_or(std::io::ErrorKind::UnexpectedEof)?;
        buf.copy_from_slice(src);
        Ok(())
    }
}

const LUMA_709: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Channel weights for colour inputs. Default Rec.709 luma; `SA2_RGB_WEIGHTS=r,g,b` overrides
/// (experiment: a Gaia-BP-like mix weights green and blue).
fn rgb_weights() -> [f32; 3] {
    static W: std::sync::OnceLock<[f32; 3]> = std::sync::OnceLock::new();
    *W.get_or_init(|| {
        std::env::var("SA2_RGB_WEIGHTS")
            .ok()
            .and_then(|v| {
                let p: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                (p.len() == 3 && p.iter().sum::<f32>() > 0.0).then(|| {
                    let t: f32 = p.iter().sum();
                    [p[0] / t, p[1] / t, p[2] / t]
                })
            })
            .unwrap_or(LUMA_709)
    })
}

/// Rec.709 luma in [0, 1] computed straight from the native sample type, in
/// parallel (no intermediate RGBA f32 copy). Fully transparent pixels are NaN.
fn luma_plane(img: DynamicImage) -> (Vec<f32>, i32, f32) {
    fn convert<T: Copy + Sync, const C: usize>(
        raw: &[T],
        scale: f32,
        to_f32: impl Fn(T) -> f32 + Sync,
    ) -> Vec<f32> {
        const CHUNK: usize = 1 << 16;
        let mut out = vec![0f32; raw.len() / C];
        out.par_chunks_mut(CHUNK)
            .zip(raw.par_chunks(C * CHUNK))
            .for_each(|(out, raw)| {
                for (o, p) in out.iter_mut().zip(raw.chunks_exact(C)) {
                    *o = match C {
                        1 => to_f32(p[0]) * scale,
                        2 => {
                            if to_f32(p[1]) <= 0.0 {
                                f32::NAN
                            } else {
                                to_f32(p[0]) * scale
                            }
                        }
                        _ => {
                            if C == 4 && to_f32(p[C - 1]) <= 0.0 {
                                f32::NAN
                            } else {
                                (rgb_weights()[0] * to_f32(p[0])
                                    + rgb_weights()[1] * to_f32(p[1])
                                    + rgb_weights()[2] * to_f32(p[2]))
                                    * scale
                            }
                        }
                    }
                }
            });
        out
    }
    const B8: f32 = 1.0 / 255.0;
    const B16: f32 = 1.0 / 65535.0;
    let u8f = |v: u8| v as f32;
    let u16f = |v: u16| v as f32;
    let f32f = |v: f32| v;
    match img {
        DynamicImage::ImageLuma8(b) => (convert::<_, 1>(b.as_raw(), B8, u8f), 8, B8),
        DynamicImage::ImageLumaA8(b) => (convert::<_, 2>(b.as_raw(), B8, u8f), 8, B8),
        DynamicImage::ImageRgb8(b) => (convert::<_, 3>(b.as_raw(), B8, u8f), 8, B8),
        DynamicImage::ImageRgba8(b) => (convert::<_, 4>(b.as_raw(), B8, u8f), 8, B8),
        DynamicImage::ImageLuma16(b) => (convert::<_, 1>(b.as_raw(), B16, u16f), 16, B16),
        DynamicImage::ImageLumaA16(b) => (convert::<_, 2>(b.as_raw(), B16, u16f), 16, B16),
        DynamicImage::ImageRgb16(b) => (convert::<_, 3>(b.as_raw(), B16, u16f), 16, B16),
        DynamicImage::ImageRgba16(b) => (convert::<_, 4>(b.as_raw(), B16, u16f), 16, B16),
        DynamicImage::ImageRgb32F(b) => (convert::<_, 3>(b.as_raw(), 1.0, f32f), -32, 0.0),
        DynamicImage::ImageRgba32F(b) => (convert::<_, 4>(b.as_raw(), 1.0, f32f), -32, 0.0),
        other => {
            let rgba = other.to_rgba32f();
            (convert::<_, 4>(rgba.as_raw(), 1.0, f32f), -32, 0.0)
        }
    }
}

/// Rows per decode band: about 1 MB of source bytes, at least one row.
fn band_rows(width: usize, bpp: usize) -> usize {
    ((1 << 20) / (width * bpp).max(1)).max(1)
}

/// How one FITS plane maps to pixel values.
#[derive(Clone, Copy)]
struct FitsScaling {
    bitpix: i64,
    scale: f64,
    zero: f64,
    blank: Option<i64>,
}

/// Convert one band of big-endian FITS samples; `weight` and `accumulate`
/// fold RGB cubes into luma without a second pass.
fn convert_fits_band(src: &[u8], out: &mut [f32], f: FitsScaling, weight: f32, accumulate: bool) {
    #[inline(always)]
    fn put(o: &mut f32, v: f32, weight: f32, accumulate: bool) {
        if accumulate {
            *o += v * weight;
        } else {
            *o = v * weight;
        }
    }
    let (s, z) = (f.scale as f32, f.zero as f32);
    macro_rules! ints {
        ($t:ty, $n:expr, $wide:expr) => {{
            let blank = f.blank.and_then(|b| <$t>::try_from(b).ok());
            for (o, c) in out.iter_mut().zip(src.chunks_exact($n)) {
                let raw = <$t>::from_be_bytes(c.try_into().unwrap());
                let v = if blank == Some(raw) {
                    f32::NAN
                } else if $wide {
                    (raw as f64 * f.scale + f.zero) as f32
                } else {
                    raw as f32 * s + z
                };
                put(o, v, weight, accumulate);
            }
        }};
    }
    match f.bitpix {
        8 => ints!(u8, 1, false),
        16 => ints!(i16, 2, false),
        32 => ints!(i32, 4, true),
        64 => ints!(i64, 8, true),
        -32 => {
            if s == 1.0 && z == 0.0 && weight == 1.0 && !accumulate {
                for (o, c) in out.iter_mut().zip(src.chunks_exact(4)) {
                    *o = f32::from_bits(u32::from_be_bytes([c[0], c[1], c[2], c[3]]));
                }
            } else {
                for (o, c) in out.iter_mut().zip(src.chunks_exact(4)) {
                    let v = f32::from_bits(u32::from_be_bytes([c[0], c[1], c[2], c[3]]));
                    put(o, v * s + z, weight, accumulate);
                }
            }
        }
        -64 => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(8)) {
                let v = (f64::from_be_bytes(c.try_into().unwrap()) * f.scale + f.zero) as f32;
                put(o, v, weight, accumulate);
            }
        }
        _ => unreachable!(),
    }
}

fn load_fits<S: Source + ?Sized>(file: &S) -> Result<(GrayImage, SourceInfo)> {
    let len = file.size()?;
    let mut pos = 0u64;
    let mut primary = None;
    loop {
        let mut header = HashMap::new();
        let mut ended = false;
        let mut blocks = 0;
        while !ended {
            let mut block = [0u8; 2880];
            file.read_at(&mut block, pos)
                .context("Truncated FITS header")?;
            pos += 2880;
            blocks += 1;
            ensure!(blocks < 4096, "FITS header too long");
            for card in block.chunks_exact(80) {
                let key = String::from_utf8_lossy(&card[..8]).trim().to_string();
                if key == "END" {
                    ended = true;
                    break;
                }
                if card[8] == b'=' {
                    let value = String::from_utf8_lossy(&card[10..])
                        .split('/')
                        .next()
                        .unwrap()
                        .trim()
                        .trim_matches('\'')
                        .trim()
                        .to_string();
                    header.insert(key, value);
                }
            }
        }
        let primary = primary.get_or_insert_with(|| header.clone());
        let number = |name: &str, default: i64| -> Result<i64> {
            match header.get(name) {
                Some(v) => Ok(v.parse().with_context(|| format!("Invalid FITS {name}"))?),
                None => Ok(default),
            }
        };
        let float = |name: &str, default: f64| -> Result<f64> {
            match header.get(name) {
                Some(v) => Ok(v
                    .replace('D', "E")
                    .parse()
                    .with_context(|| format!("Invalid FITS {name}"))?),
                None => Ok(default),
            }
        };
        let bitpix = number("BITPIX", 8)?;
        let naxis = number("NAXIS", 0)?;
        ensure!((0..=9).contains(&naxis), "Invalid FITS NAXIS");
        let mut count = if naxis == 0 { 0usize } else { 1usize };
        let mut dims = vec![];
        for i in 1..=naxis {
            let d = number(&format!("NAXIS{i}"), 0)?;
            ensure!(d >= 0, "Negative FITS dimension");
            count = count
                .checked_mul(d as usize)
                .context("FITS dimensions overflow")?;
            dims.push(d as usize);
        }
        ensure!(
            [8, 16, 32, 64, -32, -64].contains(&bitpix),
            "Unsupported FITS BITPIX"
        );
        let bpp = bitpix.unsigned_abs() as usize / 8;
        let pcount = number("PCOUNT", 0)?;
        let gcount = number("GCOUNT", 1)?;
        ensure!(pcount >= 0 && gcount >= 1, "Invalid FITS group size");
        let data_bytes = count
            .checked_add(pcount as usize)
            .and_then(|n| n.checked_mul(gcount as usize))
            .and_then(|n| n.checked_mul(bpp))
            .context("FITS size overflow")?;
        ensure!(
            pos.checked_add(data_bytes as u64)
                .is_some_and(|end| end <= len),
            "Truncated FITS data"
        );
        let image_hdu = header.get("XTENSION").is_none_or(|s| s == "IMAGE");
        if image_hdu && naxis >= 2 {
            ensure!(
                naxis == 2 || (naxis == 3 && dims[2] == 3),
                "FITS cubes require a 2D image or three RGB planes"
            );
            let (width, height) = (dims[0], dims[1]);
            ensure!(
                width > 0
                    && height > 0
                    && width.checked_mul(height).is_some_and(|n| n <= MAX_PIXELS),
                "Invalid image dimensions or image exceeds 100 MP"
            );
            let scale = float("BSCALE", 1.0)?;
            let zero = float("BZERO", 0.0)?;
            ensure!(
                scale.is_finite() && zero.is_finite(),
                "Invalid FITS scaling"
            );
            let blank = header.get("BLANK").and_then(|v| v.parse::<i64>().ok());
            let scaling = FitsScaling {
                bitpix,
                scale,
                zero,
                blank: if bitpix > 0 { blank } else { None },
            };
            let planes = if naxis == 3 { 3 } else { 1 };
            let size = width * height;
            let mut pixels = vec![0f32; size];
            // Row bands are decoded in parallel with positional reads (pread):
            // no shared cursor, no whole-file buffer, page-cache copy + byte swap only.
            let rows = band_rows(width, bpp);
            pixels
                .par_chunks_mut(rows * width)
                .enumerate()
                .try_for_each_init(Vec::<u8>::new, |buf, (band, out)| -> Result<()> {
                    buf.resize(out.len() * bpp, 0);
                    for (channel, &luma) in rgb_weights().iter().enumerate().take(planes) {
                        let offset = pos + ((channel * size + band * rows * width) * bpp) as u64;
                        file.read_at(buf, offset).context("Truncated FITS data")?;
                        let weight = if planes == 1 { 1.0 } else { luma };
                        convert_fits_band(buf, out, scaling, weight, channel > 0);
                    }
                    Ok(())
                })?;
            let mut metadata = primary.clone();
            metadata.extend(header);
            let epoch = metadata
                .get("MJD-OBS")
                .and_then(|v| v.parse::<f64>().ok())
                .map(|mjd| 2000.0 + (mjd - 51544.5) / 365.25)
                .or_else(|| {
                    metadata
                        .get("DATE-OBS")
                        .and_then(|s| s.get(..4))
                        .and_then(|s| s.parse::<f64>().ok())
                        .map(|y| y + 0.5)
                })
                .filter(|t| t.is_finite());
            let lsb = if bitpix > 0 { scale.abs() as f32 } else { 0.0 };
            return Ok((
                GrayImage {
                    width: width as u32,
                    height: height as u32,
                    pixels,
                    epoch,
                },
                SourceInfo {
                    format: SourceFormat::Fits,
                    bitpix: bitpix as i32,
                    lsb,
                    hint: crate::hints::from_fits(&metadata, width as u32, height as u32),
                },
            ));
        }
        let next = pos + (data_bytes as u64).div_ceil(2880) * 2880;
        if next >= len {
            bail!("No supported image HDU in FITS");
        }
        pos = next;
    }
}

#[cfg(test)]
mod preview_tests {
    use super::*;

    /// Linear frame with Gaussian-ish noise and a few faint, small stars: the preview must be
    /// downscaled to the size limit, keep every star visible above the background and put the
    /// background near the target grey.
    #[test]
    fn preview_downscales_by_area_and_stretches_the_background() {
        let (w, h) = (4000usize, 3000usize);
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let mut noise = || {
            let mut acc = 0.0;
            for _ in 0..4 {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                acc += (s >> 40) as f32 / (1u64 << 24) as f32;
            }
            acc - 2.0
        };
        let mut pixels: Vec<f32> = (0..w * h).map(|_| 1000.0 + 4.0 * noise()).collect();
        let stars = [(1000.5, 700.5), (2500.0, 2000.0), (3900.0, 150.0)];
        for &(sx, sy) in &stars {
            for dy in -4i32..=4 {
                for dx in -4i32..=4 {
                    let (x, y) = (sx as i32 + dx, sy as i32 + dy);
                    let r2 = (x as f64 - sx).powi(2) + (y as f64 - sy).powi(2);
                    pixels[y as usize * w + x as usize] += (400.0 * (-r2 / 2.0).exp()) as f32;
                }
            }
        }
        pixels[5] = f32::NAN;
        let img = GrayImage {
            width: w as u32,
            height: h as u32,
            pixels,
            epoch: None,
        };
        assert_eq!(img.preview_size(), (1600, 1200));
        let png = img.preview().unwrap();
        let out = image::load_from_memory(&png).unwrap().to_luma8();
        assert_eq!(out.dimensions(), (1600, 1200));
        let mut bg: Vec<u8> = out.as_raw().iter().step_by(97).copied().collect();
        bg.sort_unstable();
        let median = bg[bg.len() / 2] as f32 / 255.0;
        assert!(
            (median - PREVIEW_BACKGROUND).abs() < 0.05,
            "background {median}"
        );
        for &(sx, sy) in &stars {
            let (px, py) = ((sx / 2.5) as u32, (sy / 2.5) as u32);
            let peak = (py.saturating_sub(1)..=py + 1)
                .flat_map(|y| (px.saturating_sub(1)..=px + 1).map(move |x| (x, y)))
                .map(|(x, y)| out.get_pixel(x.min(1599), y.min(1199))[0])
                .max()
                .unwrap();
            assert!(
                peak as f32 > median * 255.0 + 60.0,
                "star at {sx},{sy}: {peak}"
            );
        }
    }

    #[test]
    fn small_images_are_not_upscaled_and_flat_frames_do_not_fail() {
        let img = GrayImage {
            width: 300,
            height: 200,
            pixels: vec![0.5; 300 * 200],
            epoch: None,
        };
        assert_eq!(img.preview_size(), (300, 200));
        let out = image::load_from_memory(&img.preview().unwrap()).unwrap();
        assert_eq!((out.width(), out.height()), (300, 200));
        let nan = GrayImage {
            width: 10,
            height: 10,
            pixels: vec![f32::NAN; 100],
            epoch: None,
        };
        assert!(nan.preview().is_err());
    }
}

#[cfg(test)]
mod decode_tests {
    use super::*;

    /// A FITS file: primary header with `cards`, then 16-bit samples (big-endian, padded).
    fn fits(
        width: usize,
        height: usize,
        cards: &[&str],
        value: impl Fn(usize, usize) -> i16,
    ) -> Vec<u8> {
        let mut head: Vec<String> = vec![
            "SIMPLE  =                    T".into(),
            "BITPIX  =                   16".into(),
            "NAXIS   =                    2".into(),
            format!("NAXIS1  = {width:>20}"),
            format!("NAXIS2  = {height:>20}"),
            "BZERO   =                32768".into(),
        ];
        head.extend(cards.iter().map(|c| c.to_string()));
        head.push("END".into());
        let mut out = vec![];
        for c in head {
            out.extend(format!("{c:<80}").bytes());
        }
        out.resize(out.len().div_ceil(2880) * 2880, b' ');
        for y in 0..height {
            for x in 0..width {
                out.extend(value(x, y).to_be_bytes());
            }
        }
        out.resize(out.len().div_ceil(2880) * 2880, 0);
        out
    }

    fn temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("skyshot-decode-{}-{name}", std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        p
    }

    fn same(a: &(GrayImage, SourceInfo), b: &(GrayImage, SourceInfo)) {
        assert_eq!(a.1.hint, b.1.hint);
        assert_eq!(
            (a.0.width, a.0.height, a.0.epoch),
            (b.0.width, b.0.height, b.0.epoch)
        );
        assert_eq!(
            (a.1.format, a.1.bitpix, a.1.lsb),
            (b.1.format, b.1.bitpix, b.1.lsb)
        );
        assert!(
            a.0.pixels
                .iter()
                .zip(&b.0.pixels)
                .all(|(p, q)| p.to_bits() == q.to_bits())
        );
    }

    /// The browser detector decodes from memory: the same file must give the same pixels as
    /// the file path (positional reads in bands) does.
    #[test]
    fn bytes_and_files_decode_to_the_same_pixels() {
        let f = fits(
            700,
            500,
            &[
                "DATE-OBS= '2024-03-01T22:10:00'",
                "ROWORDER= 'BOTTOM-UP'",
                "RA      = 83.8",
                "DEC     = -5.4",
                "PIXSCALE= 1.2",
            ],
            |x, y| ((x * 37 + y * 11) % 3000) as i16 - 1000,
        );
        let path = temp("a.fits", &f);
        let from_file = load_with_info(&path).unwrap();
        let from_bytes = decode_bytes(&f).unwrap();
        same(&from_file, &from_bytes);
        assert_eq!(from_bytes.1.format, SourceFormat::Fits);
        assert_eq!(from_bytes.0.epoch, Some(2024.5));
        assert_eq!(from_bytes.1.hint.unwrap().scale_arcsec, Some(1.2));
        assert_eq!(fits_row_order(&f).as_deref(), Some("BOTTOM-UP"));
        assert_eq!(fits_row_order(&fits(16, 16, &[], |_, _| 0)), None);
        std::fs::remove_file(path).unwrap();

        let rgb = image::RgbImage::from_fn(320, 200, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        });
        let mut png = Cursor::new(vec![]);
        rgb.write_to(&mut png, image::ImageFormat::Png).unwrap();
        let png = png.into_inner();
        let path = temp("b.png", &png);
        same(
            &load_with_info(&path).unwrap(),
            &decode_bytes(&png).unwrap(),
        );
        assert_eq!(decode_bytes(&png).unwrap().1.format, SourceFormat::Png);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn invalid_optional_astrometry_never_prevents_fits_decoding() {
        let bytes = fits(
            16,
            16,
            &[
                "CRVAL1  = NaN",
                "CRVAL2  = 95",
                "PIXSCALE= -1",
                "OBJCTRA = '25 70 00'",
                "OBJCTDEC= 'nonsense'",
            ],
            |_, _| 42,
        );
        let (gray, info) = decode_bytes(&bytes).unwrap();
        assert_eq!((gray.width, gray.height), (16, 16));
        assert!(info.hint.is_none());
    }

    #[test]
    fn image_extension_uses_primary_metadata_with_extension_overrides() {
        let mut bytes = vec![];
        for c in [
            "SIMPLE  = T",
            "BITPIX  = 8",
            "NAXIS   = 0",
            "RA      = 83.8",
            "DEC     = -5.4",
            "PIXSCALE= 1",
            "DATE-OBS= '2025-01-01'",
            "END",
        ] {
            bytes.extend(format!("{c:<80}").bytes());
        }
        bytes.resize(2880, b' ');
        let mut ext = fits(16, 16, &["PIXSCALE= 2"], |_, _| 42);
        ext[..80].copy_from_slice(format!("{:<80}", "XTENSION= 'IMAGE'").as_bytes());
        bytes.extend(ext);
        let (gray, info) = decode_bytes(&bytes).unwrap();
        assert_eq!((gray.width, gray.height), (16, 16));
        assert_eq!(gray.epoch, Some(2025.5));
        let h = info.hint.unwrap();
        assert_eq!(h.position, Some([83.8, -5.4]));
        assert_eq!(h.scale_arcsec, Some(2.0));
    }

    #[test]
    fn truncated_and_unknown_bytes_are_errors() {
        let f = fits(64, 64, &[], |_, _| 7);
        assert!(decode_bytes(&f[..f.len() - 4000]).is_err());
        assert!(decode_bytes(b"not an image at all").is_err());
        assert!(decode_bytes(&[]).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn verified_handle_decodes_original_after_path_is_replaced() {
        let original = fits(32, 32, &[], |_, _| 17);
        let path = temp("retained-handle.fits", &original);
        let file = std::fs::File::open(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, fits(32, 32, &[], |_, _| 42)).unwrap();
        same(
            &load_file_with_info(&file).unwrap(),
            &decode_bytes(&original).unwrap(),
        );
        assert_ne!(
            load_with_info(&path).unwrap().0.pixels[0],
            load_file_with_info(&file).unwrap().0.pixels[0]
        );
        std::fs::remove_file(path).unwrap();
    }
}
