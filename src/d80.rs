//! Reader for the documented ASTAP 1476-5 catalog format (Gaia DR3, epoch 2025).
//! IDs are local record addresses, not Gaia source_id values.
use crate::{geometry::unit, types::CatalogStar};
use anyhow::{Context, Result, ensure};
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

const RINGS: [usize; 18] = [
    1, 3, 9, 15, 21, 27, 33, 38, 43, 48, 52, 56, 60, 63, 65, 67, 68, 69,
];
struct Tile {
    path: PathBuf,
    id: u64,
    ra: f64,
    half_ra: f64,
    low: f64,
    high: f64,
}
pub struct D80 {
    tiles: Vec<Tile>,
    pub bytes: u64,
}
/// A catalogue record with the exact integer encoding of the tile: RA = ra * 360 / 16777215,
/// Dec = dec * 90 / 8388607 (degrees), magnitude = mag10 / 10 (BP).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawStar {
    pub ra: u32,
    pub dec: i32,
    pub mag10: i16,
}
impl RawStar {
    #[inline]
    pub fn ra_deg(&self) -> f64 {
        self.ra as f64 * 360.0 / 16777215.0
    }
    #[inline]
    pub fn dec_deg(&self) -> f64 {
        (self.dec as f64 * 90.0 / 8388607.0).clamp(-90.0, 90.0)
    }
    #[inline]
    pub fn mag(&self) -> f32 {
        self.mag10 as f32 / 10.0
    }
}

/// Decode one tile into raw records (no allocation per record).
pub fn decode_raw(data: &[u8], out: &mut Vec<RawStar>) -> Result<()> {
    ensure!(
        data.len() >= 110 && data[109] == 5 && (data.len() - 110).is_multiple_of(5),
        "Invalid 1476-5 data"
    );
    let mut band: Option<(i32, i16)> = None;
    for r in data[110..].chunks_exact(5) {
        if r[..3] == [255; 3] {
            band = Some((r[3] as i32 - 128, r[4] as i16 - 16));
            continue;
        }
        let (dec9, mag10) = band.context("Missing D80 section header")?;
        let ra = u32::from_le_bytes([r[0], r[1], r[2], 0]);
        let dec = r[3] as i32 + r[4] as i32 * 256 + dec9 * 65536;
        out.push(RawStar { ra, dec, mag10 });
    }
    Ok(())
}

impl D80 {
    /// Tile file paths in catalogue order.
    pub fn tile_paths(&self) -> Vec<PathBuf> {
        self.tiles.iter().map(|t| t.path.clone()).collect()
    }
    #[allow(dead_code)]
    pub(crate) fn for_each(&self, mut emit: impl FnMut(CatalogStar)) -> Result<()> {
        for (i, tile) in self.tiles.iter().enumerate() {
            decode_records(&std::fs::read(&tile.path)?, tile.id, &mut emit)?;
            if (i + 1) % 100 == 0 {
                eprintln!("D80 sampled {}/{} tiles", i + 1, self.tiles.len());
            }
        }
        Ok(())
    }
    pub fn open(directory: &Path) -> Result<Self> {
        let mut tiles = vec![];
        let mut bytes = 0;
        for ring in 0..36 {
            let count = RINGS[ring.min(35 - ring)];
            let low = (-90.0 + (ring as f64 - 0.5) * 180.0 / 35.0).max(-90.0);
            let high = (-90.0 + (ring as f64 + 0.5) * 180.0 / 35.0).min(90.0);
            for sector in 0..count {
                let id = ((ring + 1) * 100 + sector + 1) as u64;
                let path = directory.join(format!("d80_{id:04}.1476"));
                let mut file = File::open(&path)
                    .with_context(|| format!("Missing D80 tile {}", path.display()))?;
                let mut header = [0; 110];
                file.read_exact(&mut header)?;
                let len = file.metadata()?.len();
                ensure!(
                    header[109] == 5 && len > 110 && (len - 110).is_multiple_of(5),
                    "Unsupported D80 tile {}",
                    path.display()
                );
                let description = String::from_utf8_lossy(&header[..109]);
                ensure!(
                    description.contains("GAIA DR3") && description.contains("Epoch=2025"),
                    "Expected Gaia DR3 D80 epoch 2025"
                );
                bytes += len;
                tiles.push(Tile {
                    path,
                    id,
                    ra: (sector as f64 + 0.5) * 360.0 / count as f64,
                    half_ra: 180.0 / count as f64,
                    low,
                    high,
                });
            }
        }
        Ok(Self { tiles, bytes })
    }
    pub fn nearby(
        &self,
        ra: f64,
        dec: f64,
        radius: f64,
        cancel: &AtomicBool,
        deadline: Instant,
    ) -> Result<Vec<(u32, CatalogStar)>> {
        ensure!(
            ra.is_finite()
                && dec.is_finite()
                && dec.abs() <= 90.0
                && radius.is_finite()
                && radius > 0.0
                && radius <= 30.0,
            "Invalid catalog cone"
        );
        let ra_radius = if dec.abs() + radius >= 90.0 {
            180.0
        } else {
            (radius.to_radians().sin() / dec.to_radians().cos())
                .clamp(-1.0, 1.0)
                .asin()
                .to_degrees()
        };
        let direction = unit(ra, dec);
        let mut result = vec![];
        for tile in &self.tiles {
            let delta_ra = (ra - tile.ra + 180.0).rem_euclid(360.0) - 180.0;
            if tile.low > dec + radius
                || tile.high < dec - radius
                || delta_ra.abs() > ra_radius + tile.half_ra
            {
                continue;
            }
            ensure!(
                !cancel.load(Ordering::Relaxed) && Instant::now() < deadline,
                "Catalog query interrupted"
            );
            let data = std::fs::read(&tile.path)?;
            decode_records(&data, tile.id, |star| {
                if (star.dec - dec).abs() > radius {
                    return;
                }
                let delta = (star.ra - ra + 180.0).rem_euclid(360.0) - 180.0;
                if delta.abs() > ra_radius {
                    return;
                }
                if direction
                    .iter()
                    .zip(unit(star.ra, star.dec))
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                    >= radius.to_radians().cos()
                {
                    result.push((result.len() as u32, star));
                }
            })?;
        }
        Ok(result)
    }
}
fn decode_records(data: &[u8], tile: u64, mut emit: impl FnMut(CatalogStar)) -> Result<()> {
    ensure!(
        data.len() >= 110 && data[109] == 5 && (data.len() - 110).is_multiple_of(5),
        "Invalid 1476-5 data"
    );
    let mut band = None;
    for (index, r) in data[110..].chunks_exact(5).enumerate() {
        if r[..3] == [255; 3] {
            band = Some((r[3] as i32 - 128, (r[4] as f32 - 16.0) / 10.0));
            continue;
        }
        let (dec9, mag) = band.context("Missing D80 section header")?;
        let ra = u32::from_le_bytes([r[0], r[1], r[2], 0]) as f64 * 360.0 / 16777215.0;
        let dec = (r[3] as i32 + r[4] as i32 * 256 + dec9 * 65536) as f64 * 90.0 / 8388607.0;
        ensure!(dec.abs() <= 90.00002, "Invalid D80 declination");
        emit(CatalogStar {
            source_id: (tile << 32) | index as u64,
            ra,
            dec: dec.clamp(-90.0, 90.0),
            mag,
            pmra: 0.0,
            pmdec: 0.0,
            epoch: 2025.0,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn documented_sirius_record_and_negative_declination() {
        let mut bytes = vec![b' '; 110];
        bytes[109] = 5;
        bytes.extend([255, 255, 255, 104, 1, 195, 6, 72, 215, 57]);
        let mut stars = vec![];
        decode_records(&bytes, 501, |s| stars.push(s)).unwrap();
        assert_eq!(stars.len(), 1);
        assert!((stars[0].ra - 101.287149).abs() < 0.00002);
        assert!((stars[0].dec + 16.7161401).abs() < 0.00002);
        assert_eq!(stars[0].mag, -1.5);
        assert_eq!(stars[0].source_id, (501u64 << 32) | 1);
        assert_eq!(RINGS.iter().sum::<usize>() * 2, 1476);
        bytes.pop();
        assert!(decode_records(&bytes, 501, |_| {}).is_err());
    }
}
