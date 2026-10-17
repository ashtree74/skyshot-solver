//! Optional, untrusted astrometric hints. Shared by native decoders, WASM and the API.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Hints only narrow a preliminary search. They never replace catalogue verification;
/// a failed preliminary search is followed by the original blind search.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolveHint {
    /// Approximate equatorial position [RA, Dec], degrees.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<[f64; 2]>,
    /// Approximate sampling, arcseconds per pixel (the larger axis for non-square pixels).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale_arcsec: Option<f64>,
}

/// Below this longest-side field size AUTO tries the DEEP companion first.
pub const DEEP_FOV_DEG: f64 = 0.3;

impl SolveHint {
    /// Explicit components take precedence; fill missing ones from the file metadata.
    pub fn merge(primary: Option<Self>, fallback: Option<Self>) -> Option<Self> {
        let p = primary.and_then(Self::sanitized).unwrap_or_default();
        let f = fallback.and_then(Self::sanitized).unwrap_or_default();
        Self {
            position: p.position.or(f.position),
            scale_arcsec: p.scale_arcsec.or(f.scale_arcsec),
        }
        .sanitized()
    }

    /// Discard invalid components independently: a bad scale must not lose a good position.
    pub fn sanitized(self) -> Option<Self> {
        let h = Self {
            position: self
                .position
                .filter(|[ra, dec]| {
                    ra.is_finite()
                        && (0.0..=360.0).contains(ra)
                        && dec.is_finite()
                        && (-90.0..=90.0).contains(dec)
                })
                .map(|[ra, dec]| [ra.rem_euclid(360.0), dec]),
            scale_arcsec: self
                .scale_arcsec
                .filter(|s| s.is_finite() && *s > 0.0 && *s <= 3600.0),
        };
        (h.position.is_some() || h.scale_arcsec.is_some()).then_some(h)
    }

    pub fn fov_deg(self, width: u32, height: u32) -> Option<f64> {
        self.sanitized()?
            .scale_arcsec
            .map(|s| s * width.max(height) as f64 / 3600.0)
            .filter(|f| *f >= 0.001 && *f < 180.0)
    }

    /// Also reject sampling that cannot describe a plausible field at this image size.
    pub fn for_frame(self, width: u32, height: u32) -> Option<Self> {
        let mut h = self.sanitized()?;
        if h.fov_deg(width, height).is_none() {
            h.scale_arcsec = None;
        }
        h.sanitized()
    }
}

fn number(h: &HashMap<String, String>, key: &str) -> Option<f64> {
    h.get(key)?
        .trim()
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
}

/// Decimal RA is degrees for RA/DEC, hours for OBJCTRA/OBJCTDEC and TELRA/TELDEC.
/// Sexagesimal RA is always hours. Declinations preserve a negative zero sign.
fn coordinate(s: &str, ra: bool, hours: bool) -> Option<f64> {
    let parts: Vec<&str> = s
        .trim()
        .split(|c: char| c == ':' || c.is_ascii_whitespace())
        .filter(|s| !s.is_empty())
        .collect();
    let first = parts
        .first()?
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .ok()?;
    if !first.is_finite() {
        return None;
    }
    let value = if parts.len() == 1 {
        first * if ra && hours { 15.0 } else { 1.0 }
    } else {
        if !(2..=3).contains(&parts.len()) {
            return None;
        }
        let m = parts[1].parse::<f64>().ok()?;
        let sec = parts.get(2).map_or(Some(0.0), |s| s.parse::<f64>().ok())?;
        if !(0.0..60.0).contains(&m) || !(0.0..60.0).contains(&sec) {
            return None;
        }
        first.signum() * (first.abs() + m / 60.0 + sec / 3600.0) * if ra { 15.0 } else { 1.0 }
    };
    let limit = if ra { 360.0 } else { 90.0 };
    (value.is_finite() && value.abs() <= limit && (!ra || value >= 0.0)).then_some(value)
}

fn unit(h: &HashMap<String, String>, axis: usize) -> Option<f64> {
    match h
        .get(&format!("CUNIT{axis}"))
        .map(|s| s.trim().to_ascii_lowercase())
        .as_deref()
        .unwrap_or("deg")
    {
        "deg" | "degree" | "degrees" => Some(1.0),
        "rad" => Some(180.0 / std::f64::consts::PI),
        "arcmin" => Some(1.0 / 60.0),
        "arcsec" => Some(1.0 / 3600.0),
        _ => None,
    }
}

/// Only equatorial celestial WCS can supply a pointing/scale. Bare CRVAL/CDELT cards are
/// also common in capture files; explicitly non-celestial axes must never become sky hints.
fn wcs_axes(h: &HashMap<String, String>) -> bool {
    match (h.get("CTYPE1"), h.get("CTYPE2")) {
        (None, None) => true,
        (Some(a), Some(b)) => a.starts_with("RA---") && b.starts_with("DEC--"),
        _ => false,
    }
}

fn matrix(h: &HashMap<String, String>) -> Option<[[f64; 2]; 2]> {
    if !wcs_axes(h) {
        return None;
    }
    let units = [unit(h, 1)?, unit(h, 2)?];
    let mut cd = [[0.0; 2]; 2];
    let has_cd = ["CD1_1", "CD1_2", "CD2_1", "CD2_2"]
        .iter()
        .any(|k| h.contains_key(*k));
    for (i, row) in cd.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = if has_cd {
                let k = format!("CD{}_{}", i + 1, j + 1);
                if h.contains_key(&k) {
                    number(h, &k)?
                } else {
                    0.0
                }
            } else {
                let k = format!("PC{}_{}", i + 1, j + 1);
                let pc = if h.contains_key(&k) {
                    number(h, &k)?
                } else {
                    f64::from(i == j)
                };
                number(h, &format!("CDELT{}", i + 1))? * pc
            } * units[i];
        }
    }
    let det = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
    (det.is_finite() && det != 0.0).then_some(cd)
}

pub(crate) fn from_fits(h: &HashMap<String, String>, width: u32, height: u32) -> Option<SolveHint> {
    let cd = matrix(h);
    let wcs_position = (|| {
        if !wcs_axes(h) {
            return None;
        }
        let ra = number(h, "CRVAL1")? * unit(h, 1)?;
        let dec = number(h, "CRVAL2")? * unit(h, 2)?;
        let position = SolveHint {
            position: Some([ra, dec]),
            scale_arcsec: None,
        }
        .sanitized()?
        .position?;
        // CRVAL is the reference pixel, not necessarily the frame centre. Correct it when
        // the TAN geometry is available; otherwise the loose position cone is intentional.
        if h.get("CTYPE1")
            .is_some_and(|s| s == "RA---TAN" || s == "RA---TAN-SIP")
            && let (Some(cd), Some(x), Some(y)) = (cd, number(h, "CRPIX1"), number(h, "CRPIX2"))
        {
            let w = crate::types::Wcs {
                ra,
                dec,
                crpix: [x - 1.0, y - 1.0],
                cd,
                sip: None,
                proj: crate::types::Proj::Tan,
                native_pole: None,
                local_grid: None,
            };
            let [ra, dec] = w.pixel_to_sky((width as f64 - 1.0) / 2.0, (height as f64 - 1.0) / 2.0);
            return SolveHint {
                position: Some([ra, dec]),
                scale_arcsec: None,
            }
            .sanitized()?
            .position;
        }
        Some(position)
    })();
    let position = wcs_position.or_else(|| {
        [
            ("OBJCTRA", "OBJCTDEC", true),
            ("RA", "DEC", false),
            ("TELRA", "TELDEC", true),
        ]
        .into_iter()
        .find_map(|(r, d, hours)| {
            Some([
                coordinate(h.get(r)?, true, hours)?,
                coordinate(h.get(d)?, false, false)?,
            ])
        })
    });
    let positive = |k| number(h, k).filter(|v| *v > 0.0);
    let sampling = |k| positive(k).filter(|v| *v <= 3600.0);
    let scale_arcsec = cd
        .map(|m| m[0][0].hypot(m[1][0]).max(m[0][1].hypot(m[1][1])) * 3600.0)
        .filter(|s| s.is_finite() && *s > 0.0 && *s <= 3600.0)
        .or_else(|| sampling("PIXSCALE"))
        .or_else(|| sampling("SECPIX"))
        .or_else(|| sampling("SCALE"))
        .or_else(|| {
            // Capture programs disagree on whether XPIXSZ already includes binning. Treat
            // the recorded pitch as effective; the 2x search tolerance and blind fallback
            // also cover the alternative convention without double-counting binning.
            let pixel = positive("XPIXSZ").or_else(|| positive("YPIXSZ"))?;
            Some(206.26480624709636 * pixel / positive("FOCALLEN")?)
        });
    SolveHint {
        position,
        scale_arcsec,
    }
    .for_frame(width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(cards: &[(&str, &str)]) -> HashMap<String, String> {
        cards
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn capture_coordinates_and_sampling() {
        for ra in ["05 35 12", "05:35:12", "5.586666666666667"] {
            let h = from_fits(
                &header(&[
                    ("OBJCTRA", ra),
                    ("OBJCTDEC", "-00 30 00"),
                    ("PIXSCALE", "1.2D+00"),
                ]),
                1000,
                750,
            )
            .unwrap();
            let p = h.position.unwrap();
            assert!((p[0] - 83.8).abs() < 1e-10);
            assert_eq!(p[1], -0.5);
            assert_eq!(h.scale_arcsec, Some(1.2));
        }
        let h = from_fits(
            &header(&[
                ("RA", "83.8"),
                ("DEC", "-5.4"),
                ("FOCALLEN", "800"),
                ("XPIXSZ", "3.76"),
                ("XBINNING", "2"),
            ]),
            2000,
            1500,
        )
        .unwrap();
        assert_eq!(h.position, Some([83.8, -5.4]));
        assert!((h.scale_arcsec.unwrap() - 0.96944458936).abs() < 1e-8);
    }

    #[test]
    fn celestial_cd_and_pc_units_and_reference_pixel() {
        let mut h = header(&[
            ("CTYPE1", "RA---TAN"),
            ("CTYPE2", "DEC--TAN"),
            ("CRVAL1", "83.8"),
            ("CRVAL2", "-5.4"),
            ("CRPIX1", "500.5"),
            ("CRPIX2", "375.5"),
            ("CD1_1", "0"),
            ("CD1_2", "-0.0005"),
            ("CD2_1", "0.0005"),
            ("CD2_2", "0"),
            ("PIXSCALE", "999"),
        ]);
        let hint = from_fits(&h, 1000, 750).unwrap();
        assert_eq!(hint.position, Some([83.8, -5.4]));
        assert_eq!(hint.scale_arcsec, Some(1.8));
        h.insert("CRPIX1".into(), "1".into());
        let off = from_fits(&h, 1000, 750).unwrap().position.unwrap();
        assert!((off[1] + 5.4).abs() > 0.2);
        for k in ["CD1_1", "CD1_2", "CD2_1", "CD2_2"] {
            h.remove(k);
        }
        h.extend(header(&[
            ("CUNIT1", "arcsec"),
            ("CUNIT2", "arcsec"),
            ("CRVAL1", "301680"),
            ("CRVAL2", "-19440"),
            ("CRPIX1", "500.5"),
            ("CDELT1", "-1.8"),
            ("CDELT2", "1.8"),
            ("PC1_1", "0"),
            ("PC1_2", "-1"),
            ("PC2_1", "1"),
            ("PC2_2", "0"),
        ]));
        let pc = from_fits(&h, 1000, 750).unwrap();
        assert!((pc.position.unwrap()[0] - 83.8).abs() < 1e-10);
        assert_eq!(pc.scale_arcsec, Some(1.8));
    }

    #[test]
    fn missing_malformed_and_noncelestial_metadata_are_ignored() {
        assert_eq!(from_fits(&header(&[]), 1000, 750), None);
        for value in ["NaN", "inf", "-1", "0", "oops", "1e100"] {
            let h = from_fits(
                &header(&[
                    ("PIXSCALE", value),
                    ("OBJCTRA", "garbage"),
                    ("OBJCTDEC", "-91"),
                ]),
                1000,
                750,
            );
            assert_eq!(h, None, "{value}");
        }
        assert_eq!(
            from_fits(
                &header(&[
                    ("OBJCTRA", "25:00:00"),
                    ("OBJCTDEC", "5"),
                    ("PIXSCALE", "2")
                ]),
                1000,
                750
            )
            .unwrap()
            .position,
            None
        );
        assert_eq!(
            from_fits(
                &header(&[
                    ("CTYPE1", "GLON-TAN"),
                    ("CTYPE2", "GLAT-TAN"),
                    ("CRVAL1", "83.8"),
                    ("CRVAL2", "-5.4"),
                    ("CDELT1", "-0.001"),
                    ("CDELT2", "0.001")
                ]),
                1000,
                750
            ),
            None
        );
        let partial = from_fits(
            &header(&[("RA", "83.8"), ("DEC", "-5.4"), ("PIXSCALE", "NaN")]),
            1000,
            750,
        )
        .unwrap();
        assert_eq!(partial.position, Some([83.8, -5.4]));
        assert_eq!(partial.scale_arcsec, None);
        let fallback = from_fits(
            &header(&[("PIXSCALE", "1e100"), ("SECPIX", "2")]),
            1000,
            750,
        )
        .unwrap();
        assert_eq!(fallback.scale_arcsec, Some(2.0));
    }

    #[test]
    fn deep_uses_the_longest_side_and_finite_positive_scale() {
        let h = SolveHint {
            position: None,
            scale_arcsec: Some(0.5),
        };
        assert!(h.fov_deg(1500, 1000).unwrap() < DEEP_FOV_DEG);
        assert!(h.fov_deg(1000, 3000).unwrap() > DEEP_FOV_DEG);
        for s in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
            assert_eq!(
                SolveHint {
                    scale_arcsec: Some(s),
                    ..h
                }
                .sanitized(),
                None
            );
        }
        let merged = SolveHint::merge(
            Some(SolveHint {
                position: Some([0.0, 0.0]),
                scale_arcsec: None,
            }),
            Some(h),
        )
        .unwrap();
        assert_eq!(merged.position, Some([0.0, 0.0]));
        assert_eq!(merged.scale_arcsec, Some(0.5));
        assert!(
            SolveHint {
                position: None,
                scale_arcsec: Some(1e-100)
            }
            .for_frame(1000, 750)
            .is_none()
        );
    }
}
