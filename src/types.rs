use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Star {
    pub x: f64,
    pub y: f64,
    pub flux: f64,
    #[serde(default = "default_sigma")]
    pub sigma: f64,
    #[serde(default = "default_snr")]
    pub snr: f64,
    #[serde(default = "default_fwhm")]
    pub fwhm: f64,
    #[serde(default = "default_one")]
    pub elongation: f64,
    #[serde(default)]
    pub saturated: bool,
    /// Background-subtracted peak pixel value (native resolution).
    #[serde(default)]
    pub peak: f64,
    /// Position angle of the elongation axis in degrees, `[0, 180)`, measured from
    /// the +x axis towards +y (image coordinates).
    #[serde(default)]
    pub pa: f64,
    /// Quality flags, see [`star_flags`].
    #[serde(default)]
    pub flags: u32,
}

/// Bit masks for [`Star::flags`].
pub mod star_flags {
    /// Flat-topped (clipped) core; centroid from the plateau.
    pub const SATURATED: u32 = 1 << 0;
    /// Too sharp for the frame PSF: hot pixel or cosmic ray.
    pub const HOT_PIXEL: u32 = 1 << 1;
    /// A separate source lies within ~2 FWHM; compact window used.
    pub const BLENDED: u32 = 1 << 2;
    /// Measurement window touches the frame edge or masked pixels.
    pub const NEAR_EDGE: u32 = 1 << 3;
    /// Frame is trailed and this source follows the common trail direction.
    pub const TRAILED: u32 = 1 << 4;
    /// Member of a collinear chain (satellite/airplane streak, column defect).
    pub const ON_LINE: u32 = 1 << 5;
    /// Not locally significant above a ridge or structured background.
    pub const RIDGE: u32 = 1 << 6;
    /// Elongated far beyond the frame median (not trailed).
    pub const ELONGATED: u32 = 1 << 7;
    /// Much broader than the frame PSF (nebular knot, galaxy core).
    pub const EXTENDED: u32 = 1 << 8;
    /// Flags that remove a detection from the default output.
    pub const REJECT: u32 = HOT_PIXEL | ON_LINE | RIDGE | ELONGATED | EXTENDED;
}

impl Default for Star {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            flux: 0.0,
            sigma: default_sigma(),
            snr: default_snr(),
            fwhm: default_fwhm(),
            elongation: 1.0,
            saturated: false,
            peak: 0.0,
            pa: 0.0,
            flags: 0,
        }
    }
}
fn default_sigma() -> f64 {
    0.25
}
fn default_snr() -> f64 {
    20.0
}
fn default_fwhm() -> f64 {
    3.0
}
fn default_one() -> f64 {
    1.0
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CatalogStar {
    pub source_id: u64,
    pub ra: f64,
    pub dec: f64,
    pub mag: f32,
    #[serde(default)]
    pub pmra: f32,
    #[serde(default)]
    pub pmdec: f32,
    #[serde(default = "default_epoch")]
    pub epoch: f32,
}
fn default_epoch() -> f32 {
    2016.0
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StarField {
    pub width: u32,
    pub height: u32,
    pub stars: Vec<Star>,
    #[serde(default)]
    pub epoch: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SolveOptions {
    pub timeout_seconds: f64,
    pub min_fov: f64,
    pub max_fov: f64,
    pub max_patterns: usize,
    pub tolerance: f64,
    pub min_matches: usize,
    pub sip: bool,
    pub active: bool,
    pub ranker: Option<String>,
    #[serde(default)]
    pub all_sky: Option<AllSkyOptions>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AllSkyOptions {
    pub executable: std::path::PathBuf,
    pub database: std::path::PathBuf,
}
impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            timeout_seconds: 30.0,
            min_fov: 0.2,
            max_fov: 20.0,
            max_patterns: 10080,
            tolerance: 0.0025,
            min_matches: 10,
            sip: true,
            active: false,
            ranker: None,
            all_sky: None,
        }
    }
}

/// SIP distortion of arbitrary order. Coefficients follow `Sip::terms(order)`, i.e. for
/// n = 2..=order, p = n..=0: (p, n - p). Pixel offsets u, v are relative to CRPIX (0-based):
/// x' = u + sum A_pq u^p v^q, y' = v + sum B_pq u^p v^q, then CD maps (x', y') to the plane.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sip {
    pub order: u32,
    pub a: Vec<f64>,
    pub b: Vec<f64>,
}
impl Sip {
    pub fn terms(order: u32) -> Vec<(u32, u32)> {
        let mut t = vec![];
        for n in 2..=order {
            for p in (0..=n).rev() {
                t.push((p, n - p));
            }
        }
        t
    }
    /// Powers 1, x, x^2, ..., x^n (n <= 15).
    #[inline]
    fn powers(x: f64, n: usize) -> [f64; 16] {
        let mut p = [0.0; 16];
        p[0] = 1.0;
        for i in 1..=n.min(15) {
            p[i] = p[i - 1] * x;
        }
        p
    }
    #[inline]
    pub fn eval(&self, u: f64, v: f64) -> (f64, f64) {
        let n = self.order as usize;
        let (pu, pv) = (Self::powers(u, n), Self::powers(v, n));
        let mut fa = 0.0;
        let mut fb = 0.0;
        let mut k = 0;
        for n in 2..=n {
            for p in (0..=n).rev() {
                let m = pu[p] * pv[n - p];
                fa += self.a[k] * m;
                fb += self.b[k] * m;
                k += 1;
            }
        }
        (fa, fb)
    }
    /// Distortion and Jacobian together, sharing the powers and coefficient traversal.
    /// Accumulation order matches `eval` and `jacobian` exactly.
    #[inline]
    pub fn eval_jacobian(&self, u: f64, v: f64) -> ((f64, f64), [[f64; 2]; 2]) {
        let n = self.order as usize;
        let (pu, pv) = (Self::powers(u, n), Self::powers(v, n));
        let (mut fa, mut fb) = (0.0, 0.0);
        let mut j = [[1.0, 0.0], [0.0, 1.0]];
        let mut k = 0;
        for n in 2..=n {
            for p in (0..=n).rev() {
                let q = n - p;
                let m = pu[p] * pv[q];
                fa += self.a[k] * m;
                fb += self.b[k] * m;
                let du = if p > 0 {
                    p as f64 * pu[p - 1] * pv[q]
                } else {
                    0.0
                };
                let dv = if q > 0 {
                    q as f64 * pu[p] * pv[q - 1]
                } else {
                    0.0
                };
                j[0][0] += self.a[k] * du;
                j[0][1] += self.a[k] * dv;
                j[1][0] += self.b[k] * du;
                j[1][1] += self.b[k] * dv;
                k += 1;
            }
        }
        ((fa, fb), j)
    }

    /// Jacobian of (u + f, v + g) with respect to (u, v).
    #[inline]
    pub fn jacobian(&self, u: f64, v: f64) -> [[f64; 2]; 2] {
        let n = self.order as usize;
        let (pu, pv) = (Self::powers(u, n), Self::powers(v, n));
        let mut j = [[1.0, 0.0], [0.0, 1.0]];
        let mut k = 0;
        for n in 2..=n {
            for p in (0..=n).rev() {
                let q = n - p;
                let du = if p > 0 {
                    p as f64 * pu[p - 1] * pv[q]
                } else {
                    0.0
                };
                let dv = if q > 0 {
                    q as f64 * pu[p] * pv[q - 1]
                } else {
                    0.0
                };
                j[0][0] += self.a[k] * du;
                j[0][1] += self.a[k] * dv;
                j[1][0] += self.b[k] * du;
                j[1][1] += self.b[k] * dv;
                k += 1;
            }
        }
        j
    }
}

/// FITS celestial projection. Cylindrical models use `Wcs::native_pole` to describe
/// the panorama's axis independently of celestial north (including transverse panoramas).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Proj {
    #[default]
    #[serde(rename = "TAN")]
    Tan,
    #[serde(rename = "ARC")]
    Arc,
    #[serde(rename = "STG")]
    Stg,
    #[serde(rename = "CAR")]
    Car,
    #[serde(rename = "MER")]
    Mer,
    /// Perspective cylindrical, FITS CYP with mu=0, lambda=1: y = tan(latitude).
    #[serde(rename = "CYP")]
    Cyp,
}

impl Proj {
    pub fn is_tan(&self) -> bool {
        *self == Proj::Tan
    }
    pub fn is_cylindrical(&self) -> bool {
        matches!(self, Self::Car | Self::Mer | Self::Cyp)
    }
    /// The FITS projection code (`CTYPEi` = `RA---{code}`).
    pub fn code(&self) -> &'static str {
        match self {
            Self::Tan => "TAN",
            Self::Arc => "ARC",
            Self::Stg => "STG",
            Self::Car => "CAR",
            Self::Mer => "MER",
            Self::Cyp => "CYP",
        }
    }
}

/// Atmospheric refraction measured from the residuals of a solved wide landscape (see
/// `solve::refract`): the zenith and the strength of the refraction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Refraction {
    /// Zenith (ICRS right ascension, declination; deg): the observer's local sidereal time and
    /// latitude, up to precession and nutation of the equinox of date.
    pub zenith: [f64; 2],
    /// Refraction relative to the standard atmosphere of Saemundsson's formula (1010 hPa, 10 C).
    pub scale: f64,
    /// Rms (px) of the lens model without refraction and with the measured one.
    pub rms_px: f64,
    pub rms_refracted_px: f64,
    /// Lowest altitude (deg) of a matched star under the measured zenith.
    pub min_alt_deg: f64,
    pub n_pairs: usize,
}

/// CRPIX uses zero-based pixels internally; FITS export adds one.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Wcs {
    pub ra: f64,
    pub dec: f64,
    pub crpix: [f64; 2],
    pub cd: [[f64; 2]; 2],
    pub sip: Option<Sip>,
    /// Projection (absent in JSON = TAN, so TAN solutions serialize as before).
    #[serde(default, skip_serializing_if = "Proj::is_tan")]
    pub proj: Proj,
    /// ICRS RA/Dec of the cylindrical panorama's north pole. Absent: local north
    /// at CRVAL, so the reference point lies on the native equator. Ignored by zenithal models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_pole: Option<[f64; 2]>,
    /// Independently validated local registration after projection and SIP. JSON carries
    /// the exact map; a standalone FITS header can describe only the base projection/SIP.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_grid: Option<crate::pixelgrid::PixelGrid>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StarMatch {
    pub image_index: usize,
    pub source_id: String,
    pub x: f64,
    pub y: f64,
    pub predicted_x: f64,
    pub predicted_y: f64,
    pub residual_px: f64,
    pub ra: f64,
    pub dec: f64,
    pub seed: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Timings {
    pub decode_ms: f64,
    pub detect_ms: f64,
    pub patterns_ms: f64,
    pub search_ms: f64,
    pub verify_ms: f64,
    pub total_ms: f64,
    #[serde(default)]
    pub fallback_ms: f64,
}

/// Outcome of a solve. Serialised in snake_case (`"solved"`, `"ambiguous"`, ...), the same
/// strings as `solve::Solution::status`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SolveStatus {
    Solved,
    Unsolved,
    Timeout,
    Cancelled,
    InvalidInput,
    /// Two geometrically inconsistent hypotheses both passed verification and the better one
    /// does not lead by `SolveConfig::min_margin_nats`; no WCS is reported (a wrong WCS is
    /// worse than none). `Solution::margin_nats` carries the measured gap.
    Ambiguous,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SolveResult {
    pub status: SolveStatus,
    pub reason: String,
    pub width: u32,
    pub height: u32,
    pub wcs: Option<Wcs>,
    pub center: Option<[f64; 2]>,
    pub scale_arcsec: Option<f64>,
    pub rotation_deg: Option<f64>,
    pub mirrored: Option<bool>,
    pub rms_px: Option<f64>,
    /// Conservative Poisson tail score for unseeded coincidences; not a calibrated posterior.
    pub log_likelihood_ratio: Option<f64>,
    pub matches: Vec<StarMatch>,
    pub detections: Vec<Star>,
    pub patterns_tried: usize,
    pub candidates_verified: usize,
    pub timings: Timings,
    pub atlas: String,
    pub warnings: Vec<String>,
    #[serde(default = "default_backend")]
    pub backend: String,
    #[serde(default)]
    pub held_out_matches: usize,
    #[serde(default)]
    pub backend_detail: Option<String>,
}
fn default_backend() -> String {
    "skyanchor".into()
}
impl SolveResult {
    pub fn empty(field: &StarField, atlas: String) -> Self {
        Self {
            status: SolveStatus::Unsolved,
            reason: "No verified hypothesis".into(),
            width: field.width,
            height: field.height,
            wcs: None,
            center: None,
            scale_arcsec: None,
            rotation_deg: None,
            mirrored: None,
            rms_px: None,
            log_likelihood_ratio: None,
            matches: vec![],
            detections: field.stars.clone(),
            patterns_tried: 0,
            candidates_verified: 0,
            timings: Timings::default(),
            atlas,
            warnings: vec![],
            backend: default_backend(),
            held_out_matches: 0,
            backend_detail: None,
        }
    }
}
