//! Serializable solver types, star rows, attempt ordering and WCS output.
//! This package has no HTTP transport or hosted service.

pub mod field;

pub use crate::hints::SolveHint;
pub use crate::skymask::SkyMask;
use crate::types::{Star, StarField, Wcs};
use crate::{compat, solve::Solution};
pub use field::FieldSummary;
use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, SeqAccess, Visitor},
    ser::SerializeSeq,
};
use std::fmt;

pub const API_VERSION: &str = "v1";
/// Stars per attempt (the solver uses the brightest `SolveConfig::max_stars` = 6000).
pub const MAX_STARS: usize = 6000;
pub const MAX_ATTEMPTS: usize = 3;
/// Largest request body (uncompressed; compressed bodies are refused).
pub const MAX_BODY_BYTES: usize = 512 * 1024;
pub const MIN_SIDE: u32 = 16;
pub const MAX_SIDE: u32 = 50_000;
/// Centroid sigma (px) assumed when a row has none.
pub const DEFAULT_SIGMA: f64 = 0.25;
/// Longest client-supplied string kept (names, versions), in characters.
pub const MAX_CLIENT_STR: usize = 32;

/// The detection attempts in order, with the share of the deadline each gets when another one
/// follows; the last attempt gets what is left. Clients extract the next attempt only when the
/// previous one did not solve; the server applies the same split to the attempts of a request
/// (`pipeline::solve_attempts`), and the local pipeline uses the same table.
pub const ATTEMPTS: [(&str, f64); 3] = [
    ("default", 0.25),
    ("local_background_compact", 0.25),
    ("highpass_dog", 1.0),
];

/// The attempt after `name`, if any.
pub fn next_attempt(name: &str) -> Option<&'static str> {
    let i = ATTEMPTS.iter().position(|a| a.0 == name)?;
    ATTEMPTS.get(i + 1).map(|a| a.0)
}

/// Deadline share of `name` when attempts follow it.
pub fn attempt_share(name: &str) -> Option<f64> {
    ATTEMPTS.iter().find(|a| a.0 == name).map(|a| a.1)
}

/// One detected star: position (zero-based native pixels), brightness and centroid sigma.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Row {
    pub x: f64,
    pub y: f64,
    pub flux: Option<f64>,
    pub sigma: Option<f64>,
}

impl Serialize for Row {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let n = match (self.flux, self.sigma) {
            (_, Some(_)) => 4,
            (Some(_), None) => 3,
            (None, None) => 2,
        };
        let mut seq = s.serialize_seq(Some(n))?;
        seq.serialize_element(&self.x)?;
        seq.serialize_element(&self.y)?;
        if self.flux.is_some() || self.sigma.is_some() {
            // a sigma needs the flux slot before it
            seq.serialize_element(&self.flux.unwrap_or(0.0))?;
        }
        if let Some(sg) = self.sigma {
            seq.serialize_element(&sg)?;
        }
        seq.end()
    }
}

impl<'de> Deserialize<'de> for Row {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Row;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a star row [x, y, flux?, sigma?]")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Row, A::Error> {
                let x: f64 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let y: f64 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                let flux: Option<f64> = seq.next_element()?;
                let sigma: Option<f64> = seq.next_element()?;
                if seq.next_element::<de::IgnoredAny>()?.is_some() {
                    return Err(de::Error::invalid_length(5, &self));
                }
                Ok(Row { x, y, flux, sigma })
            }
        }
        d.deserialize_seq(V)
    }
}

/// Rows for the solver: finite positions, brightest first (stable, as `solve::context` sorts),
/// at most [`MAX_STARS`], with the rank as flux (`n - i`; only the order matters) and the
/// centroid sigma when it is finite.
pub fn rows_from_field(field: &StarField) -> Vec<Row> {
    let mut stars: Vec<&Star> = field
        .stars
        .iter()
        .filter(|s| s.x.is_finite() && s.y.is_finite())
        .collect();
    stars.sort_by(|a, b| {
        b.flux
            .partial_cmp(&a.flux)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    stars.truncate(MAX_STARS);
    let n = stars.len();
    stars
        .iter()
        .enumerate()
        .map(|(i, s)| Row {
            x: s.x,
            y: s.y,
            flux: Some((n - i) as f64),
            sigma: s.sigma.is_finite().then_some(s.sigma),
        })
        .collect()
}

/// A star list for the solver from validated rows.
pub fn field_from_rows(width: u32, height: u32, rows: &[Row], epoch: Option<f64>) -> StarField {
    let n = rows.len();
    StarField {
        width,
        height,
        stars: rows
            .iter()
            .enumerate()
            .map(|(i, r)| Star {
                x: r.x,
                y: r.y,
                flux: r.flux.unwrap_or((n - i) as f64),
                sigma: r.sigma.unwrap_or(DEFAULT_SIGMA),
                snr: 0.0,
                fwhm: 0.0,
                elongation: 1.0,
                saturated: false,
                peak: 0.0,
                pa: 0.0,
                flags: 0,
            })
            .collect(),
        epoch,
    }
}

/// `POST /v1/solve`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolveRequest {
    pub width: u32,
    pub height: u32,
    /// One star list (attempt `default`), or use `attempts`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stars: Option<Vec<Row>>,
    /// Several detections of the same frame, solved in order with the [`ATTEMPTS`] split of
    /// the deadline (a client that cannot wait for a round trip sends them all at once).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attempts: Vec<AttemptIn>,
    #[serde(default)]
    pub options: SolveOptions,
    /// Accepted and ignored: the catalogue has no proper motions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epoch: Option<f64>,
    /// Untrusted client telemetry, logged (strings cut to [`MAX_CLIENT_STR`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client: Option<ClientInfo>,
    /// Where stars are hidden (ground, buildings, outside a fisheye's image circle), as
    /// estimated by the client from the pixels ([`crate::skymask::estimate`]); detections and
    /// catalogue stars there are ignored. Servers before 2026-10-11 refuse the field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask: Option<SkyMask>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttemptIn {
    pub name: String,
    pub stars: Vec<Row>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolveOptions {
    /// Untrusted position/scale, e.g. from FITS. Invalid components are ignored; a failed
    /// hinted attempt falls back to the original blind mode. Capability: `hints`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hint: Option<SolveHint>,
    /// AUTO tries the normal index before the optional narrow-field companion, unless a
    /// narrow-field hint selects the companion first.
    #[serde(default, skip_serializing_if = "SolveMode::is_auto")]
    pub mode: SolveMode,
    /// Solve budget; capped by the key's tier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deadline_ms: Option<f64>,
    /// Return the verified star pairs.
    #[serde(default)]
    pub matches: bool,
    /// Return the FITS WCS header text.
    #[serde(default)]
    pub fits: bool,
    /// The client reads `wcs.proj`: fields too wide for TAN may be answered with a zenithal
    /// equidistant (ARC) projection. Without either projection capability the WCS is always TAN (+SIP).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub arc: bool,
    /// Client supports STG, CAR, MER, CYP and `wcs.native_pole`. Independent of `arc`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub panorama: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum SolveMode {
    #[default]
    Auto,
    Standard,
    Deep,
}

impl SolveMode {
    pub fn is_auto(&self) -> bool {
        *self == Self::Auto
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ClientInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Build hash of the detector (for comparing detector builds by solve rate).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detect_ms: Option<f64>,
}

impl ClientInfo {
    /// The same with strings cut to [`MAX_CLIENT_STR`] characters and non-finite times dropped.
    pub fn sanitized(&self) -> Self {
        let cut = |s: &Option<String>| {
            s.as_ref()
                .map(|v| v.chars().take(MAX_CLIENT_STR).collect::<String>())
        };
        let ok = |v: Option<f64>| v.filter(|v| v.is_finite() && *v >= 0.0);
        Self {
            name: cut(&self.name),
            version: cut(&self.version),
            build: cut(&self.build),
            decode_ms: ok(self.decode_ms),
            detect_ms: ok(self.detect_ms),
        }
    }
}

/// Why a request was refused before the solver ran.
#[derive(Clone, Debug, PartialEq)]
pub struct Invalid {
    pub code: &'static str,
    pub message: String,
}

fn invalid(code: &'static str, message: impl Into<String>) -> Invalid {
    Invalid {
        code,
        message: message.into(),
    }
}

impl SolveRequest {
    /// Check the request and turn it into named star lists in attempt order.
    pub fn validate(&self) -> Result<Vec<(String, StarField)>, Invalid> {
        let (w, h) = (self.width, self.height);
        if !(MIN_SIDE..=MAX_SIDE).contains(&w) || !(MIN_SIDE..=MAX_SIDE).contains(&h) {
            return Err(invalid(
                "bad_field",
                format!("width and height must be {MIN_SIDE}..{MAX_SIDE} pixels"),
            ));
        }
        let lists: Vec<(String, &[Row])> = match (&self.stars, self.attempts.is_empty()) {
            (Some(s), true) => vec![(ATTEMPTS[0].0.to_string(), &s[..])],
            (None, false) => self
                .attempts
                .iter()
                .map(|a| (a.name.clone(), &a.stars[..]))
                .collect(),
            (Some(_), false) => {
                return Err(invalid(
                    "bad_request",
                    "send `stars` or `attempts`, not both",
                ));
            }
            (None, true) => return Err(invalid("bad_request", "no stars")),
        };
        if lists.len() > MAX_ATTEMPTS {
            return Err(invalid(
                "too_many_attempts",
                format!("at most {MAX_ATTEMPTS} attempts"),
            ));
        }
        let mut out = Vec::with_capacity(lists.len());
        for (name, rows) in lists {
            if attempt_share(&name).is_none() {
                return Err(invalid("bad_field", format!("unknown attempt `{name}`")));
            }
            if out.iter().any(|(n, _)| *n == name) {
                return Err(invalid("bad_field", format!("attempt `{name}` twice")));
            }
            if rows.len() > MAX_STARS {
                return Err(invalid(
                    "too_many_stars",
                    format!("at most {MAX_STARS} stars per attempt"),
                ));
            }
            // detections sit inside the frame; synthetic and distorted lists may reach a little
            // past the edge, which the solver handles, so allow a margin of 10 % of the frame
            let m = (0.1 * w.max(h) as f64).max(16.0);
            let (xmin, ymin, xmax, ymax) =
                (-0.5 - m, -0.5 - m, w as f64 - 0.5 + m, h as f64 - 0.5 + m);
            for (i, r) in rows.iter().enumerate() {
                let finite = r.x.is_finite()
                    && r.y.is_finite()
                    && r.flux.is_none_or(f64::is_finite)
                    && r.sigma.is_none_or(|s| s.is_finite() && s >= 0.0);
                if !finite {
                    return Err(invalid(
                        "bad_field",
                        format!("star {i}: not a finite number"),
                    ));
                }
                if r.x < xmin || r.y < ymin || r.x > xmax || r.y > ymax {
                    return Err(invalid("bad_field", format!("star {i}: outside the frame")));
                }
            }
            out.push((name, field_from_rows(w, h, rows, self.epoch)));
        }
        // keep the attempt order of the plan
        out.sort_by_key(|(n, _)| ATTEMPTS.iter().position(|a| a.0 == n));
        Ok(out)
    }

    /// The request's sky mask, checked: a well-formed grid that covers the frame (to within a
    /// cell) from the frame's corner. `Ok(None)` without one.
    pub fn validated_mask(&self) -> Result<Option<SkyMask>, Invalid> {
        let Some(m) = &self.mask else {
            return Ok(None);
        };
        m.validate()
            .map_err(|e| invalid("bad_field", format!("mask: {e}")))?;
        let (w, h) = (self.width as f64, self.height as f64);
        let (gw, gh) = (m.cols as f64 * m.cell[0], m.rows as f64 * m.cell[1]);
        let covers = m.origin == [0.0, 0.0]
            && gw >= w - 1.0
            && gh >= h - 1.0
            && gw < w + m.cell[0] + 1.0
            && gh < h + m.cell[1] + 1.0;
        if !covers {
            return Err(invalid("bad_field", "mask: the grid must cover the frame"));
        }
        Ok(Some(m.clone().trimmed()))
    }
}

/// What a solved frame's quality looks like to a user.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Quality {
    pub n_matches: usize,
    pub rms_px: f64,
    /// `high` (>= 20 matches and RMS <= 1 px) or `ok`; fixed thresholds, not the evidence score.
    pub confidence: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Timing {
    /// Time spent waiting for a free solver (the server was busy), not solving.
    #[serde(default)]
    pub queue_ms: f64,
    /// Wall time of the solver alone, measured on the server.
    pub solve_ms: f64,
    /// Request received to response ready, on the server (includes the queue).
    pub total_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CatalogInfo {
    pub name: String,
    /// Identifies the index build (results can differ between builds).
    pub index_id: String,
}

/// `[x, y, ra, dec, px, py, residual_px]`: the detection, the catalogue star and its projection
/// through the final WCS (zero-based pixels, degrees).
pub type MatchRow = [f64; 7];

/// Response of `POST /v1/solve`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicSolution {
    pub id: String,
    /// `solved` or `unsolved` (`ambiguous` is reserved).
    pub status: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub deep: bool,
    /// The attempt that solved, or the last one tried.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempt: Option<String>,
    pub width: u32,
    pub height: u32,
    pub wcs: Option<Wcs>,
    pub field: Option<FieldSummary>,
    pub quality: Option<Quality>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub matches: Vec<MatchRow>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wcs_fits: Option<String>,
    /// Atmospheric refraction measured on a wide landscape: the zenith (local sidereal time and
    /// latitude) and the refraction strength.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refraction: Option<crate::types::Refraction>,
    /// When unsolved: the attempt to extract and send next.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next: Option<String>,
    pub timing: Timing,
    pub catalog: CatalogInfo,
    /// Owner keys on a server started with `--owner-debug` only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub debug: Option<serde_json::Value>,
}

impl PublicSolution {
    pub fn solved(&self) -> bool {
        self.status == "solved" && self.wcs.is_some()
    }
}

impl PublicSolution {
    /// Project a solver result to what a client may see.
    pub fn from_solution(
        id: String,
        sol: &Solution,
        attempt: Option<&str>,
        options: &SolveOptions,
        catalog: CatalogInfo,
    ) -> Self {
        let solved = sol.status == "solved" && sol.wcs.is_some();
        let wcs = sol.wcs.clone().filter(|_| solved);
        let matches = match (&wcs, options.matches) {
            (Some(w), true) => sol
                .matches
                .iter()
                .filter_map(|m| {
                    let p = w.sky_to_pixel(m.ra, m.dec)?;
                    let row = [m.x, m.y, m.ra, m.dec, p[0], p[1], m.residual_px];
                    row.iter().all(|v| v.is_finite()).then_some(row)
                })
                .collect(),
            _ => vec![],
        };
        let outcome = compat::Outcome::from(sol);
        Self {
            id,
            status: if solved { "solved" } else { "unsolved" }.into(),
            deep: sol.stats.deep_search,
            attempt: attempt.map(str::to_string),
            width: sol.width,
            height: sol.height,
            field: wcs
                .as_ref()
                .map(|w| field::field_summary_in(w, sol.width, sol.height, sol.sky_extent)),
            quality: solved.then(|| Quality {
                n_matches: sol.n_matches,
                rms_px: (sol.rms_px * 1000.0).round() / 1000.0,
                confidence: if sol.n_matches >= 20 && sol.rms_px <= 1.0 {
                    "high"
                } else {
                    "ok"
                }
                .into(),
            }),
            wcs_fits: match (&wcs, options.fits) {
                (Some(w), true) => Some(compat::header(w, &outcome)),
                _ => None,
            },
            refraction: sol.refraction.clone().filter(|_| solved),
            wcs,
            matches,
            next: if solved {
                None
            } else {
                attempt.and_then(next_attempt).map(str::to_string)
            },
            timing: Timing::default(),
            catalog,
            debug: None,
        }
    }

    /// A solver-shaped result for code written against `Solution` (the local panel and the CLI
    /// writers in remote mode). The search statistics and the evidence score come from the
    /// `debug` block when the server sent one, otherwise they are empty.
    pub fn to_solution(&self) -> Solution {
        let debug = self.debug.as_ref();
        let num = |k: &str| debug.and_then(|d| d.get(k)).and_then(|v| v.as_f64());
        let mut stats: crate::solve::SolveStats = debug
            .and_then(|d| d.get("stats"))
            .and_then(|s| serde_json::from_value(s.clone()).ok())
            .unwrap_or_default();
        stats.deep_search = self.deep;
        let matches = self
            .matches
            .iter()
            .map(|m| crate::solve::MatchOut {
                x: m[0],
                y: m[1],
                ra: m[2],
                dec: m[3],
                residual_px: m[6],
            })
            .collect();
        match (&self.wcs, self.solved()) {
            (Some(w), true) => {
                let c = w.pixel_to_sky(
                    (self.width as f64 - 1.0) / 2.0,
                    (self.height as f64 - 1.0) / 2.0,
                );
                let q = self.quality.as_ref();
                Solution {
                    status: "solved".into(),
                    width: self.width,
                    height: self.height,
                    ra: c[0],
                    dec: c[1],
                    scale_arcsec: w.scale_arcsec(),
                    rotation_deg: w.rotation_deg(),
                    parity: if w.determinant() > 0.0 { 1 } else { -1 },
                    // the server's field width (measured over the open sky of a masked frame)
                    fov_width_deg: self
                        .field
                        .as_ref()
                        .map_or_else(|| w.width_deg(self.width, self.height), |f| f.width_deg),
                    n_matches: q.map_or(0, |q| q.n_matches),
                    n_catalog: num("n_catalog").unwrap_or(0.0) as usize,
                    rms_px: q.map_or(f64::NAN, |q| q.rms_px),
                    log_odds: num("log_odds").unwrap_or(f64::NAN),
                    cluster_anchors: num("cluster_anchors").unwrap_or(0.0) as usize,
                    verify_level: num("verify_level").unwrap_or(0.0) as u32,
                    stats,
                    matches,
                    wcs: Some(w.clone()),
                    margin_nats: num("margin_nats").unwrap_or(f64::NAN),
                    n_candidates: num("n_candidates").unwrap_or(1.0) as u32,
                    window: debug
                        .and_then(|d| d.get("window"))
                        .and_then(|v| serde_json::from_value(v.clone()).ok()),
                    refraction: self.refraction.clone(),
                    sky_extent: None,
                }
            }
            _ => Solution {
                status: "unsolved".into(),
                width: self.width,
                height: self.height,
                wcs: None,
                ra: f64::NAN,
                dec: f64::NAN,
                scale_arcsec: f64::NAN,
                rotation_deg: f64::NAN,
                parity: 0,
                fov_width_deg: f64::NAN,
                n_matches: 0,
                n_catalog: 0,
                rms_px: f64::NAN,
                log_odds: 0.0,
                cluster_anchors: 0,
                verify_level: 0,
                stats,
                matches: vec![],
                margin_nats: f64::NAN,
                n_candidates: 0,
                window: None,
                refraction: None,
                sky_extent: None,
            },
        }
    }
}

/// Error body of every non-2xx response: `{"error": message, "code": code, "detail"?}`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ErrorBody {
    pub error: String,
    pub code: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[cfg(test)]
mod tests {
    #[test]
    fn solve_mode_is_backward_compatible_and_validated() {
        let defaults: super::SolveOptions = serde_json::from_str("{}").unwrap();
        assert_eq!(defaults.mode, super::SolveMode::Auto);
        assert!(
            serde_json::to_value(defaults)
                .unwrap()
                .get("mode")
                .is_none()
        );
        let deep: super::SolveOptions = serde_json::from_str(r#"{"mode":"deep"}"#).unwrap();
        assert_eq!(deep.mode, super::SolveMode::Deep);
        assert_eq!(serde_json::to_value(deep).unwrap()["mode"], "deep");
        assert!(serde_json::from_str::<super::SolveOptions>(r#"{"mode":"anything"}"#).is_err());
    }
    use super::*;

    fn star(x: f64, y: f64, flux: f64, sigma: f64) -> Star {
        Star {
            x,
            y,
            flux,
            sigma,
            snr: 10.0,
            fwhm: 3.0,
            elongation: 1.0,
            saturated: false,
            peak: 0.0,
            pa: 0.0,
            flags: 0,
        }
    }

    #[test]
    fn panorama_capability_is_an_explicit_opt_in() {
        let old: SolveOptions = serde_json::from_str(r#"{"arc":true}"#).unwrap();
        assert!(!old.panorama);
        assert!(!serde_json::to_string(&old).unwrap().contains("panorama"));
        let new: SolveOptions = serde_json::from_str(r#"{"panorama":true}"#).unwrap();
        assert!(new.panorama);
        assert!(!new.arc);
    }

    #[test]
    fn hints_are_optional_and_roundtrip_without_changing_the_requested_mode() {
        let old: SolveOptions = serde_json::from_str("{}").unwrap();
        assert!(old.hint.is_none());
        assert!(!serde_json::to_string(&old).unwrap().contains("hint"));
        let options: SolveOptions =
            serde_json::from_str(r#"{"hint":{"position":[83.8,-5.4],"scale_arcsec":0.5}}"#)
                .unwrap();
        assert_eq!(options.mode, super::SolveMode::Auto);
        let roundtrip: SolveOptions =
            serde_json::from_str(&serde_json::to_string(&options).unwrap()).unwrap();
        assert_eq!(roundtrip.hint, options.hint);
        let invalid: SolveOptions =
            serde_json::from_str(r#"{"hint":{"position":[500,95],"scale_arcsec":-1}}"#).unwrap();
        assert!(invalid.hint.unwrap().sanitized().is_none());
    }

    #[test]
    fn rows_keep_the_solver_order_and_roundtrip_through_json() {
        let field = StarField {
            width: 100,
            height: 80,
            stars: vec![
                star(10.5, 20.25, 5.0, 0.1),
                star(f64::NAN, 1.0, 99.0, 0.1),
                star(30.0, 40.0, 50.0, 0.2),
                star(1.0, 2.0, 5.0, f64::NAN),
            ],
            epoch: None,
        };
        let rows = rows_from_field(&field);
        // NaN position dropped; brightest first; equal fluxes keep their order (stable)
        assert_eq!(rows.len(), 3);
        assert_eq!((rows[0].x, rows[0].flux), (30.0, Some(3.0)));
        assert_eq!((rows[1].x, rows[1].flux), (10.5, Some(2.0)));
        assert_eq!((rows[2].x, rows[2].sigma), (1.0, None));
        let json = serde_json::to_string(&rows).unwrap();
        assert_eq!(
            json,
            "[[30.0,40.0,3.0,0.2],[10.5,20.25,2.0,0.1],[1.0,2.0,1.0]]"
        );
        let back: Vec<Row> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, rows);
        let f = field_from_rows(100, 80, &back, None);
        assert_eq!(f.stars[2].sigma, DEFAULT_SIGMA);
    }

    #[test]
    fn rows_accept_two_to_four_numbers_only() {
        assert!(serde_json::from_str::<Row>("[1,2]").is_ok());
        assert!(serde_json::from_str::<Row>("[1,2,3,0.1]").is_ok());
        assert!(serde_json::from_str::<Row>("[1]").is_err());
        assert!(serde_json::from_str::<Row>("[1,2,3,4,5]").is_err());
        assert!(serde_json::from_str::<Row>("[1,\"a\"]").is_err());
    }

    #[test]
    fn validation_refuses_bad_requests() {
        let ok = SolveRequest {
            width: 100,
            height: 80,
            stars: Some(vec![Row {
                x: 1.0,
                y: 2.0,
                flux: None,
                sigma: None,
            }]),
            ..Default::default()
        };
        assert_eq!(ok.validate().unwrap()[0].0, "default");
        let code = |r: SolveRequest| r.validate().unwrap_err().code;
        assert_eq!(
            code(SolveRequest {
                width: 8,
                ..ok.clone()
            }),
            "bad_field"
        );
        let mut outside = ok.clone();
        outside.stars.as_mut().unwrap()[0].x = 100.0 + 16.0;
        assert_eq!(code(outside), "bad_field");
        let mut near_edge = ok.clone();
        near_edge.stars.as_mut().unwrap()[0].x = 103.4;
        assert!(near_edge.validate().is_ok());
        let mut many = ok.clone();
        many.stars = Some(vec![many.stars.unwrap()[0]; MAX_STARS + 1]);
        assert_eq!(code(many), "too_many_stars");
        let mut both = ok.clone();
        both.attempts = vec![AttemptIn {
            name: "default".into(),
            stars: vec![],
        }];
        assert_eq!(code(both), "bad_request");
        let unknown = SolveRequest {
            stars: None,
            attempts: vec![AttemptIn {
                name: "magic".into(),
                stars: vec![],
            }],
            ..ok.clone()
        };
        assert_eq!(code(unknown), "bad_field");
        let reordered = SolveRequest {
            stars: None,
            attempts: vec![
                AttemptIn {
                    name: "highpass_dog".into(),
                    stars: vec![],
                },
                AttemptIn {
                    name: "default".into(),
                    stars: vec![],
                },
            ],
            ..ok
        };
        let names: Vec<String> = reordered
            .validate()
            .unwrap()
            .into_iter()
            .map(|a| a.0)
            .collect();
        assert_eq!(names, ["default", "highpass_dog"]);
    }

    #[test]
    fn unknown_fields_are_refused() {
        let r =
            serde_json::from_str::<SolveRequest>(r#"{"width":100,"height":80,"stars":[],"x":1}"#);
        assert!(r.is_err());
    }

    #[test]
    fn attempt_plan_is_ordered() {
        assert_eq!(next_attempt("default"), Some("local_background_compact"));
        assert_eq!(next_attempt("highpass_dog"), None);
        assert_eq!(attempt_share("default"), Some(0.25));
    }
}
