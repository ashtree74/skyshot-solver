//! The detection attempts of the plan ([`crate::api::ATTEMPTS`]) as star lists. Client side:
//! the local pipeline, the command-line client and the WebAssembly detector all call this, so
//! an attempt name means the same detection everywhere.

use crate::{
    extract::{ExtractOptions, extract},
    image_io::{GrayImage, SourceInfo},
    types::StarField,
};
use serde::Serialize;

/// Detections a panel draws (the brightest ones).
pub const MAX_DETECTIONS: usize = 3000;

/// The star list of detection attempt `name`: the default extraction, a re-extraction against a
/// local annulus background with compact windows (nebulae, dense fields), or the
/// difference-of-Gaussians detector. `None` for an unknown name.
pub fn attempt_field(gray: &GrayImage, info: &SourceInfo, name: &str) -> Option<StarField> {
    match name {
        "default" => Some(extract(gray, &ExtractOptions::for_source(info))),
        "local_background_compact" => Some(extract(
            gray,
            &ExtractOptions {
                local_background: true,
                compact: true,
                ..ExtractOptions::for_source(info)
            },
        )),
        "highpass_dog" => Some(crate::highpass::detect_highpass(gray, 2.5, 5.0)),
        _ => None,
    }
}

/// The brightest detections of an attempt for drawing, as `[x, y, fwhm, snr]` rounded to 0.01.
#[derive(Clone, Debug, Serialize)]
pub struct Detections {
    pub total: usize,
    pub columns: [&'static str; 4],
    pub stars: Vec<[f32; 4]>,
}

pub fn detections(field: Option<&StarField>) -> Detections {
    let stars = field.map_or(&[][..], |f| &f.stars[..]);
    let r2 = |v: f64| ((v * 100.0).round() / 100.0) as f32;
    Detections {
        total: stars.len(),
        columns: ["x", "y", "fwhm", "snr"],
        stars: stars
            .iter()
            .filter(|s| s.x.is_finite() && s.y.is_finite())
            .take(MAX_DETECTIONS)
            .map(|s| {
                [
                    r2(s.x),
                    r2(s.y),
                    r2(if s.fwhm.is_finite() { s.fwhm } else { 0.0 }),
                    r2(if s.snr.is_finite() { s.snr } else { 0.0 }),
                ]
            })
            .collect(),
    }
}
