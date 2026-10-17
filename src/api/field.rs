//! Field geometry derived from a solved WCS: centre, size, scale, orientation, corners and the
//! sexagesimal / galactic forms the clients show. Shared by the local panel, the solver API and
//! the command-line client.

use crate::{
    compat,
    geometry::{project, unit},
    types::Wcs,
};
use serde::{Deserialize, Serialize};

/// Field geometry derived from a WCS. Directions refer to the image as stored: row 0 at the
/// top, as JPEG/PNG are shown (FITS viewers usually draw row 0 at the bottom, which mirrors
/// the frame and turns "up" by 180°).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldSummary {
    /// Sky position of the frame centre ((w - 1) / 2, (h - 1) / 2), degrees.
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub ra_hms: String,
    pub dec_dms: String,
    pub galactic_l_deg: f64,
    pub galactic_b_deg: f64,
    /// Angular size edge to edge through the centre row / column, along the row / column (a
    /// fisheye frame can span more than 180 deg).
    pub width_deg: f64,
    pub height_deg: f64,
    pub scale_arcsec: f64,
    /// Position angle of the frame's "up" (towards row 0), east of north, [0, 360).
    pub up_pa_deg: f64,
    /// True when the stored raster is a mirror image of the sky.
    pub mirrored: bool,
    /// ASTAP / FITS paper II decomposition of the CD matrix.
    pub crota2_deg: f64,
    pub cdelt1_deg: f64,
    pub cdelt2_deg: f64,
    /// Sky positions of the outer corners: top-left, top-right, bottom-right, bottom-left.
    pub corners: [[f64; 2]; 4],
    /// "N" or "S" when a celestial pole lies inside the frame.
    pub pole: Option<String>,
    /// SIP distortion order (0 = pure TAN).
    pub sip_order: u32,
}

/// Derived geometry of a solved `width` x `height` frame.
pub fn field_summary(wcs: &Wcs, width: u32, height: u32) -> FieldSummary {
    field_summary_in(wcs, width, height, None)
}

/// [`field_summary`] with the field size measured over `sky` (x0, y0, x1, y1: the open sky of a
/// masked frame, where the WCS is fitted) instead of the whole frame.
pub fn field_summary_in(wcs: &Wcs, width: u32, height: u32, sky: Option<[f64; 4]>) -> FieldSummary {
    let (w, h) = (width as f64, height as f64);
    let c = [(w - 1.0) / 2.0, (h - 1.0) / 2.0];
    let centre = wcs.pixel_to_sky(c[0], c[1]);
    // tangent-plane offset (east, north) of a one-pixel step from the centre
    let step = |dx: f64, dy: f64| {
        let s = wcs.pixel_to_sky(c[0] + dx, c[1] + dy);
        project(s[0], s[1], centre).unwrap_or([0.0; 2])
    };
    let (jx, jy) = (step(1.0, 0.0), step(0.0, 1.0));
    let up = step(0.0, -1.0);
    let (x0, y0, x1, y1) = (-0.5, -0.5, w - 0.5, h - 0.5);
    let corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(|p| wcs.pixel_to_sky(p[0], p[1]));
    let pole = [(90.0, "N"), (-90.0, "S")]
        .into_iter()
        .find(|&(dec, _)| {
            wcs.sky_to_pixel(0.0, dec)
                .is_some_and(|p| p[0] >= x0 && p[0] <= x1 && p[1] >= y0 && p[1] <= y1)
        })
        .map(|(_, name)| name.to_string());
    let (cdelt1, cdelt2, _, crota2) = compat::cdelt_crota(wcs.cd);
    let (l, b) = galactic(centre[0], centre[1]);
    FieldSummary {
        ra_deg: centre[0],
        dec_deg: centre[1],
        ra_hms: format_hms(centre[0]),
        dec_dms: format_dms(centre[1]),
        galactic_l_deg: l,
        galactic_b_deg: b,
        // along the pixel path (a fisheye frame can span more than 180 deg), over the open sky
        width_deg: match sky {
            Some(e) => {
                let y = (e[1] + e[3] - 1.0) / 2.0;
                wcs.arc_deg([e[0] - 0.5, y], [e[2] - 0.5, y])
            }
            None => wcs.arc_deg([x0, c[1]], [x1, c[1]]),
        },
        height_deg: match sky {
            Some(e) => {
                let x = (e[0] + e[2] - 1.0) / 2.0;
                wcs.arc_deg([x, e[1] - 0.5], [x, e[3] - 0.5])
            }
            None => wcs.arc_deg([c[0], y0], [c[0], y1]),
        },
        scale_arcsec: (jx[0] * jy[1] - jy[0] * jx[1]).abs().sqrt() * 3600.0,
        up_pa_deg: up[0].atan2(up[1]).to_degrees().rem_euclid(360.0),
        mirrored: jx[0] * jy[1] - jy[0] * jx[1] < 0.0,
        crota2_deg: crota2,
        cdelt1_deg: cdelt1,
        cdelt2_deg: cdelt2,
        corners,
        pole,
        sip_order: wcs.sip.as_ref().map_or(0, |s| s.order),
    }
}

/// ICRS (RA, Dec) -> galactic (l, b), degrees.
pub fn galactic(ra: f64, dec: f64) -> (f64, f64) {
    // Hipparcos / ICRS -> galactic rotation (ESA 1997, vol. 1, sect. 1.5.3)
    #[allow(clippy::excessive_precision)]
    const M: [[f64; 3]; 3] = [
        [
            -0.054_875_560_416_215_4,
            -0.873_437_090_234_885,
            -0.483_835_015_548_713_2,
        ],
        [
            0.494_109_427_875_583_7,
            -0.444_829_629_960_011_2,
            0.746_982_244_497_218_9,
        ],
        [
            -0.867_666_149_019_004_7,
            -0.198_076_373_431_201_5,
            0.455_983_776_175_066_9,
        ],
    ];
    let u = unit(ra, dec);
    let g: Vec<f64> = M
        .iter()
        .map(|r| r[0] * u[0] + r[1] * u[1] + r[2] * u[2])
        .collect();
    (
        g[1].atan2(g[0]).to_degrees().rem_euclid(360.0),
        g[2].clamp(-1.0, 1.0).asin().to_degrees(),
    )
}

/// Right ascension as `HHh MMm SS.SSs`.
pub fn format_hms(ra_deg: f64) -> String {
    const DAY: i64 = 24 * 3600 * 100;
    let cs = ((ra_deg / 15.0 * 3600.0 * 100.0).round() as i64).rem_euclid(DAY);
    format!(
        "{:02}h {:02}m {:05.2}s",
        cs / 360_000,
        cs / 6000 % 60,
        (cs % 6000) as f64 / 100.0
    )
}

/// Declination as `±DD° MM′ SS.S″`.
pub fn format_dms(dec_deg: f64) -> String {
    let ds = (dec_deg.abs() * 36_000.0).round() as i64;
    format!(
        "{}{:02}° {:02}′ {:04.1}″",
        if dec_deg < 0.0 && ds > 0 { '−' } else { '+' },
        ds / 36_000,
        ds / 600 % 60,
        (ds % 600) as f64 / 10.0
    )
}
