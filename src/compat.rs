//! Drop-in output contracts so capture/processing tools can call Skyshot Solver like existing
//! solvers. Position/scale hints can order searches but never change acceptance; a failed
//! hinted search falls back to blind solving.
//!
//! * ASTAP (`astap_cli -f file [-ra h] [-spd deg] [-fov deg] [-r deg] [-z n] [-s n] [-wcs] [-o base]`):
//!   writes `<base>.ini` (PLTSOLVD=T/F, CRVAL, CDELT, CROTA, CD, ...) and, on success with `-wcs`
//!   or always, `<base>.wcs` (FITS header). Exit code 0 when solved, 1 otherwise.
//! * astrometry.net solve-field: writes `<base>.wcs` and an empty `<base>.solved` on success.
//!
//! The writers take an [`Outcome`]: what a user may see of a result (status, WCS, size, matches,
//! RMS). They write no search statistics or evidence score, so the same files come out of a
//! local solve and of the solver API.

use crate::solve::Solution;
use crate::{api::PublicSolution, types::Wcs};
use anyhow::Result;
use std::{fs, path::Path};

/// Classic CDELT/CROTA decomposition of a CD matrix (FITS paper II, as written by ASTAP).
pub fn cdelt_crota(cd: [[f64; 2]; 2]) -> (f64, f64, f64, f64) {
    let det = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
    let sign = if det < 0.0 { -1.0 } else { 1.0 };
    let cdelt1 = sign * (cd[0][0] * cd[0][0] + cd[1][0] * cd[1][0]).sqrt();
    let cdelt2 = (cd[0][1] * cd[0][1] + cd[1][1] * cd[1][1]).sqrt();
    let crota1 = (sign * cd[1][0]).atan2(sign * cd[0][0]).to_degrees();
    let crota2 = (-cd[0][1]).atan2(cd[1][1]).to_degrees();
    (cdelt1, cdelt2, crota1, crota2)
}

/// The public part of a result, as the output files show it.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub solved: bool,
    pub wcs: Option<Wcs>,
    pub width: u32,
    pub height: u32,
    pub n_matches: usize,
    pub rms_px: f64,
}

impl From<&Solution> for Outcome {
    fn from(s: &Solution) -> Self {
        let solved = s.status == "solved" && s.wcs.is_some();
        Self {
            solved,
            wcs: s.wcs.clone().filter(|_| solved),
            width: s.width,
            height: s.height,
            n_matches: s.n_matches,
            rms_px: s.rms_px,
        }
    }
}

impl From<&PublicSolution> for Outcome {
    fn from(s: &PublicSolution) -> Self {
        let solved = s.solved();
        Self {
            solved,
            wcs: s.wcs.clone().filter(|_| solved),
            width: s.width,
            height: s.height,
            n_matches: s.quality.as_ref().map_or(0, |q| q.n_matches),
            rms_px: s.quality.as_ref().map_or(f64::NAN, |q| q.rms_px),
        }
    }
}

fn fmt_e(v: f64) -> String {
    format!("{v:.12E}")
}

/// Write ASTAP-style `.ini` (and `.wcs` when solved). Returns true when solved.
pub fn write_astap(base: &Path, sol: &Outcome, cmdline: &str, write_wcs: bool) -> Result<bool> {
    fs::write(base.with_extension("ini"), astap_ini(sol, cmdline))?;
    if let (true, true, Some(w)) = (sol.solved, write_wcs, &sol.wcs) {
        fs::write(base.with_extension("wcs"), header(w, sol))?;
    }
    Ok(sol.solved)
}

/// ASTAP-style `.ini` contents (PLTSOLVD=T/F, CRPIX, CRVAL, CDELT, CROTA, CD, ...).
pub fn astap_ini(sol: &Outcome, cmdline: &str) -> String {
    let mut ini = String::new();
    ini.push_str(&format!(
        "PLTSOLVD={}\n",
        if sol.solved { "T" } else { "F" }
    ));
    if let (true, Some(w)) = (sol.solved, &sol.wcs) {
        let (c1, c2, r1, r2) = cdelt_crota(w.cd);
        ini.push_str(&format!("CRPIX1={}\n", fmt_e(w.crpix[0] + 1.0)));
        ini.push_str(&format!("CRPIX2={}\n", fmt_e(w.crpix[1] + 1.0)));
        ini.push_str(&format!("CRVAL1={}\n", fmt_e(w.ra)));
        ini.push_str(&format!("CRVAL2={}\n", fmt_e(w.dec)));
        ini.push_str(&format!("CDELT1={}\n", fmt_e(c1)));
        ini.push_str(&format!("CDELT2={}\n", fmt_e(c2)));
        ini.push_str(&format!("CROTA1={}\n", fmt_e(r1)));
        ini.push_str(&format!("CROTA2={}\n", fmt_e(r2)));
        ini.push_str(&format!("CD1_1={}\n", fmt_e(w.cd[0][0])));
        ini.push_str(&format!("CD1_2={}\n", fmt_e(w.cd[0][1])));
        ini.push_str(&format!("CD2_1={}\n", fmt_e(w.cd[1][0])));
        ini.push_str(&format!("CD2_2={}\n", fmt_e(w.cd[1][1])));
        ini.push_str(&format!(
            "WARNING=Skyshot Solver blind solve: {} matches, rms {:.2} px\n",
            sol.n_matches, sol.rms_px
        ));
    } else {
        ini.push_str("ERROR=No solution found\n");
    }
    ini.push_str(&format!("CMDLINE={cmdline}\n"));
    ini
}

/// Write astrometry.net-style `.wcs` + `.solved`. Returns true when solved.
pub fn write_solve_field(base: &Path, sol: &Outcome) -> Result<bool> {
    if let (true, Some(w)) = (sol.solved, &sol.wcs) {
        fs::write(base.with_extension("wcs"), header(w, sol))?;
        fs::write(base.with_extension("solved"), [1u8])?;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// FITS header (2880-byte blocks) for a solution, including image size and provenance cards.
pub fn header(w: &Wcs, sol: &Outcome) -> String {
    let base = w.fits_header();
    // insert IMAGEW/IMAGEH and provenance before END
    let mut cards: Vec<String> = base
        .as_bytes()
        .chunks(80)
        .map(|c| String::from_utf8_lossy(c).to_string())
        .filter(|c| !c.trim().is_empty() && c.trim() != "END")
        .collect();
    cards.push(format!("{:<8}= {:20}", "IMAGEW", sol.width));
    cards.push(format!("{:<8}= {:20}", "IMAGEH", sol.height));
    cards.push(format!("{:<8}= {:>20}", "PLTSOLVD", "T"));
    cards.push(format!(
        "COMMENT Skyshot Solver matches={} rms_px={:.3}",
        sol.n_matches, sol.rms_px
    ));
    cards.push("END".into());
    let mut out: String = cards
        .iter()
        .map(|c| format!("{:<80}", &c[..c.len().min(80)]))
        .collect();
    let n = out.len().div_ceil(2880) * 2880;
    out.extend(std::iter::repeat_n(' ', n - out.len()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdelt_crota_roundtrip_for_rotated_and_mirrored_cd() {
        for &(rot, s, mirror) in &[
            (0.0f64, 1e-4, false),
            (37.0, 2e-4, false),
            (-120.0, 5e-5, true),
        ] {
            // FITS Paper II: CD1_1 = c1 cos r, CD1_2 = -c2 sin r, CD2_1 = c1 sin r, CD2_2 = c2 cos r
            let r = rot.to_radians();
            let c1 = if mirror { s } else { -s };
            let cd = [[c1 * r.cos(), -s * r.sin()], [c1 * r.sin(), s * r.cos()]];
            let (d1, d2, _r1, r2) = cdelt_crota(cd);
            assert!((d1.abs() - s).abs() < 1e-15 && (d2 - s).abs() < 1e-15);
            assert!(
                ((r2 - rot + 540.0) % 360.0 - 180.0).abs() < 1e-9,
                "{r2} vs {rot}"
            );
        }
    }
}
