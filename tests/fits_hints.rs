//! Self-contained catalogue fixtures: real index probes and verification, without D80 or
//! external images. No test in this file silently skips for a missing local index.

use skyshot::{
    api::{SolveHint, SolveMode},
    builder::{BuildConfig, write_index},
    image_io::{self, GrayImage, SourceFormat},
    index::Index,
    pipeline,
    solve::{DeepIndex, SolveConfig, solve_field},
    sphere::{angle, unit},
    types::{Proj, Star, StarField, Wcs},
};
use std::{
    path::PathBuf,
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

struct Fixture {
    dir: PathBuf,
    index: Index,
    field: StarField,
    wcs: Wcs,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn fixture(scale: f64, deep: bool) -> Fixture {
    let dir = std::env::temp_dir().join(format!(
        "skyshot-hints-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let wcs = Wcs {
        ra: 83.8,
        dec: -5.4,
        crpix: [499.5, 374.5],
        cd: [[-scale / 3600.0, 0.0], [0.0, scale / 3600.0]],
        sip: None,
        proj: Proj::Tan,
        native_pole: None,
        local_grid: None,
    };
    let mut seed = if deep {
        0x1234_5678_9abc_def0u64
    } else {
        0x9e37_79b9_7f4a_7c15u64
    };
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut stars = vec![];
    let mut exact = vec![];
    let mut dist = vec![];
    let mut mags = vec![];
    for i in 0..800 {
        let (x, y) = (rnd() * 1600.0 - 300.0, rnd() * 1350.0 - 300.0);
        let [ra, dec] = wcs.pixel_to_sky(x, y);
        let v = unit(ra, dec);
        let d = exact
            .iter()
            .map(|&b| angle(v, b))
            .fold(std::f64::consts::PI, f64::min);
        exact.push(v);
        dist.push(d);
        let mag = 6.0 + i as f32 * 0.01;
        mags.push(mag);
        if (0.0..1000.0).contains(&x) && (0.0..750.0).contains(&y) {
            stars.push(Star {
                x,
                y,
                flux: 10f64.powf(-0.4 * mag as f64),
                sigma: 0.2,
                ..Star::default()
            });
        }
    }
    write_index(
        &BuildConfig {
            out: dir.clone(),
            profile: if deep { "deep" } else { "full" }.into(),
            r7_arcmin: if deep { 0.485 } else { 1.94 },
            dump_levels: false,
            caps: vec![],
            ..Default::default()
        },
        Instant::now(),
        &exact,
        &dist,
        &mags,
        exact.len() as u64,
        "deterministic synthetic hints fixture".into(),
        None,
    )
    .unwrap();
    let index = Index::open(&dir).unwrap();
    Fixture {
        dir,
        index,
        field: StarField {
            width: 1000,
            height: 750,
            stars,
            epoch: None,
        },
        wcs,
    }
}

fn correct(sol: &skyshot::solve::Solution, wcs: &Wcs) {
    assert_eq!(sol.status, "solved", "{sol:?}");
    assert!(
        angle(unit(sol.ra, sol.dec), unit(wcs.ra, wcs.dec)).to_degrees() * 3600.0
            / wcs.scale_arcsec()
            < 1.0
    );
    assert!((sol.scale_arcsec / wcs.scale_arcsec() - 1.0).abs() < 0.01);
}

#[test]
fn good_partial_and_wrong_hints_still_require_a_verified_solution() {
    let f = fixture(4.5, false);
    let base = SolveConfig {
        deadline_ms: 2000.0,
        windows: false,
        ..Default::default()
    };
    correct(&solve_field(&f.index, &f.field, &base), &f.wcs);
    for hint in [
        SolveHint {
            position: Some([83.8, -5.4]),
            scale_arcsec: Some(4.5),
        },
        // An ordinary mount offset and approximate sampling must stay in the hinted path.
        SolveHint {
            position: Some([84.6, -6.0]),
            scale_arcsec: Some(4.8),
        },
        SolveHint {
            position: None,
            scale_arcsec: Some(4.5),
        },
        SolveHint {
            position: Some([83.8, -5.4]),
            scale_arcsec: None,
        },
        SolveHint {
            position: Some([250.0, 60.0]),
            scale_arcsec: Some(4.5),
        },
        SolveHint {
            position: Some([83.8, -5.4]),
            scale_arcsec: Some(0.05),
        },
    ] {
        let wrong = hint.position.is_some_and(|p| p[0] == 250.0) || hint.scale_arcsec == Some(0.05);
        let sol = solve_field(
            &f.index,
            &f.field,
            &SolveConfig {
                hint: Some(hint),
                ..base.clone()
            },
        );
        correct(&sol, &f.wcs);
        assert!(sol.stats.hint_used);
        assert_eq!(sol.stats.hint_fallback, wrong);
    }
    let invalid = solve_field(
        &f.index,
        &f.field,
        &SolveConfig {
            hint: Some(SolveHint {
                position: Some([f64::NAN, 100.0]),
                scale_arcsec: Some(-2.0),
            }),
            ..base.clone()
        },
    );
    correct(&invalid, &f.wcs);
    assert!(!invalid.stats.hint_used);
    let outside_bounds = solve_field(
        &f.index,
        &f.field,
        &SolveConfig {
            hint: Some(SolveHint {
                position: Some([83.8, -5.4]),
                scale_arcsec: Some(4.5),
            }),
            min_fov_deg: 3.0,
            ..base.clone()
        },
    );
    assert_eq!(
        outside_bounds.status, "unsolved",
        "file hints must respect explicit field bounds"
    );
    let mut noise = f.field.clone();
    for (i, s) in noise.stars.iter_mut().enumerate() {
        s.x = (i as f64 * 0.61803398875).fract() * 1000.0;
        s.y = (i as f64 * 0.41421356237).fract() * 750.0;
    }
    let negative = solve_field(
        &f.index,
        &noise,
        &SolveConfig {
            hint: Some(SolveHint {
                position: Some([83.8, -5.4]),
                scale_arcsec: Some(4.5),
            }),
            ..base
        },
    );
    assert_eq!(negative.status, "unsolved");
    assert!(negative.wcs.is_none());
    assert!(negative.stats.hint_used && negative.stats.hint_fallback);
}

#[test]
fn auto_tries_deep_for_small_fits_scale_and_recovers_from_false_deep_hint() {
    let narrow = fixture(0.5, true);
    let normal = fixture(4.5, false);
    let cfg = SolveConfig {
        deadline_ms: 2000.0,
        windows: false,
        deep_index: Some(DeepIndex(Arc::new(Index::open(&narrow.dir).unwrap()))),
        ..Default::default()
    };
    let hint = SolveHint {
        position: Some([83.8, -5.4]),
        scale_arcsec: Some(0.5),
    };
    let sol = solve_field(
        &normal.index,
        &narrow.field,
        &SolveConfig {
            hint: Some(hint),
            ..cfg.clone()
        },
    );
    correct(&sol, &narrow.wcs);
    assert!(sol.stats.hint_used && !sol.stats.hint_fallback && sol.stats.deep_search);
    let direct = solve_field(
        &narrow.index,
        &narrow.field,
        &SolveConfig {
            hint: Some(hint),
            windows: false,
            ..Default::default()
        },
    );
    correct(&direct, &narrow.wcs);
    assert!(direct.stats.hint_used && !direct.stats.hint_fallback && direct.stats.deep_search);
    let wrong = solve_field(
        &normal.index,
        &normal.field,
        &SolveConfig {
            hint: Some(hint),
            ..cfg.clone()
        },
    );
    correct(&wrong, &normal.wcs);
    assert!(wrong.stats.hint_used && wrong.stats.hint_fallback);
    assert!(
        !wrong.stats.deep_search,
        "a failed DEEP hint must restore the ordinary index"
    );
    let standard = solve_field(
        &normal.index,
        &normal.field,
        &SolveConfig {
            mode: SolveMode::Standard,
            hint: Some(hint),
            ..cfg
        },
    );
    correct(&standard, &normal.wcs);
    assert!(!standard.stats.deep_search);
}

#[test]
fn decoded_fits_hints_reach_the_image_pipeline() {
    let f = fixture(4.5, false);
    let mut img = GrayImage {
        width: 1000,
        height: 750,
        pixels: vec![1000.0; 750_000],
        epoch: None,
    };
    for (i, p) in img.pixels.iter_mut().enumerate() {
        *p += ((i * 37 % 11) as f32 - 5.0) * 0.2;
    }
    for s in &f.field.stars {
        for y in (s.y as i32 - 6).max(0)..=(s.y as i32 + 6).min(749) {
            for x in (s.x as i32 - 6).max(0)..=(s.x as i32 + 6).min(999) {
                let r2 = (x as f64 - s.x).powi(2) + (y as f64 - s.y).powi(2);
                img.pixels[y as usize * 1000 + x as usize] +=
                    (s.flux * 8e6 * (-r2 / 3.0).exp()) as f32;
            }
        }
    }
    let mut bytes = vec![];
    for c in [
        "SIMPLE  = T",
        "BITPIX  = -32",
        "NAXIS   = 2",
        "NAXIS1  = 1000",
        "NAXIS2  = 750",
        "OBJCTRA = '05 35 12'",
        "OBJCTDEC= '-05 24 00'",
        "PIXSCALE= 4.5",
        "END",
    ] {
        bytes.extend(format!("{c:<80}").bytes());
    }
    bytes.resize(bytes.len().div_ceil(2880) * 2880, b' ');
    for p in &img.pixels {
        bytes.extend(p.to_be_bytes());
    }
    bytes.resize(bytes.len().div_ceil(2880) * 2880, 0);
    let (gray, info) = image_io::decode_bytes(&bytes).unwrap();
    assert_eq!(info.format, SourceFormat::Fits);
    let pos = info.hint.unwrap().position.unwrap();
    assert!((pos[0] - 83.8).abs() < 1e-10 && (pos[1] + 5.4).abs() < 1e-10);
    let cfg = SolveConfig {
        deadline_ms: 2000.0,
        auto_mask: false,
        windows: false,
        ..Default::default()
    };
    let r = pipeline::solve_gray(&f.index, &gray, &info, &cfg);
    correct(&r.solution, &f.wcs);
    assert!(r.solution.stats.hint_used && !r.solution.stats.hint_fallback);
    // Caller-supplied astrometry overrides the file, but stale caller data also falls back.
    let explicit = pipeline::solve_gray(
        &f.index,
        &gray,
        &info,
        &SolveConfig {
            hint: Some(SolveHint {
                position: Some([250.0, 60.0]),
                scale_arcsec: Some(0.05),
            }),
            ..cfg
        },
    );
    correct(&explicit.solution, &f.wcs);
    assert!(explicit.solution.stats.hint_fallback);
}
