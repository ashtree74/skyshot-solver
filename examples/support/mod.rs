//! Deterministic artificial sky for examples and integration tests; no external catalogue.
use anyhow::Result;
use skyshot::{
    builder::{BuildConfig, write_index},
    sphere::{angle, unit},
    types::{Proj, Star, StarField, Wcs},
};
use std::{path::Path, time::Instant};

pub fn synthetic_catalog(dir: &Path) -> Result<(StarField, Wcs)> {
    std::fs::create_dir(dir)?;
    let wcs = Wcs {
        ra: 83.8,
        dec: -5.4,
        crpix: [499.5, 374.5],
        cd: [[-3.0 / 3600.0, 0.0], [0.0, 3.0 / 3600.0]],
        sip: None,
        proj: Proj::Tan,
        native_pole: None,
        local_grid: None,
    };
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let (mut stars, mut exact, mut dist, mut mags) = (vec![], vec![], vec![], vec![]);
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
            out: dir.to_owned(),
            dump_levels: false,
            caps: vec![],
            ..Default::default()
        },
        Instant::now(),
        &exact,
        &dist,
        &mags,
        exact.len() as u64,
        "synthetic-demo-v1; artificial stars, not a real sky catalogue".into(),
        None,
    )?;
    Ok((
        StarField {
            width: 1000,
            height: 750,
            stars,
            epoch: None,
        },
        wcs,
    ))
}
