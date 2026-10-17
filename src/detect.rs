use crate::{
    image_io::GrayImage,
    types::{Star, StarField},
};
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};
fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let n = v.len() / 2;
    v.select_nth_unstable_by(n, f32::total_cmp).1.to_owned()
}

pub fn detect(image: &GrayImage, cancel: &AtomicBool, deadline: Instant) -> StarField {
    detect_aperture(image, cancel, deadline, 5, false)
}

/// Alternate measurement for undersampled stars and structured nebular backgrounds.
pub fn detect_compact(image: &GrayImage, cancel: &AtomicBool, deadline: Instant) -> StarField {
    detect_aperture(image, cancel, deadline, 2, false)
}
/// Measure sources against a surrounding annulus on structured backgrounds.
pub fn detect_local(image: &GrayImage, cancel: &AtomicBool, deadline: Instant) -> StarField {
    detect_aperture(image, cancel, deadline, 5, true)
}
fn detect_aperture(
    image: &GrayImage,
    cancel: &AtomicBool,
    deadline: Instant,
    aperture: usize,
    local_background: bool,
) -> StarField {
    let factor = (image.width.max(image.height) / 2400).max(1) as usize;
    let (w, h) = (
        image.width as usize / factor,
        image.height as usize / factor,
    );
    let mut data = vec![f32::NAN; w * h];
    for y in 0..h {
        if y % 32 == 0 && (cancel.load(Ordering::Relaxed) || Instant::now() >= deadline) {
            break;
        }
        for x in 0..w {
            let mut sum = 0.0;
            let mut n = 0;
            for dy in 0..factor {
                for dx in 0..factor {
                    let v =
                        image.pixels[(y * factor + dy) * image.width as usize + x * factor + dx];
                    if v.is_finite() {
                        sum += v;
                        n += 1;
                    }
                }
            }
            if n > 0 {
                data[y * w + x] = sum / n as f32;
            }
        }
    }
    let (nx, ny) = (w.div_ceil(64), h.div_ceil(64));
    let mut backgrounds = vec![(0f32, 1f32); nx * ny];
    for by in 0..ny {
        for bx in 0..nx {
            let mut sample = vec![];
            for y in (by * 64..((by + 1) * 64).min(h)).step_by(2) {
                for x in (bx * 64..((bx + 1) * 64).min(w)).step_by(2) {
                    let v = data[y * w + x];
                    if v.is_finite() {
                        sample.push(v);
                    }
                }
            }
            let bg = median(&mut sample);
            for v in &mut sample {
                *v = (*v - bg).abs();
            }
            let noise = (median(&mut sample) * 1.4826).max(bg.abs() * 1e-6 + 1e-8);
            backgrounds[by * nx + bx] = (bg, noise);
        }
    }
    let mut peaks = vec![];
    for y in 2..h.saturating_sub(2) {
        if y % 32 == 0 && (cancel.load(Ordering::Relaxed) || Instant::now() >= deadline) {
            break;
        }
        for x in 2..w.saturating_sub(2) {
            let v = data[y * w + x];
            let (bg, noise) = backgrounds[(y / 64) * nx + x / 64];
            if !v.is_finite() || v < bg + 5.0 * noise {
                continue;
            }
            let mut largest = true;
            let mut support = 0;
            for yy in y - 1..=y + 1 {
                for xx in x - 1..=x + 1 {
                    let q = data[yy * w + xx];
                    if q > v || (q == v && (yy * w + xx) < y * w + x) {
                        largest = false;
                    }
                    if q > bg + 2.0 * noise {
                        support += 1;
                    }
                }
            }
            if largest && support >= 3 {
                peaks.push((v - bg, x, y));
            }
        }
    }
    peaks.sort_by(|a, b| b.0.total_cmp(&a.0));
    peaks.truncate(20000);
    let mut stars: Vec<Star> = vec![];
    let mut measured = rstar::RTree::<rstar::primitives::GeomWithData<[f64; 2], f64>>::new();
    let mut occupied = std::collections::HashSet::new();
    for (_, x, y) in peaks {
        if cancel.load(Ordering::Relaxed) || Instant::now() >= deadline {
            break;
        }
        if occupied.contains(&(x, y)) {
            continue;
        }
        let (mut bg, noise) = backgrounds[(y / 64) * nx + x / 64];
        let r = aperture;
        if local_background {
            let outer = r + 3;
            let mut annulus = Vec::with_capacity(128);
            for yy in y.saturating_sub(outer)..=(y + outer).min(h - 1) {
                for xx in x.saturating_sub(outer)..=(x + outer).min(w - 1) {
                    if xx.abs_diff(x).max(yy.abs_diff(y)) > r {
                        let value = data[yy * w + xx];
                        if value.is_finite() {
                            annulus.push(value);
                        }
                    }
                }
            }
            if annulus.len() < 16 {
                continue;
            }
            bg = median(&mut annulus);
            if data[y * w + x] < bg + 5.0 * noise {
                continue;
            }
        }
        let (x0, x1, y0, y1) = (
            x.saturating_sub(r),
            (x + r).min(w - 1),
            y.saturating_sub(r),
            (y + r).min(h - 1),
        );
        let (mut flux, mut sx, mut sy) = (0.0, 0.0, 0.0);
        let mut support = 0;
        for yy in y0..=y1 {
            for xx in x0..=x1 {
                let value = data[yy * w + xx];
                if !value.is_finite() {
                    continue;
                }
                let f = (value - bg - 1.5 * noise).max(0.0) as f64;
                if f > 0.0 {
                    flux += f;
                    sx += f * xx as f64;
                    sy += f * yy as f64;
                    support += 1;
                }
            }
        }
        if flux <= 0.0 || support < 3 {
            continue;
        }
        let (cx, cy) = (sx / flux, sy / flux);
        if (cx - x as f64).hypot(cy - y as f64) > 2.5 {
            continue;
        }
        let (mut xxm, mut yym, mut xym) = (0.0, 0.0, 0.0);
        for yy in y0..=y1 {
            for xx in x0..=x1 {
                let v = data[yy * w + xx];
                if !v.is_finite() {
                    continue;
                }
                let f = (v - bg - 1.5 * noise).max(0.0) as f64;
                let (dx, dy) = (xx as f64 - cx, yy as f64 - cy);
                xxm += f * dx * dx;
                yym += f * dy * dy;
                xym += f * dx * dy;
            }
        }
        xxm /= flux;
        yym /= flux;
        xym /= flux;
        let disc = ((xxm - yym).powi(2) + 4.0 * xym * xym).sqrt();
        let minor = ((xxm + yym - disc) / 2.0).max(0.02);
        let major = ((xxm + yym + disc) / 2.0).max(minor);
        let elongation = (major / minor).sqrt();
        if elongation > 5.0 || minor < 0.06 {
            continue;
        }
        let fwhm = 2.35482 * ((major + minor) / 2.0).sqrt() * factor as f64;
        if local_background {
            let point = [cx, cy];
            let radius = (fwhm / factor as f64 * 0.75).max(2.0);
            if measured
                .locate_within_distance(point, radius * radius)
                .any(|p| {
                    let limit = (fwhm.min(p.data) / factor as f64 * 0.75).max(2.0);
                    (p.geom()[0] - cx).hypot(p.geom()[1] - cy) < limit
                })
            {
                continue;
            }
            measured.insert(rstar::primitives::GeomWithData::new(point, fwhm));
        }
        let snr = flux / (noise as f64 * (support as f64).sqrt());
        // Flat-top proxy: clipping levels are not reliably recorded in stacks/FITS.
        // Five nearly identical core pixels distinguish a plateau from a symmetric
        // four-pixel Gaussian peak. Keep the source, but inflate its uncertainty.
        let peak = data[y * w + x];
        let tolerance = (peak.abs() * 1e-6).max(noise * 0.05);
        let (ix, iy) = (cx.round() as usize, cy.round() as usize);
        let mut plateau = 0;
        for yy in iy.saturating_sub(1)..=(iy + 1).min(h - 1) {
            for xx in ix.saturating_sub(1)..=(ix + 1).min(w - 1) {
                if (data[yy * w + xx] - peak).abs() <= tolerance {
                    plateau += 1;
                }
            }
        }
        let saturated = snr > 50.0 && plateau >= 5;
        stars.push(Star {
            x: (cx + 0.5) * factor as f64 - 0.5,
            y: (cy + 0.5) * factor as f64 - 0.5,
            flux: flux * (factor * factor) as f64,
            sigma: (fwhm / (2.35 * snr)).clamp(if saturated { 0.5 } else { 0.08 }, 1.5),
            snr,
            fwhm,
            elongation,
            saturated,
            peak: (peak - bg) as f64,
            pa: (0.5 * (2.0 * xym).atan2(xxm - yym))
                .to_degrees()
                .rem_euclid(180.0),
            flags: if saturated {
                crate::types::star_flags::SATURATED
            } else {
                0
            },
        });
        for yy in y.saturating_sub(2)..=(y + 2).min(h - 1) {
            for xx in x.saturating_sub(2)..=(x + 2).min(w - 1) {
                occupied.insert((xx, yy));
            }
        }
        if stars.len() >= 5000 {
            break;
        }
    }
    StarField {
        width: image.width,
        height: image.height,
        stars,
        epoch: image.epoch,
    }
}

#[cfg(test)]
mod aperture_tests {
    use super::*;
    #[test]
    fn local_background_preserves_star_on_nebula_and_merges_clipped_peaks() {
        let image = GrayImage {
            width: 128,
            height: 96,
            epoch: None,
            pixels: (0..128 * 96)
                .map(|i| {
                    let x = (i % 128) as f64;
                    let y = (i / 128) as f64;
                    let nebula = 70.0 * (-((x - 46.3).powi(2) + (y - 45.2).powi(2)) / 220.0).exp();
                    let star = 130.0 * (-((x - 46.3).powi(2) + (y - 45.2).powi(2)) / 2.0).exp();
                    let clipped = (180.0
                        * (-((x - 96.0).powi(2) + (y - 46.0).powi(2)) / 24.0).exp())
                    .min(75.0);
                    // Small compression-like ripples split the saturated plateau.
                    let ripple = if clipped >= 75.0 {
                        0.3 * (x * 2.1).cos() * (y * 2.1).cos()
                    } else {
                        0.0
                    };
                    (10.0 + 0.02 * x + nebula + star + clipped + ripple) as f32
                })
                .collect(),
        };
        let token = AtomicBool::new(false);
        let deadline = Instant::now() + std::time::Duration::from_secs(2);
        let normal = detect(&image, &token, deadline);
        let local = detect_local(&image, &token, deadline);
        let error = |f: &StarField| {
            f.stars
                .iter()
                .map(|s| (s.x - 46.3).hypot(s.y - 45.2))
                .fold(f64::INFINITY, f64::min)
        };
        assert!(error(&local) < 0.6, "local error {}", error(&local));
        let flux = |f: &StarField| {
            f.stars
                .iter()
                .filter(|s| (s.x - 46.3).hypot(s.y - 45.2) < 1.0)
                .map(|s| s.flux)
                .fold(0.0, f64::max)
        };
        assert!(flux(&normal) > flux(&local) * 2.0);
        let peaks = |f: &StarField| {
            f.stars
                .iter()
                .filter(|s| (s.x - 96.0).hypot(s.y - 46.0) < 6.0)
                .count()
        };
        assert_eq!(peaks(&local), 1);
        assert!(peaks(&normal) > 1);
    }

    #[test]
    fn local_background_honors_mask_cancellation_and_deadline() {
        let image = GrayImage {
            width: 32,
            height: 24,
            pixels: vec![f32::NAN; 32 * 24],
            epoch: None,
        };
        let token = AtomicBool::new(false);
        assert!(
            detect_local(
                &image,
                &token,
                Instant::now() + std::time::Duration::from_secs(1)
            )
            .stars
            .is_empty()
        );
        assert!(
            detect_local(&image, &token, Instant::now())
                .stars
                .is_empty()
        );
        assert!(
            detect_local(
                &image,
                &AtomicBool::new(true),
                Instant::now() + std::time::Duration::from_secs(1)
            )
            .stars
            .is_empty()
        );
    }

    #[test]
    fn compact_aperture_separates_close_point_sources() {
        let image = GrayImage {
            width: 64,
            height: 64,
            epoch: None,
            pixels: (0..4096)
                .map(|i| {
                    let x = (i % 64) as f64;
                    let y = (i / 64) as f64;
                    (10.0
                        + 0.01 * x
                        + 100.0 * (-((x - 25.3).powi(2) + (y - 27.2).powi(2)) / 1.62).exp()
                        + 60.0 * (-((x - 29.8).powi(2) + (y - 27.2).powi(2)) / 1.62).exp())
                        as f32
                })
                .collect(),
        };
        let token = AtomicBool::new(false);
        let deadline = Instant::now() + std::time::Duration::from_secs(2);
        let wide = detect(&image, &token, deadline);
        let compact = detect_compact(&image, &token, deadline);
        let error = |f: &StarField| {
            f.stars
                .iter()
                .map(|s| (s.x - 25.3).hypot(s.y - 27.2))
                .fold(f64::INFINITY, f64::min)
        };
        assert!(error(&compact) < 0.2, "compact error {}", error(&compact));
        assert!(error(&wide) > 0.6, "wide error {}", error(&wide));
    }
}
