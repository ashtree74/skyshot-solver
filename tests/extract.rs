//! Synthetic tests for the fast extractor (`skyshot::extract`).
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use skyshot::{
    Star, StarField,
    extract::{ExtractOptions, extract},
    image_io::GrayImage,
    star_flags,
};

/// Synthetic frame: background + Gaussian noise + pixel-integrated Gaussian stars.
struct Synth {
    w: usize,
    h: usize,
    px: Vec<f32>,
    rng: ChaCha8Rng,
}

impl Synth {
    fn new(w: usize, h: usize, bg: f32, noise: f32, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let px = (0..w * h)
            .map(|_| {
                // Box-Muller
                let (u1, u2): (f32, f32) = (rng.r#gen::<f32>().max(1e-12), rng.r#gen());
                bg + noise * (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
            })
            .collect();
        Self { w, h, px, rng }
    }

    /// Add a Gaussian star of total `flux` and FWHM, integrated over pixels
    /// (5×5 sub-samples), optionally elongated along `angle` with `trail` px.
    fn star(&mut self, x: f64, y: f64, flux: f64, fwhm: f64) {
        self.trailed(x, y, flux, fwhm, 0.0, 0.0);
    }

    fn trailed(&mut self, x: f64, y: f64, flux: f64, fwhm: f64, trail: f64, angle: f64) {
        let s = fwhm / 2.3548;
        let steps = (trail.ceil() as usize * 2).max(1);
        let r = (5.0 * s + trail / 2.0).ceil() as isize;
        let (ca, sa) = (angle.cos(), angle.sin());
        let norm = flux / (2.0 * std::f64::consts::PI * s * s) / 25.0 / steps as f64;
        for yy in (y as isize - r).max(0)..=(y as isize + r).min(self.h as isize - 1) {
            for xx in (x as isize - r).max(0)..=(x as isize + r).min(self.w as isize - 1) {
                let mut v = 0.0;
                for k in 0..steps {
                    let t = if steps == 1 {
                        0.0
                    } else {
                        (k as f64 / (steps - 1) as f64 - 0.5) * trail
                    };
                    let (cx, cy) = (x + t * ca, y + t * sa);
                    for sy in 0..5 {
                        for sx in 0..5 {
                            let dx = xx as f64 - 0.4 + 0.2 * sx as f64 - cx;
                            let dy = yy as f64 - 0.4 + 0.2 * sy as f64 - cy;
                            v += (-(dx * dx + dy * dy) / (2.0 * s * s)).exp();
                        }
                    }
                }
                self.px[yy as usize * self.w + xx as usize] += (v * norm) as f32;
            }
        }
    }

    fn image(&self) -> GrayImage {
        GrayImage {
            width: self.w as u32,
            height: self.h as u32,
            pixels: self.px.clone(),
            epoch: None,
        }
    }

    /// Random positions on a grid with jitter, away from the edges.
    fn positions(&mut self, n: usize, margin: f64) -> Vec<(f64, f64)> {
        let cols = (n as f64).sqrt().ceil() as usize;
        let rows = n.div_ceil(cols);
        let (dx, dy) = (
            (self.w as f64 - 2.0 * margin) / cols as f64,
            (self.h as f64 - 2.0 * margin) / rows as f64,
        );
        (0..n)
            .map(|i| {
                let (c, r) = (i % cols, i / cols);
                (
                    margin + (c as f64 + 0.3 + 0.4 * self.rng.r#gen::<f64>()) * dx,
                    margin + (r as f64 + 0.3 + 0.4 * self.rng.r#gen::<f64>()) * dy,
                )
            })
            .collect()
    }
}

fn nearest(field: &StarField, x: f64, y: f64) -> Option<(&Star, f64)> {
    field
        .stars
        .iter()
        .map(|s| (s, (s.x - x).hypot(s.y - y)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
}

fn count_near(field: &StarField, x: f64, y: f64, r: f64) -> usize {
    field
        .stars
        .iter()
        .filter(|s| (s.x - x).hypot(s.y - y) < r)
        .count()
}

/// Flux giving SNR ≈ `snr` in a 1.5·FWHM aperture (background-limited).
fn flux_for_snr(snr: f64, noise: f64, fwhm: f64) -> f64 {
    let r = (1.5 * fwhm).max(2.5);
    snr * noise * (std::f64::consts::PI * r * r).sqrt()
}

#[test]
fn subpixel_centroids_at_snr_50() {
    for (fwhm, seed) in [(2.5, 1), (3.5, 2), (5.0, 3)] {
        let mut s = Synth::new(640, 480, 200.0, 8.0, seed);
        let truth = s.positions(48, 25.0);
        let flux = flux_for_snr(50.0, 8.0, fwhm);
        for &(x, y) in &truth {
            s.star(x, y, flux, fwhm);
        }
        let field = extract(&s.image(), &ExtractOptions::default());
        let mut sq = 0.0;
        for &(x, y) in &truth {
            let (star, d) = nearest(&field, x, y).expect("star");
            assert!(
                d < 0.5,
                "fwhm {fwhm}: star at ({x:.2},{y:.2}) off by {d:.3}"
            );
            assert!(
                (star.fwhm - fwhm).abs() < 0.35 * fwhm,
                "fwhm {} vs {fwhm}",
                star.fwhm
            );
            sq += d * d;
        }
        let rms = (sq / truth.len() as f64).sqrt();
        assert!(rms < 0.1, "fwhm {fwhm}: rms centroid error {rms:.3}");
        assert_eq!(
            field.stars.len(),
            truth.len(),
            "fwhm {fwhm}: spurious detections"
        );
        // Brightness order and uncertainty floor.
        assert!(field.stars.windows(2).all(|p| p[0].flux >= p[1].flux));
        assert!(field.stars.iter().all(|s| s.sigma >= 0.05 && s.snr > 20.0));
    }
}

#[test]
fn bright_star_centroids_are_unbiased() {
    // High SNR, slightly undersampled and well-sampled PSFs at all pixel phases.
    for fwhm in [1.8, 2.5, 4.0] {
        let mut s = Synth::new(400, 400, 100.0, 2.0, 7);
        let truth = s.positions(36, 20.0);
        for &(x, y) in &truth {
            s.star(x, y, 2.0e5, fwhm);
        }
        let field = extract(&s.image(), &ExtractOptions::default());
        let worst = truth
            .iter()
            .map(|&(x, y)| nearest(&field, x, y).unwrap().1)
            .fold(0.0, f64::max);
        assert!(
            worst < 0.05,
            "fwhm {fwhm}: worst bright-star error {worst:.3}"
        );
    }
}

#[test]
fn hot_pixels_are_rejected() {
    let mut s = Synth::new(512, 512, 300.0, 10.0, 11);
    let truth = s.positions(25, 30.0);
    for &(x, y) in &truth {
        s.star(x, y, flux_for_snr(80.0, 10.0, 3.0), 3.0);
    }
    let mut hot = vec![];
    for i in 0..60 {
        let (x, y) = (s.rng.gen_range(10..502usize), s.rng.gen_range(10..502usize));
        let amp = if i % 2 == 0 { 2000.0 } else { 60000.0 };
        s.px[y * 512 + x] += amp;
        hot.push((x as f64, y as f64));
    }
    // A few two-pixel "warm pairs" (cosmic-ray-like).
    for i in 0..5 {
        let (x, y) = (60 + 80 * i, 470);
        s.px[y * 512 + x] += 5000.0;
        s.px[y * 512 + x + 1] += 3000.0;
        hot.push((x as f64 + 0.4, y as f64));
    }
    let image = s.image();
    let field = extract(&image, &ExtractOptions::default());
    for &(x, y) in &hot {
        if truth.iter().any(|t| (t.0 - x).hypot(t.1 - y) < 8.0) {
            continue;
        }
        assert_eq!(
            count_near(&field, x, y, 2.0),
            0,
            "hot pixel at ({x},{y}) kept"
        );
    }
    for &(x, y) in &truth {
        assert!(nearest(&field, x, y).unwrap().1 < 0.3);
    }
    // With keep_rejected the hot pixels surface with their flag.
    let all = extract(
        &image,
        &ExtractOptions {
            keep_rejected: true,
            ..Default::default()
        },
    );
    assert!(
        all.stars
            .iter()
            .any(|s| s.flags & star_flags::HOT_PIXEL != 0)
    );
}

#[test]
fn clipped_cosmic_rays_on_a_well_sampled_frame() {
    // FWHM 7 px stars; 2×2, 3×3 and 1×3 clusters clipped at 65504 (as on
    // bm_1_ic1795, where a 2×2 block ranked 88th): too compact for any star of
    // this frame even though their FWHM exceeds the absolute 1.2 px limit.
    let mut s = Synth::new(600, 500, 2000.0, 60.0, 47);
    let truth = s.positions(30, 30.0);
    for (i, &(x, y)) in truth.iter().enumerate() {
        s.star(x, y, 2.0e5 + 2.0e4 * i as f64, 7.0);
    }
    let clusters: [&[(usize, usize)]; 3] = [
        &[(0, 0), (1, 0), (0, 1), (1, 1)],
        &[
            (0, 0),
            (1, 0),
            (2, 0),
            (0, 1),
            (1, 1),
            (2, 1),
            (0, 2),
            (1, 2),
            (2, 2),
        ],
        &[(0, 0), (1, 1), (2, 2)],
    ];
    let mut hits = vec![];
    for k in 0..12 {
        let (x0, y0) = (25 + 47 * k, 15 + 37 * k);
        if truth
            .iter()
            .any(|t| (t.0 - x0 as f64).hypot(t.1 - y0 as f64) < 25.0)
        {
            continue;
        }
        for &(dx, dy) in clusters[k % 3] {
            s.px[(y0 + dy) * 600 + x0 + dx] = 65504.0;
        }
        hits.push((x0 as f64 + 1.0, y0 as f64 + 1.0));
    }
    assert!(hits.len() >= 6);
    let field = extract(&s.image(), &ExtractOptions::default());
    for &(x, y) in &hits {
        assert_eq!(
            count_near(&field, x, y, 4.0),
            0,
            "cluster at ({x},{y}) kept"
        );
    }
    for &(x, y) in &truth {
        assert!(
            nearest(&field, x, y).unwrap().1 < 0.3,
            "star ({x:.0},{y:.0}) lost"
        );
    }
}

#[test]
fn saturated_disc_yields_one_detection() {
    let mut s = Synth::new(300, 300, 1000.0, 15.0, 5);
    s.star(150.37, 140.81, 6.0e7, 6.0);
    s.star(60.2, 60.7, 2.0e4, 3.0);
    let clip = 60000.0;
    for (i, v) in s.px.iter_mut().enumerate() {
        if *v > clip {
            // Clipped plateau with small compression-like ripples.
            let (x, y) = ((i % 300) as f32, (i / 300) as f32);
            *v = clip + 30.0 * (x * 2.1).cos() * (y * 1.7).cos();
        }
    }
    let field = extract(&s.image(), &ExtractOptions::default());
    assert_eq!(
        count_near(&field, 150.37, 140.81, 25.0),
        1,
        "{:?}",
        field.stars
    );
    let (star, d) = nearest(&field, 150.37, 140.81).unwrap();
    assert!(d < 0.3, "saturated centroid off by {d:.3}");
    assert!(star.saturated && star.flags & star_flags::SATURATED != 0);
    assert!(star.sigma >= 0.3);
    assert_eq!(field.stars[0].x, star.x, "saturated star ranks first");
    assert!(nearest(&field, 60.2, 60.7).unwrap().1 < 0.3);
}

#[test]
fn large_saturated_disc_is_not_background() {
    // A clipped disc ~60 px across fills its whole 64 px background tile: its
    // median must not become the sky level there (the brightest star of a
    // wide-field frame would vanish). Also with JPEG-like ripples on the top.
    for ripple in [0.0f32, 400.0] {
        let mut s = Synth::new(400, 300, 1000.0, 15.0, 41);
        let truth = s.positions(12, 25.0);
        for &(x, y) in &truth {
            if (x - 200.0).hypot(y - 150.0) > 60.0 {
                s.star(x, y, 3.0e4, 3.0);
            }
        }
        s.star(200.2, 150.4, 1.4e10, 20.0);
        let clip = 60000.0;
        for (i, v) in s.px.iter_mut().enumerate() {
            if *v > clip {
                let (x, y) = ((i % 400) as f32, (i / 400) as f32);
                *v = clip - ripple * (0.5 + 0.5 * (x * 1.3).cos() * (y * 0.9).cos());
            }
        }
        let field = extract(&s.image(), &ExtractOptions::default());
        assert_eq!(
            count_near(&field, 200.2, 150.4, 45.0),
            1,
            "ripple {ripple}: {:?}",
            field.stars
        );
        let (star, d) = nearest(&field, 200.2, 150.4).unwrap();
        assert!(d < 0.5, "ripple {ripple}: disc centroid off by {d:.2}");
        assert!(star.flags & star_flags::SATURATED != 0);
        assert_eq!(field.stars[0].x, star.x, "the disc ranks first");
        for &(x, y) in &truth {
            if (x - 200.0).hypot(y - 150.0) > 60.0 {
                assert!(nearest(&field, x, y).unwrap().1 < 0.3, "({x},{y}) lost");
            }
        }
    }
}

#[test]
fn very_noisy_stretched_frame() {
    // Stretched 8-bit frame with σ ≈ 0.11 of full scale: no star reaches 10σ
    // above the sky. Clipped discs must be centroided on their plateau (the
    // windowed centroid stays at the plateau corner where the peak search
    // lands), and the frame PSF must still be measured: otherwise saturated
    // stars with a broad halo are probed inside their own halo and rejected
    // as ridges.
    let mut s = Synth::new(800, 600, 0.25, 0.11, 43);
    let truth = s.positions(48, 30.0);
    let saturated = |i: usize| i.is_multiple_of(4);
    for (i, &(x, y)) in truth.iter().enumerate() {
        if i % 8 == 0 {
            s.star(x, y, 400.0, 9.0); // clipped disc ~14 px across
        } else if i % 8 == 4 {
            s.star(x, y, 100.0, 5.0); // smaller disc in a broad halo
            s.star(x, y, 300.0, 16.0);
        } else {
            s.star(x, y, 8.0 + 0.1 * i as f64, 3.5);
        }
    }
    let pixels: Vec<f32> =
        s.px.iter()
            .map(|v| (v * 255.0).round().clamp(0.0, 255.0) / 255.0)
            .collect();
    let image = GrayImage {
        pixels,
        ..s.image()
    };
    let field = extract(&image, &ExtractOptions::default());
    for (i, &(x, y)) in truth.iter().enumerate() {
        if saturated(i) {
            let (star, d) = nearest(&field, x, y).expect("star");
            assert!(
                d < 0.7,
                "saturated star at ({x:.1},{y:.1}): nearest detection {d:.1} px away"
            );
            assert!(star.saturated, "{star:?}");
        }
    }
    let found = truth
        .iter()
        .filter(|&&(x, y)| nearest(&field, x, y).is_some_and(|n| n.1 < 1.0))
        .count();
    assert!(found >= 40, "found {found}/48");
    // Two clipped discs that touch form one elongated clipped region: they
    // must stay two sources (as on bm_1_heart-nebula), not one plateau.
    let mut s = Synth::new(300, 200, 0.25, 0.11, 44);
    s.star(140.3, 100.6, 400.0, 9.0);
    s.star(156.8, 97.2, 400.0, 9.0);
    let pixels: Vec<f32> =
        s.px.iter()
            .map(|v| (v * 255.0).round().clamp(0.0, 255.0) / 255.0)
            .collect();
    let pair = extract(
        &GrayImage {
            pixels,
            ..s.image()
        },
        &ExtractOptions::default(),
    );
    for (x, y) in [(140.3, 100.6), (156.8, 97.2)] {
        let d = nearest(&pair, x, y).unwrap().1;
        assert!(
            d < 3.0,
            "pair member ({x},{y}): {d:.1} px, {:?}",
            pair.stars
        );
    }
}

#[test]
fn nan_masks_are_ignored() {
    let mut s = Synth::new(400, 300, 50.0, 3.0, 9);
    let truth = [(80.3, 70.6), (300.7, 220.2), (200.1, 150.9)];
    for &(x, y) in &truth {
        s.star(x, y, 3000.0, 3.0);
    }
    // Masked block, masked border columns and scattered blank pixels.
    for y in 100..200 {
        for x in 20..120 {
            s.px[y * 400 + x] = f32::NAN;
        }
    }
    for y in 0..300 {
        s.px[y * 400 + 399] = f32::NAN;
        s.px[y * 400] = f32::INFINITY;
    }
    for i in (0..s.px.len()).step_by(997) {
        s.px[i] = f32::NAN;
    }
    let field = extract(&s.image(), &ExtractOptions::default());
    for &(x, y) in &truth {
        assert!(nearest(&field, x, y).unwrap().1 < 0.3);
    }
    assert!(field.stars.len() <= truth.len() + 1, "{:?}", field.stars);
    assert!(field.stars.iter().all(|s| s.x.is_finite() && s.flux > 0.0));
    let blank = GrayImage {
        width: 64,
        height: 48,
        pixels: vec![f32::NAN; 64 * 48],
        epoch: None,
    };
    assert!(extract(&blank, &ExtractOptions::default()).stars.is_empty());
    let tiny = GrayImage {
        width: 3,
        height: 2,
        pixels: vec![1.0; 6],
        epoch: None,
    };
    assert!(extract(&tiny, &ExtractOptions::default()).stars.is_empty());
}

#[test]
fn strong_gradient_background() {
    let mut s = Synth::new(800, 600, 0.0, 6.0, 13);
    for y in 0..600 {
        for x in 0..800 {
            // Steep linear ramp plus a broad vignetting bump.
            let r2 = ((x as f32 - 400.0).powi(2) + (y as f32 - 300.0).powi(2)) / 1.0e5;
            s.px[y * 800 + x] += 200.0 + 8.0 * x as f32 + 3000.0 * (-r2).exp();
        }
    }
    let truth = s.positions(40, 25.0);
    for &(x, y) in &truth {
        s.star(x, y, flux_for_snr(40.0, 6.0, 3.0), 3.0);
    }
    let field = extract(&s.image(), &ExtractOptions::default());
    let found = truth
        .iter()
        .filter(|&&(x, y)| nearest(&field, x, y).is_some_and(|n| n.1 < 0.5))
        .count();
    assert!(found >= 39, "found {found}/40");
    assert!(field.stars.len() <= 42, "{} detections", field.stars.len());
}

#[test]
fn posterized_8bit_sky() {
    // Stretched JPEG-like frame: black point clipped, 8-bit levels.
    let mut s = Synth::new(600, 400, -0.3, 1.2, 17);
    let truth = s.positions(30, 20.0);
    for &(x, y) in &truth {
        s.star(x, y, 900.0, 2.8);
    }
    let pixels: Vec<f32> =
        s.px.iter()
            .map(|v| v.round().clamp(0.0, 255.0) / 255.0)
            .collect();
    let zeros = pixels.iter().filter(|&&v| v == 0.0).count();
    assert!(zeros * 2 > pixels.len(), "sky should be mostly clipped");
    let image = GrayImage {
        width: 600,
        height: 400,
        pixels,
        epoch: None,
    };
    for opts in [
        ExtractOptions::default(),
        ExtractOptions {
            lsb: 1.0 / 255.0,
            ..Default::default()
        },
    ] {
        let field = extract(&image, &opts);
        let found = truth
            .iter()
            .filter(|&&(x, y)| nearest(&field, x, y).is_some_and(|n| n.1 < 0.7))
            .count();
        assert!(found >= 28, "found {found}/30");
        assert!(
            field.stars.len() <= 36,
            "{} detections on a posterized sky",
            field.stars.len()
        );
    }
}

#[test]
fn sky_clipped_at_zero() {
    // 16-bit frame with the pedestal subtracted and negatives written as 0:
    // just under half the sky sits on the floor, so the median is still
    // above it and the plain MAD reads the noise ~2.5× too low (hundreds of
    // noise peaks at 5σ). Native resolution (no binning) as below 6.5 MP.
    let mut s = Synth::new(1024, 768, 0.6, 2.2, 23);
    let truth = s.positions(30, 20.0);
    for &(x, y) in &truth {
        s.star(x, y, 4000.0, 2.2);
    }
    let pixels: Vec<f32> = s.px.iter().map(|v| v.round().max(0.0)).collect();
    let zeros = pixels.iter().filter(|&&v| v == 0.0).count();
    assert!(
        zeros * 4 > pixels.len() && zeros * 2 < pixels.len(),
        "{zeros} zeros"
    );
    let image = GrayImage {
        width: 1024,
        height: 768,
        pixels,
        epoch: None,
    };
    let field = extract(&image, &ExtractOptions::default());
    let found = truth
        .iter()
        .filter(|&&(x, y)| nearest(&field, x, y).is_some_and(|n| n.1 < 0.5))
        .count();
    assert!(found >= 29, "found {found}/30");
    assert!(
        field.stars.len() <= 33,
        "{} detections on a clipped sky",
        field.stars.len()
    );
}

#[test]
fn zero_padding_is_not_a_clipped_sky() {
    // A registered frame: a zero band on the left (a third of one tile column)
    // is a gap below the sky, not a censored noise tail.
    let mut s = Synth::new(512, 512, 400.0, 8.0, 29);
    let truth: Vec<(f64, f64)> = s
        .positions(25, 30.0)
        .into_iter()
        .map(|(x, y)| (120.0 + (x - 30.0) * 0.72, y))
        .collect();
    for &(x, y) in &truth {
        s.star(x, y, flux_for_snr(60.0, 8.0, 3.0), 3.0);
    }
    for row in s.px.chunks_mut(512) {
        row[..84].fill(0.0);
    }
    let field = extract(&s.image(), &ExtractOptions::default());
    for &(x, y) in &truth {
        assert!(
            nearest(&field, x, y).unwrap().1 < 0.3,
            "star at ({x:.0},{y:.0})"
        );
    }
    assert!(field.stars.len() <= 27, "{} detections", field.stars.len());
}

#[test]
fn highpass_ranks_clipped_stars_by_size() {
    // An 8-bit export of the clipped sky above with an extreme stretch: one level per ADU
    // up to 7 ADU, everything brighter at 255. Every star is a flat disc; only the DoG
    // fallback finds them, and it must still rank them by brightness (the solver's
    // dominance levels depend on that order).
    let mut s = Synth::new(1024, 768, 0.6, 2.2, 31);
    let truth = s.positions(30, 24.0);
    let fluxes: Vec<f64> = (0..30).map(|i| 600.0 * 1.15f64.powi(i)).collect();
    for (&(x, y), &f) in truth.iter().zip(&fluxes) {
        s.star(x, y, f, 2.1);
    }
    const LUT: [f32; 9] = [0.0, 113.0, 160.0, 188.0, 208.0, 223.0, 236.0, 246.0, 255.0];
    let pixels: Vec<f32> =
        s.px.iter()
            .map(|v| LUT[v.round().clamp(0.0, 8.0) as usize])
            .collect();
    let image = GrayImage {
        width: 1024,
        height: 768,
        pixels,
        epoch: None,
    };
    let field = skyshot::highpass::detect_highpass(&image, 2.5, 5.0);
    let found: Vec<(f64, f64)> = truth
        .iter()
        .zip(&fluxes)
        .filter_map(|(&(x, y), &f)| {
            nearest(&field, x, y)
                .filter(|n| n.1 < 1.0)
                .map(|n| (f, n.0.flux))
        })
        .collect();
    assert!(found.len() >= 27, "found {}/30", found.len());
    // Kendall's tau between the true and the measured brightness
    let (mut concordant, mut pairs) = (0, 0);
    for (i, a) in found.iter().enumerate() {
        for b in &found[i + 1..] {
            pairs += 1;
            concordant += ((a.0 - b.0) * (a.1 - b.1) > 0.0) as i32;
        }
    }
    let tau = 2.0 * concordant as f64 / pairs as f64 - 1.0;
    assert!(tau > 0.6, "rank agreement tau {tau:.2}");
}

#[test]
fn detection_is_scale_free() {
    // Linear float frames normalized to [0, 1] with a dark sky (every value
    // below 0.002) and frames in physical units (1e-18 erg/s/cm²/Å and below)
    // must neither be taken for an integer lattice nor hit an absolute noise floor.
    let mut s = Synth::new(500, 400, 100.0, 5.0, 37);
    let truth = s.positions(30, 20.0);
    for &(x, y) in &truth {
        s.star(x, y, flux_for_snr(60.0, 5.0, 3.0), 3.0);
    }
    let reference = extract(&s.image(), &ExtractOptions::default());
    assert_eq!(reference.stars.len(), truth.len());
    for scale in [1e-6f32, 1e-12, 1e-20] {
        let image = GrayImage {
            pixels: s.px.iter().map(|v| v * scale).collect(),
            ..s.image()
        };
        assert!(image.pixels.iter().all(|&v| v < 2e-3));
        let field = extract(&image, &ExtractOptions::default());
        assert_eq!(field.stars.len(), truth.len(), "scale {scale:e}");
        for st in &reference.stars {
            let d = nearest(&field, st.x, st.y).unwrap().1;
            assert!(d < 1e-3, "scale {scale:e}: centroid moved by {d}");
        }
    }
    // An 8-bit frame whose sampled sky is a single level (crushed black).
    let pixels: Vec<f32> = (0..300 * 200)
        .map(|i| {
            let (x, y) = ((i % 300) as f64 - 150.3, (i / 300) as f64 - 100.6);
            (200.0 * (-(x * x + y * y) / 3.2).exp()).round() as f32 / 255.0
        })
        .collect();
    let image = GrayImage {
        width: 300,
        height: 200,
        pixels,
        epoch: None,
    };
    let field = extract(&image, &ExtractOptions::default());
    assert_eq!(field.stars.len(), 1, "{:?}", field.stars);
}

#[test]
fn compact_mode_separates_close_pairs() {
    let mut s = Synth::new(128, 128, 10.0, 0.5, 21);
    s.star(60.3, 62.2, 400.0, 2.1);
    s.star(64.8, 62.2, 240.0, 2.1);
    s.star(30.4, 90.6, 300.0, 2.1);
    s.star(33.1, 93.4, 300.0, 2.1);
    let compact = extract(
        &s.image(),
        &ExtractOptions {
            compact: true,
            ..Default::default()
        },
    );
    for (x, y) in [(60.3, 62.2), (64.8, 62.2), (30.4, 90.6), (33.1, 93.4)] {
        let (star, d) = nearest(&compact, x, y).unwrap();
        assert!(d < 0.2, "({x},{y}) error {d:.3} in {:?}", compact.stars);
        // The second pair is closer than 2 FWHM: both components are flagged.
        if x < 40.0 {
            assert!(star.flags & star_flags::BLENDED != 0);
        }
    }
    assert_eq!(compact.stars.len(), 4);
}

#[test]
fn satellite_streak_is_removed() {
    let mut s = Synth::new(800, 600, 500.0, 10.0, 23);
    let truth = s.positions(30, 30.0);
    for &(x, y) in &truth {
        s.star(x, y, flux_for_snr(60.0, 10.0, 3.0), 3.0);
    }
    // Bright streak with brightness modulation (tumbling satellite), and a
    // dotted line of airplane strobes.
    let (a, b) = ((40.0f64, 520.0f64), (760.0f64, 90.0f64));
    let len = (b.0 - a.0).hypot(b.1 - a.1);
    for k in 0..(len as usize * 2) {
        let t = k as f64 / (len * 2.0);
        let (x, y) = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
        let amp = 400.0 * (1.0 + 0.6 * (t * 40.0).sin());
        s.star(x, y, amp, 2.2);
    }
    for k in 0..12 {
        s.star(100.0 + 50.0 * k as f64, 40.0 + 3.0 * k as f64, 4000.0, 2.5);
    }
    let field = extract(&s.image(), &ExtractOptions::default());
    let on_streak = |st: &Star| {
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let d = ((st.x - a.0) * dy - (st.y - a.1) * dx).abs() / len;
        d < 3.0
    };
    let on_dots = |st: &Star| ((st.y - 40.0) - (st.x - 100.0) * 0.06).abs() < 3.0 && st.x < 700.0;
    for st in &field.stars {
        let real = truth.iter().any(|t| (t.0 - st.x).hypot(t.1 - st.y) < 2.0);
        if !real {
            assert!(!on_streak(st), "streak detection kept: {st:?}");
            assert!(!on_dots(st), "strobe detection kept: {st:?}");
        }
    }
    let found = truth
        .iter()
        .filter(|&&(x, y)| nearest(&field, x, y).is_some_and(|n| n.1 < 0.5))
        .count();
    // Stars lying on the streak itself may be lost; everything else stays.
    assert!(found >= 27, "found {found}/30");
}

#[test]
fn streak_does_not_distort_frame_psf() {
    // In a sparse field a satellite streak supplies more PSF samples than the
    // stars. It must neither inflate the frame FWHM (apertures, merge radii)
    // nor switch the frame to trailed mode.
    let build = |streak: bool| {
        let mut s = Synth::new(800, 600, 500.0, 10.0, 23);
        let truth = s.positions(30, 30.0);
        for &(x, y) in &truth {
            s.star(x, y, flux_for_snr(60.0, 10.0, 3.0), 3.0);
        }
        if streak {
            let (a, b) = ((40.0f64, 520.0f64), (760.0f64, 90.0f64));
            let len = (b.0 - a.0).hypot(b.1 - a.1);
            for k in 0..(len as usize * 2) {
                let t = k as f64 / (len * 2.0);
                let amp = 400.0 * (1.0 + 0.6 * (t * 40.0).sin());
                s.star(a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t, amp, 2.2);
            }
        }
        (truth, s.image())
    };
    let opts = ExtractOptions {
        keep_rejected: true,
        ..Default::default()
    };
    let (truth, clean) = build(false);
    let (_, streaked) = build(true);
    let reference = extract(&clean, &opts);
    let field = extract(&streaked, &opts);
    assert!(
        field
            .stars
            .iter()
            .all(|s| s.flags & star_flags::TRAILED == 0),
        "frame taken for trailed"
    );
    let mut compared = 0;
    for &(x, y) in &truth {
        // Stars next to the streak share its background tiles.
        let off_streak = ((x - 40.0) * -430.0 - (y - 520.0) * 720.0).abs() / 838.6;
        if off_streak < 40.0 {
            continue;
        }
        let (a, da) = nearest(&reference, x, y).unwrap();
        let (b, db) = nearest(&field, x, y).unwrap();
        if da < 0.5 && db < 0.5 && b.flags & star_flags::REJECT == 0 {
            compared += 1;
            assert!(
                (b.flux / a.flux - 1.0).abs() < 0.05 && (b.snr / a.snr - 1.0).abs() < 0.1,
                "star at ({x:.0},{y:.0}): flux {:.0} vs {:.0}, snr {:.1} vs {:.1}",
                b.flux,
                a.flux,
                b.snr,
                a.snr
            );
        }
    }
    assert!(compared >= 20, "{compared} stars compared");
}

#[test]
fn trailed_frame_is_kept_and_flagged() {
    let mut s = Synth::new(700, 500, 300.0, 6.0, 29);
    let truth = s.positions(40, 30.0);
    let angle = 30f64.to_radians();
    for &(x, y) in &truth {
        s.trailed(x, y, 4.0e4, 3.0, 14.0, angle);
    }
    let field = extract(&s.image(), &ExtractOptions::default());
    let mut found = 0;
    for &(x, y) in &truth {
        if let Some((star, d)) = nearest(&field, x, y)
            && d < 1.0
        {
            found += 1;
            assert!(star.flags & star_flags::TRAILED != 0, "{star:?}");
            assert!(star.elongation > 2.0);
            let diff = (star.pa - 30.0).abs();
            assert!(diff < 5.0, "pa {}", star.pa);
            assert!(star.sigma >= 1.0);
        }
    }
    assert!(found >= 38, "found {found}/40");
    assert!(field.stars.len() <= 42, "{} detections", field.stars.len());
}

#[test]
fn nebular_ridges_are_not_stars() {
    let mut s = Synth::new(600, 400, 100.0, 4.0, 31);
    // Curved filament ~8 px wide with 25σ amplitude, plus knots of noise.
    for y in 0..400 {
        for x in 0..600 {
            let cy = 200.0 + 60.0 * ((x as f32) / 90.0).sin();
            let d = y as f32 - cy;
            s.px[y * 600 + x] += 100.0 * (-(d * d) / (2.0 * 4.0 * 4.0)).exp();
        }
    }
    let truth = [
        (150.4, 80.2),
        (420.7, 330.6),
        (300.3, 200.0 + 60.0 * (300.0f64 / 90.0).sin()),
    ];
    for &(x, y) in &truth {
        s.star(x, y, 6000.0, 3.0);
    }
    let field = extract(&s.image(), &ExtractOptions::default());
    for &(x, y) in &truth {
        assert!(
            nearest(&field, x, y).unwrap().1 < 0.5,
            "star at ({x},{y}) lost"
        );
    }
    assert!(
        field.stars.len() <= truth.len() + 1,
        "ridge detections: {:?}",
        field.stars
    );
}

#[test]
fn star_json_is_backward_compatible() {
    let old = r#"{"x":1.5,"y":2.5,"flux":10.0,"sigma":0.2,"snr":30.0,"fwhm":3.0,"elongation":1.1,"saturated":false}"#;
    let star: Star = serde_json::from_str(old).unwrap();
    assert_eq!((star.peak, star.pa, star.flags), (0.0, 0.0, 0));
    let field = StarField {
        width: 10,
        height: 10,
        stars: vec![star],
        epoch: None,
    };
    let back: StarField = serde_json::from_str(&serde_json::to_string(&field).unwrap()).unwrap();
    assert_eq!(back.stars.len(), 1);
}
