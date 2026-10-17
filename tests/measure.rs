//! Synthetic tests for forced photometry (`skyshot::measure`, DESIGN.md §7): known Gaussian
//! sources on a gradient, the frame FWHM, honest and calibrated noise (white and correlated),
//! statuses by rule R, saturation, edges and masks, blends, the zero point and the photon
//! transfer gain, the request path (`run`) and its errors. Client modules only: no
//! `required-features`.
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use skyshot::{
    attempt,
    image_io::{GrayImage, SourceFormat, SourceInfo},
    measure::{
        self, Content, FrameModel, FrameOptions, Measurement, Point, RefStar, Request, Session,
        Status, Target, flags,
    },
    types::Star,
};

const FWHM: f64 = 3.0;

/// Synthetic frame: a background plane plus Gaussian noise (optionally with variance growing
/// as level / gain, optionally box-smoothed) and pixel-integrated Gaussian stars.
struct Synth {
    w: usize,
    h: usize,
    px: Vec<f64>,
    rng: ChaCha8Rng,
}

impl Synth {
    fn new(w: usize, h: usize, seed: u64) -> Self {
        Self {
            w,
            h,
            px: vec![0.0; w * h],
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    fn gauss(&mut self) -> f64 {
        let (u1, u2): (f64, f64) = (self.rng.r#gen::<f64>().max(1e-12), self.rng.r#gen());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Background c + gx·x + gy·y.
    fn background(&mut self, c: f64, gx: f64, gy: f64) {
        for y in 0..self.h {
            for x in 0..self.w {
                self.px[y * self.w + x] += c + gx * x as f64 + gy * y as f64;
            }
        }
    }

    /// White Gaussian noise of `sigma`, plus a Poisson-like part level / gain when given.
    fn noise(&mut self, sigma: f64, gain: Option<f64>) {
        for i in 0..self.px.len() {
            let var = sigma * sigma + gain.map_or(0.0, |g| self.px[i].max(0.0) / g);
            self.px[i] += var.sqrt() * self.gauss();
        }
    }

    /// Box-smooth the frame (k x k): pixel-correlated noise, as stacked or resampled data.
    fn smooth(&mut self, k: usize) {
        let (w, h, r) = (self.w, self.h, k as isize / 2);
        let src = self.px.clone();
        for y in 0..h as isize {
            for x in 0..w as isize {
                let (mut s, mut n) = (0.0, 0.0);
                for dy in -r..=r {
                    for dx in -r..=r {
                        let (xx, yy) = (x + dx, y + dy);
                        if xx >= 0 && yy >= 0 && xx < w as isize && yy < h as isize {
                            s += src[yy as usize * w + xx as usize];
                            n += 1.0;
                        }
                    }
                }
                self.px[y as usize * w + x as usize] = s / n;
            }
        }
    }

    /// A Gaussian star of total `flux`, integrated over pixels (5 x 5 sub-samples).
    fn star(&mut self, x: f64, y: f64, flux: f64, fwhm: f64) {
        let s = fwhm / 2.3548;
        let r = (6.0 * s).ceil() as isize;
        let norm = flux / (2.0 * std::f64::consts::PI * s * s) / 25.0;
        for yy in (y as isize - r).max(0)..=(y as isize + r).min(self.h as isize - 1) {
            for xx in (x as isize - r).max(0)..=(x as isize + r).min(self.w as isize - 1) {
                let mut v = 0.0;
                for sy in 0..5 {
                    for sx in 0..5 {
                        let dx = xx as f64 - 0.4 + 0.2 * sx as f64 - x;
                        let dy = yy as f64 - 0.4 + 0.2 * sy as f64 - y;
                        v += (-(dx * dx + dy * dy) / (2.0 * s * s)).exp();
                    }
                }
                self.px[yy as usize * self.w + xx as usize] += v * norm;
            }
        }
    }

    fn image(&self, clip: Option<f64>) -> GrayImage {
        GrayImage {
            width: self.w as u32,
            height: self.h as u32,
            pixels: self
                .px
                .iter()
                .map(|&v| clip.map_or(v, |c| v.min(c)) as f32)
                .collect(),
            epoch: None,
        }
    }
}

fn fits() -> SourceInfo {
    SourceInfo {
        format: SourceFormat::Fits,
        bitpix: -32,
        lsb: 0.0,
        hint: None,
    }
}

fn detect(img: &GrayImage, info: &SourceInfo) -> Vec<Star> {
    attempt::attempt_field(img, info, "default")
        .expect("default attempt")
        .stars
}

/// Total flux of a FWHM-3 star giving a plain-aperture SNR ≈ `snr` on white noise `sigma`
/// (aperture r = 1 FWHM holds 94 % of a Gaussian; annulus 2.5–4 FWHM).
fn flux_for_snr(snr: f64, sigma: f64) -> f64 {
    let n = std::f64::consts::PI * FWHM * FWHM;
    let n_ann = std::f64::consts::PI * (16.0 - 6.25) * FWHM * FWHM;
    snr * sigma * (n + n * n / n_ann).sqrt() / 0.937
}

/// Bright stars (for the PSF fit and the zero point) near the nodes of a 100 px grid, jittered
/// so that no three are collinear (the detector removes collinear chains as trails); returns
/// their positions and fluxes.
fn bright_grid(s: &mut Synth, flux: f64) -> Vec<(f64, f64, f64)> {
    let mut out = vec![];
    let mut k = 0;
    for gy in 0..5 {
        for gx in 0..6 {
            let (jx, jy): (f64, f64) = (s.rng.gen_range(-12.0..12.0), s.rng.gen_range(-12.0..12.0));
            let (x, y) = (40.0 + 100.0 * gx as f64 + jx, 45.0 + 100.0 * gy as f64 + jy);
            let f = flux * (1.0 + 0.1 * (k % 7) as f64);
            s.star(x, y, f, FWHM);
            out.push((x, y, f));
            k += 1;
        }
    }
    out
}

fn target(x: f64, y: f64) -> Target {
    Target { x, y, mag: None }
}

fn model(img: &GrayImage, info: &SourceInfo, refs: &[RefStar], opts: &FrameOptions) -> FrameModel {
    let stars = detect(img, info);
    FrameModel::new(img, info, &stars, refs, None, opts).expect("model")
}

#[test]
fn known_sources_on_a_gradient() {
    let mut s = Synth::new(620, 520, 1);
    s.background(1000.0, 0.3, -0.2);
    let bright = bright_grid(&mut s, 60_000.0);
    // recorded (SNR ~ 25, offset by a known sub-pixel amount), trace (SNR ~ 4.5: within the
    // trace band 3-6 at ±1.5σ), empty; all between the nodes of the bright grid
    let rec = [(90.3, 95.6), (290.7, 195.2), (490.1, 295.8), (190.5, 395.4)];
    let empty = [
        (390.0, 395.0),
        (490.0, 95.0),
        (290.0, 495.0),
        (590.0, 195.0),
    ];
    let free = |p: &(f64, f64)| {
        rec.iter()
            .chain(&empty)
            .all(|q| (p.0 - q.0).hypot(p.1 - q.1) > 5.0)
    };
    let tr: Vec<(f64, f64)> = (0..30)
        .map(|k| (90.4 + 100.0 * (k % 6) as f64, 95.2 + 100.0 * (k / 6) as f64))
        .filter(free)
        .take(12)
        .collect();
    let f_rec = flux_for_snr(25.0, 10.0);
    for &(x, y) in &rec {
        s.star(x + 0.4, y - 0.3, f_rec, FWHM);
    }
    let f_tr = flux_for_snr(4.5, 10.0);
    for &(x, y) in &tr {
        s.star(x, y, f_tr, FWHM);
    }
    s.noise(10.0, None);
    let img = s.image(None);
    let info = fits();
    let m = model(&img, &info, &[], &FrameOptions::default());
    let sum = m.summary();
    assert!(
        (sum.fwhm - 3.08).abs() < 0.15,
        "fitted FWHM {} (pixel-integrated FWHM 3 is 3.08)",
        sum.fwhm
    );
    assert!(
        (sum.sigma_sky - 10.0).abs() < 1.0,
        "sky noise {}",
        sum.sigma_sky
    );
    // white noise: the formula sigma is honest, chi close to 1
    let chi = sum.chi.all.expect("calibrated");
    assert!((0.85..1.2).contains(&chi), "chi {chi} on white noise");
    assert!(
        sum.linear && sum.gain.is_none(),
        "no photon-transfer slope in pure read noise"
    );

    let mut t: Vec<Target> = rec.iter().map(|&(x, y)| target(x, y)).collect();
    t.extend(tr.iter().map(|&(x, y)| target(x, y)));
    t.extend(empty.iter().map(|&(x, y)| target(x, y)));
    let out = measure::measure(&img, &m, &t);
    for (i, r) in out[..4].iter().enumerate() {
        assert_eq!(r.status, Status::Recorded, "source {i}: {r:?}");
        // the aperture holds ~94 % of the flux
        let frac = r.flux.unwrap() / f_rec;
        assert!((0.85..1.0).contains(&frac), "flux fraction {frac}");
        let (dx, dy) = (r.dx.unwrap(), r.dy.unwrap());
        assert!(
            (dx - 0.4).abs() < 0.12 && (dy + 0.3).abs() < 0.12,
            "centroid offset ({dx:.3}, {dy:.3}) vs (0.4, -0.3)"
        );
        assert!(r.snr_dog.unwrap() > 3.0);
        assert_eq!(r.flags & flags::NOT_ASSESSABLE, 0);
    }
    // SNR 4.5 ± 1: most in the trace band, none recorded
    let faint = &out[4..4 + tr.len()];
    let traces = faint.iter().filter(|r| r.status == Status::Trace).count();
    assert!(
        traces >= 8,
        "{traces} of {} faint sources are traces",
        faint.len()
    );
    assert!(faint.iter().all(|r| r.status != Status::Recorded));
    let mut snr: Vec<f64> = faint.iter().map(|r| r.snr_cal.unwrap()).collect();
    snr.sort_by(f64::total_cmp);
    assert!(
        (3.5..5.5).contains(&snr[snr.len() / 2]),
        "median SNR {snr:?}"
    );
    for r in &out[4 + tr.len()..] {
        assert_eq!(r.status, Status::Below, "{r:?}");
        assert!(r.snr_cal.unwrap().abs() < 3.0);
    }
    // the bright stars are known too: offsets near 0
    let tb: Vec<Target> = bright.iter().map(|&(x, y, _)| target(x, y)).collect();
    for r in measure::measure(&img, &m, &tb) {
        let off = r.dx.unwrap().hypot(r.dy.unwrap());
        assert!(off < 0.05, "bright star centroid off by {off}");
    }
}

#[test]
fn correlated_noise_is_calibrated() {
    // 3 x 3 box smoothing: the per-pixel sigma understates aperture noise by ~2.5x; the
    // formula SNR at empty positions spreads far wider than 1 and its 'recorded' rate is
    // percent level, the calibrated one is not
    let mut s = Synth::new(640, 560, 2);
    s.background(500.0, 0.0, 0.0);
    let _ = bright_grid(&mut s, 400_000.0);
    s.noise(30.0, None);
    s.smooth(3);
    let img = s.image(None);
    let info = fits();
    let m = model(&img, &info, &[], &FrameOptions::default());
    let chi = m.summary().chi.sky.expect("sky calibrated");
    assert!(chi > 2.0, "raw chi {chi} should show the correlation");
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let empty: Vec<Target> = (0..3000)
        .map(|_| target(rng.gen_range(20.0..620.0), rng.gen_range(20.0..540.0)))
        .filter(|t| {
            // away from the (jittered) bright grid
            let (dx, dy) = (
                (t.x - 40.0).rem_euclid(100.0),
                (t.y - 45.0).rem_euclid(100.0),
            );
            dx.min(100.0 - dx).hypot(dy.min(100.0 - dy)) > 35.0
        })
        .collect();
    let out = measure::measure(&img, &m, &empty);
    let ok: Vec<&Measurement> = out
        .iter()
        .filter(|r| r.status != Status::NotAssessable)
        .collect();
    assert!(ok.len() > 1000);
    let raw_hits = ok.iter().filter(|r| r.snr.unwrap() >= 6.0).count();
    let cal_hits = ok.iter().filter(|r| r.status == Status::Recorded).count();
    let mut cal: Vec<f64> = ok.iter().map(|r| r.snr_cal.unwrap()).collect();
    cal.sort_by(f64::total_cmp);
    let med = cal[cal.len() / 2];
    let mut dev: Vec<f64> = cal.iter().map(|v| (v - med).abs()).collect();
    dev.sort_by(f64::total_cmp);
    let sd = 1.4826 * dev[dev.len() / 2];
    assert!((0.85..1.2).contains(&sd), "calibrated SNR spread {sd}");
    assert!(
        raw_hits as f64 > 0.005 * ok.len() as f64,
        "raw SNR >= 6 at {raw_hits} of {} empty positions",
        ok.len()
    );
    assert!(
        cal_hits as f64 <= 0.002 * ok.len() as f64,
        "calibrated recorded at {cal_hits} of {} empty positions",
        ok.len()
    );
}

#[test]
fn saturation_edges_and_masks() {
    let mut s = Synth::new(600, 500, 3);
    s.background(800.0, 0.1, 0.1);
    let bright = bright_grid(&mut s, 40_000.0);
    s.star(300.0, 250.0, 3_000_000.0, FWHM); // clipped below
    s.noise(8.0, None);
    let mut img = s.image(Some(20_000.0));
    // a masked block (no data), as a transparent PNG area or a content mask
    for y in 330..380 {
        for x in 120..170 {
            img.pixels[y * 600 + x] = f32::NAN;
        }
    }
    let info = fits();
    let m = model(&img, &info, &[], &FrameOptions::default());
    let t = [
        target(300.0, 250.0), // saturated core
        target(1.0, 250.0),   // aperture off the frame
        target(-40.0, -40.0), // outside
        target(145.0, 355.0), // masked centre
        target(145.0, 384.5), // aperture in data, annulus half masked: still measurable or edge
        target(bright[7].0, bright[7].1),
    ];
    let out = measure::measure(&img, &m, &t);
    assert_eq!(out[0].status, Status::NotAssessable);
    assert_ne!(out[0].flags & flags::SATURATED, 0, "{:?}", out[0]);
    for r in &out[1..3] {
        assert_eq!(r.status, Status::NotAssessable);
        assert_ne!(r.flags & flags::NEAR_EDGE, 0);
        assert!(r.flux.is_none() && r.snr.is_none());
    }
    assert_eq!(out[3].status, Status::NotAssessable);
    assert_ne!(out[3].flags & flags::NO_DATA, 0, "{:?}", out[3]);
    assert_ne!(out[4].status, Status::Recorded);
    assert_eq!(out[5].status, Status::Recorded, "{:?}", out[5]);
}

#[test]
fn blends_zero_point_and_depth() {
    let mut s = Synth::new(620, 520, 4);
    s.background(1200.0, -0.2, 0.25);
    let bright = bright_grid(&mut s, 50_000.0);
    // a faint cluster 3 px (1 FWHM) east of a bright star
    let (bx, by) = (bright[8].0 + 3.0, bright[8].1);
    s.star(bx, by, flux_for_snr(5.0, 9.0), FWHM);
    s.noise(9.0, None);
    let img = s.image(None);
    let info = fits();
    // reference stars: catalogue magnitudes of the bright stars (zp 25 on total flux), plus
    // faint field stars that only serve as neighbours
    let mut refs: Vec<RefStar> = bright
        .iter()
        .map(|&(x, y, f)| RefStar {
            x,
            y,
            mag: 25.0 - 2.5 * f.log10(),
        })
        .collect();
    refs.push(RefStar {
        x: 500.0,
        y: 500.0,
        mag: 21.0,
    });
    let m = model(&img, &info, &refs, &FrameOptions::default());
    let sum = m.summary();
    let z = sum.zero_point.clone().expect("zero point");
    // the aperture holds ~94 % of the flux: zp ≈ 25 + 2.5 log10(0.937)
    assert!((z.zp - 24.93).abs() < 0.05, "zp {}", z.zp);
    assert!(z.scatter < 0.05 && z.linear && sum.photometric, "{z:?}");
    let m_lim = sum.m_lim.expect("depth");
    // SNR 5: ~5 * 9 * sqrt(28 + 3) ≈ 250 units → 25 - 2.5 log10(250 / 0.937) ≈ 18.9
    assert!((m_lim - 18.9).abs() < 0.4, "m_lim {m_lim}");
    // the cluster next to the bright star is blended; one far from everything is not
    let out = measure::measure(
        &img,
        &m,
        &[
            Target {
                x: bx,
                y: by,
                mag: Some(19.0),
            },
            Target {
                x: 90.0,
                y: 95.0,
                mag: Some(19.0),
            },
        ],
    );
    assert_eq!(out[0].status, Status::NotAssessable);
    assert_ne!(out[0].flags & flags::BLENDED, 0);
    assert_eq!(out[1].flags & flags::BLENDED, 0);
    // without reference stars the detections decide: the bright star is detected and brighter
    let m2 = model(&img, &info, &[], &FrameOptions::default());
    let out2 = measure::measure(&img, &m2, &[target(bx, by)]);
    assert_ne!(out2[0].flags & flags::BLENDED, 0, "{:?}", out2[0]);
}

#[test]
fn photon_transfer_gain_of_a_linear_source() {
    let mut s = Synth::new(640, 640, 5);
    // a steep background so that tiles span a range of levels; variance = 4² + level / 2
    s.background(500.0, 4.0, 3.0);
    let _ = bright_grid(&mut s, 80_000.0);
    s.noise(4.0, Some(2.0));
    let img = s.image(None);
    let m = model(&img, &fits(), &[], &FrameOptions::default());
    let g = m.summary().gain.expect("gain of a linear source");
    assert!((g - 2.0).abs() < 0.3, "gain {g}");
    // the same pixels as a gamma-encoded 16-bit image: no gain, flagged non-linear
    let png = SourceInfo {
        format: SourceFormat::Png,
        bitpix: 16,
        lsb: 0.0,
        hint: None,
    };
    let m2 = model(&img, &png, &[], &FrameOptions::default());
    assert!(m2.summary().gain.is_none() && !m2.photometric());
    let r = &measure::measure(&img, &m2, &[target(40.0, 45.0)])[0];
    assert_ne!(r.flags & flags::NONLINEAR, 0);
    // explicitly linear: gain again
    let lin = FrameOptions {
        linear: Some(true),
        ..FrameOptions::default()
    };
    assert!(model(&img, &png, &[], &lin).summary().gain.is_some());
}

fn request_frame() -> (GrayImage, SourceInfo) {
    let mut s = Synth::new(400, 360, 6);
    s.background(300.0, 0.0, 0.0);
    for k in 0..12 {
        // jittered: no collinear chains (the detector removes those as trails)
        let (jx, jy): (f64, f64) = (s.rng.gen_range(-10.0..10.0), s.rng.gen_range(-10.0..10.0));
        s.star(
            40.0 + 80.0 * (k % 5) as f64 + jx,
            40.0 + 90.0 * (k / 5) as f64 + jy,
            30_000.0,
            FWHM,
        );
    }
    s.star(200.0, 300.0, flux_for_snr(20.0, 5.0), FWHM);
    s.noise(5.0, None);
    (s.image(None), fits())
}

fn px(x: f64, y: f64) -> Point {
    Point {
        x: Some(x),
        y: Some(y),
        ..Point::default()
    }
}

#[test]
fn request_path_is_deterministic_and_cached() {
    let (img, info) = request_frame();
    let req = Request {
        targets: vec![px(200.0, 300.0), px(100.0, 300.0), px(-5.0, 3.0)],
        ..Request::default()
    };
    let detections = |g: &GrayImage| detect(g, &info);
    let mut a = img.pixels.clone();
    let mut img1 = GrayImage {
        pixels: std::mem::take(&mut a),
        ..img
    };
    let mut session = Session::default();
    let r1 = measure::run(&mut img1, &info, &req, &mut session, detections).unwrap();
    // cached model: the detector must not run again
    let r2 = measure::run(&mut img1, &info, &req, &mut session, |_| {
        panic!("the model was rebuilt")
    })
    .unwrap();
    assert_eq!(r1, r2);
    // a fresh session gives the same bytes
    let mut s2 = Session::default();
    let r3 = measure::run(&mut img1, &info, &req, &mut s2, detections).unwrap();
    assert_eq!(
        serde_json::to_string(&r1).unwrap(),
        serde_json::to_string(&r3).unwrap()
    );
    assert_eq!(r1.rows.len(), 3);
    assert_eq!(r1.rows[0].12, Status::Recorded);
    assert_eq!(r1.rows[1].12, Status::Below);
    assert_eq!(r1.rows[2].12, Status::NotAssessable);
    assert_eq!(
        r1.counts.recorded + r1.counts.below + r1.counts.not_assessable,
        3
    );
    // a given FWHM rebuilds the model and is used as given
    let given = Request {
        frame: FrameOptions {
            fwhm: Some(3.2),
            ..FrameOptions::default()
        },
        ..req.clone()
    };
    let r4 = measure::run(&mut img1, &info, &given, &mut session, detections).unwrap();
    assert_eq!(r4.model.fwhm, 3.2);
}

#[test]
fn request_errors_never_panic() {
    let (img, info) = request_frame();
    let detections = |g: &GrayImage| detect(g, &info);
    let run = |req: &Request, session: &mut Session| {
        let mut im = GrayImage {
            width: img.width,
            height: img.height,
            pixels: img.pixels.clone(),
            epoch: None,
        };
        measure::run(&mut im, &info, req, session, detections)
    };
    let code = |req: Request| run(&req, &mut Session::default()).unwrap_err().code;
    let sky = Point {
        ra: Some(10.0),
        dec: Some(41.0),
        ..Point::default()
    };
    assert_eq!(
        code(Request {
            targets: vec![sky.clone()],
            ..Request::default()
        }),
        "bad_request"
    );
    assert_eq!(
        code(Request {
            refs: vec![px(10.0, 10.0)],
            ..Request::default()
        }),
        "bad_request"
    );
    assert_eq!(
        code(Request {
            targets: vec![Point {
                x: Some(1.0),
                ..Point::default()
            }],
            ..Request::default()
        }),
        "bad_request"
    );
    assert_eq!(
        code(Request {
            frame: FrameOptions {
                fwhm: Some(500.0),
                ..FrameOptions::default()
            },
            ..Request::default()
        }),
        "bad_request"
    );
    // content applied once; a different mask later conflicts
    let mut session = Session::default();
    let a = Request {
        targets: vec![px(200.0, 300.0)],
        frame: FrameOptions {
            content: Some(Content {
                include: vec![[0.0, 0.0, 400.0, 360.0]],
                exclude: vec![[0.0, 0.0, 50.0, 50.0]],
            }),
            ..FrameOptions::default()
        },
        ..Request::default()
    };
    let mut im = GrayImage {
        width: img.width,
        height: img.height,
        pixels: img.pixels.clone(),
        epoch: None,
    };
    measure::run(&mut im, &info, &a, &mut session, detections).unwrap();
    assert!(im.pixels[10 * 400 + 10].is_nan() && im.pixels[100 * 400 + 100].is_finite());
    let mut b = a.clone();
    b.frame.content.as_mut().unwrap().exclude.clear();
    assert_eq!(
        measure::run(&mut im, &info, &b, &mut session, detections)
            .unwrap_err()
            .code,
        "content_conflict"
    );
    // a frame without stars has no FWHM unless one is given
    let mut flat = Synth::new(200, 200, 7);
    flat.background(100.0, 0.0, 0.0);
    flat.noise(3.0, None);
    let mut fi = flat.image(None);
    let req = Request {
        targets: vec![px(100.0, 100.0)],
        ..Request::default()
    };
    let none = |_: &GrayImage| Vec::<Star>::new();
    let e = measure::run(&mut fi, &info, &req, &mut Session::default(), none).unwrap_err();
    assert_eq!(e.code, "no_psf");
    let given = Request {
        frame: FrameOptions {
            fwhm: Some(3.0),
            ..FrameOptions::default()
        },
        ..req
    };
    let out = measure::run(&mut fi, &info, &given, &mut Session::default(), none).unwrap();
    assert_eq!(out.rows[0].12, Status::Below);
    // hostile JSON is refused by the parser
    for bad in [
        &b"{"[..],
        b"{\"targets\": 5}",
        b"{\"targets\": [{\"x\": 1, \"y\": 2, \"z\": 3}]}",
        b"{\"frame\": {\"fwhm\": \"wide\"}}",
        b"[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]",
    ] {
        assert!(serde_json::from_slice::<Request>(bad).is_err());
    }
    // a tiny image is refused, not a panic
    let mut tiny = GrayImage {
        width: 8,
        height: 8,
        pixels: vec![1.0; 64],
        epoch: None,
    };
    assert_eq!(
        measure::run(&mut tiny, &info, &given, &mut Session::default(), none)
            .unwrap_err()
            .code,
        "no_frame"
    );
}

#[test]
fn zero_point_helper() {
    let cat: Vec<f64> = (0..60).map(|i| 9.0 + 0.1 * i as f64).collect();
    // a stretched response (slope 0.7) with a few outliers
    let inst: Vec<f64> = cat
        .iter()
        .enumerate()
        .map(|(i, m)| 0.7 * m - 17.0 + if i % 13 == 0 { 1.5 } else { 0.0 })
        .collect();
    let z = measure::fit_zero_point(&cat, &inst).unwrap();
    assert!((z.slope.unwrap() - 0.7).abs() < 1e-6, "{z:?}");
    assert!(!z.linear);
    assert!(z.n < 60 && z.n > 4);
}
