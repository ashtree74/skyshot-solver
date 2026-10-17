use crate::types::{CatalogStar, Proj, Sip, Wcs};
use nalgebra::{DMatrix, DVector, Matrix2, SMatrix, SVector, Vector2};
use std::f64::consts::PI;

pub fn unit(ra: f64, dec: f64) -> [f64; 3] {
    let (r, d) = (ra.to_radians(), dec.to_radians());
    [d.cos() * r.cos(), d.cos() * r.sin(), d.sin()]
}
pub fn separation(a: [f64; 3], b: [f64; 3]) -> f64 {
    let cross = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    (cross.iter().map(|x| x * x).sum::<f64>().sqrt())
        .atan2(a.iter().zip(b).map(|(x, y)| x * y).sum())
}
pub fn project(ra: f64, dec: f64, center: [f64; 2]) -> Option<[f64; 2]> {
    let (r, d, r0, d0) = (
        ra.to_radians(),
        dec.to_radians(),
        center[0].to_radians(),
        center[1].to_radians(),
    );
    let dr = r - r0;
    let den = d0.sin() * d.sin() + d0.cos() * d.cos() * dr.cos();
    if den <= 0.0 {
        return None;
    }
    Some([
        (d.cos() * dr.sin() / den).to_degrees(),
        ((d0.cos() * d.sin() - d0.sin() * d.cos() * dr.cos()) / den).to_degrees(),
    ])
}
pub fn unproject(x: f64, y: f64, center: [f64; 2]) -> [f64; 2] {
    let (x, y, r, d) = (
        x.to_radians(),
        y.to_radians(),
        center[0].to_radians(),
        center[1].to_radians(),
    );
    let den = d.cos() - y * d.sin();
    [
        (r + x.atan2(den)).to_degrees().rem_euclid(360.0),
        ((d.sin() + y * d.cos()).atan2((den * den + x * x).sqrt())).to_degrees(),
    ]
}

/// Unit vector of the reference point and the east and north unit vectors there.
fn zenithal_basis(center: [f64; 2]) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let (sa, ca) = center[0].to_radians().sin_cos();
    let (sd, cd) = center[1].to_radians().sin_cos();
    (
        [cd * ca, cd * sa, sd],
        [-sa, ca, 0.0],
        [-sd * ca, -sd * sa, cd],
    )
}

/// Zenithal equidistant (ARC) projection: intermediate coordinates (degrees) whose length is the
/// angular distance from `center`, in the direction of the TAN coordinates (x east, y north).
/// `None` within 1e-6 rad of the antipode.
pub fn project_arc(ra: f64, dec: f64, center: [f64; 2]) -> Option<[f64; 2]> {
    let v = unit(ra, dec);
    let (c, e, n) = zenithal_basis(center);
    let dot = |a: [f64; 3]| a[0] * v[0] + a[1] * v[1] + a[2] * v[2];
    let (ve, vn, vc) = (dot(e), dot(n), dot(c));
    let s = ve.hypot(vn);
    let rho = s.atan2(vc);
    if rho > PI - 1e-6 {
        return None;
    }
    let k = if s < 1e-12 { 1.0 } else { rho / s };
    Some([(ve * k).to_degrees(), (vn * k).to_degrees()])
}

/// Inverse of [`project_arc`].
pub fn unproject_arc(x: f64, y: f64, center: [f64; 2]) -> [f64; 2] {
    let (xr, yr) = (x.to_radians(), y.to_radians());
    let rho = xr.hypot(yr);
    let (c, e, n) = zenithal_basis(center);
    let k = if rho < 1e-12 { 1.0 } else { rho.sin() / rho };
    let cr = rho.cos();
    let v: Vec<f64> = (0..3)
        .map(|i| c[i] * cr + k * (xr * e[i] + yr * n[i]))
        .collect();
    [
        v[1].atan2(v[0]).to_degrees().rem_euclid(360.0),
        v[2].atan2(v[0].hypot(v[1])).to_degrees(),
    ]
}

/// Projection parameters shared by fitting and pixel/sky transforms. A fixed native pole
/// allows the reference point to move during fitting without changing a cylinder's axis.
#[derive(Clone, Copy, Debug)]
pub struct Projection {
    pub proj: Proj,
    pub native_pole: Option<[f64; 2]>,
}

impl From<Proj> for Projection {
    fn from(proj: Proj) -> Self {
        Self {
            proj,
            native_pole: None,
        }
    }
}

impl Projection {
    pub fn fixed(mut self, center: [f64; 2]) -> Self {
        if self.proj.is_cylindrical() && self.native_pole.is_none() {
            let (_, _, n) = panorama_basis(center);
            self.native_pole = Some(vector_sky(n));
        }
        self
    }
    pub fn prepare(self, center: [f64; 2]) -> PreparedProjection {
        PreparedProjection::with_pole(self.proj, center, self.native_pole)
    }
    pub fn unproject(self, x: f64, y: f64, center: [f64; 2]) -> [f64; 2] {
        self.prepare(center).unproject(x, y)
    }
}

// Shared by native and WebAssembly atlas evaluation; use libm for deterministic
// transcendental functions, as the existing atlas TAN/ARC transforms do.
fn panorama_unit(ra: f64, dec: f64) -> [f64; 3] {
    let (a, d) = (ra.to_radians(), dec.to_radians());
    [
        libm::cos(d) * libm::cos(a),
        libm::cos(d) * libm::sin(a),
        libm::sin(d),
    ]
}
fn panorama_basis(center: [f64; 2]) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let (a, d) = (center[0].to_radians(), center[1].to_radians());
    let (sa, ca, sd, cd) = (libm::sin(a), libm::cos(a), libm::sin(d), libm::cos(d));
    (
        [cd * ca, cd * sa, sd],
        [-sa, ca, 0.],
        [-sd * ca, -sd * sa, cd],
    )
}
fn vector_sky(v: [f64; 3]) -> [f64; 2] {
    [
        libm::atan2(v[1], v[0]).to_degrees().rem_euclid(360.0),
        libm::atan2(v[2], libm::hypot(v[0], v[1])).to_degrees(),
    ]
}

fn cylinder_y(proj: Proj, lat: f64) -> f64 {
    match proj {
        Proj::Car => lat,
        Proj::Mer => libm::asinh(libm::tan(lat)),
        _ => libm::tan(lat),
    }
}

/// Sky -> intermediate coordinates (degrees) for `proj` with the default native frame.
#[inline]
pub fn project_p(proj: Proj, ra: f64, dec: f64, center: [f64; 2]) -> Option<[f64; 2]> {
    PreparedProjection::new(proj, center).project(ra, dec)
}

/// Intermediate coordinates (degrees) -> sky for `proj` with the default native frame.
#[inline]
pub fn unproject_p(proj: Proj, x: f64, y: f64, center: [f64; 2]) -> [f64; 2] {
    PreparedProjection::new(proj, center).unproject(x, y)
}

/// Projection with fixed trigonometry prepared once. TAN and ARC preserve the
/// historical arithmetic order at matching boundaries.
pub struct PreparedProjection {
    proj: Proj,
    center: [f64; 2],
    ra0: f64,
    sin_dec0: f64,
    cos_dec0: f64,
    basis: ([f64; 3], [f64; 3], [f64; 3]),
    latitude0: f64,
}

impl PreparedProjection {
    pub fn new(proj: Proj, center: [f64; 2]) -> Self {
        Self::with_pole(proj, center, None)
    }
    pub fn with_pole(proj: Proj, center: [f64; 2], pole: Option<[f64; 2]>) -> Self {
        let d0 = center[1].to_radians();
        let mut basis = if proj.is_tan() {
            ([0.0; 3], [0.0; 3], [0.0; 3])
        } else if proj == Proj::Arc {
            zenithal_basis(center)
        } else {
            panorama_basis(center)
        };
        let mut latitude0 = 0.0;
        if proj.is_cylindrical()
            && let Some(pole) = pole
        {
            let c = panorama_unit(center[0], center[1]);
            let n = panorama_unit(pole[0], pole[1]);
            let dot = c.iter().zip(n).map(|(a, b)| a * b).sum::<f64>();
            latitude0 = libm::asin(dot.clamp(-1.0, 1.0));
            let a = std::array::from_fn::<_, 3, _>(|i| c[i] - dot * n[i]);
            let norm = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let a = a.map(|x| x / norm);
            let e = [
                n[1] * a[2] - n[2] * a[1],
                n[2] * a[0] - n[0] * a[2],
                n[0] * a[1] - n[1] * a[0],
            ];
            basis = (a, e, n);
        }
        Self {
            proj,
            center,
            ra0: center[0].to_radians(),
            sin_dec0: d0.sin(),
            cos_dec0: d0.cos(),
            basis,
            latitude0,
        }
    }

    #[inline]
    pub fn project(&self, ra: f64, dec: f64) -> Option<[f64; 2]> {
        if self.proj.is_tan() {
            let (r, d) = (ra.to_radians(), dec.to_radians());
            let dr = r - self.ra0;
            let den = self.sin_dec0 * d.sin() + self.cos_dec0 * d.cos() * dr.cos();
            if den <= 0.0 {
                return None;
            }
            return Some([
                (d.cos() * dr.sin() / den).to_degrees(),
                ((self.cos_dec0 * d.sin() - self.sin_dec0 * d.cos() * dr.cos()) / den).to_degrees(),
            ]);
        }
        let v = if self.proj == Proj::Arc {
            unit(ra, dec)
        } else {
            panorama_unit(ra, dec)
        };
        let (c, e, n) = self.basis;
        let dot = |a: [f64; 3]| a[0] * v[0] + a[1] * v[1] + a[2] * v[2];
        let (ve, vn, vc) = (dot(e), dot(n), dot(c));
        let xy = if self.proj.is_cylindrical() {
            // Longitude is undefined at the native poles. Keep one branch at the seam.
            if libm::hypot(ve, vc) < 1e-10 {
                return None;
            }
            let lat = libm::atan2(vn, libm::hypot(ve, vc));
            [
                libm::atan2(ve, vc).to_degrees(),
                (cylinder_y(self.proj, lat) - cylinder_y(self.proj, self.latitude0)).to_degrees(),
            ]
        } else {
            let s = if self.proj == Proj::Arc {
                ve.hypot(vn)
            } else {
                libm::hypot(ve, vn)
            };
            let rho = if self.proj == Proj::Arc {
                s.atan2(vc)
            } else {
                libm::atan2(s, vc)
            };
            if rho > PI - 1e-6 {
                return None;
            }
            let k = if self.proj == Proj::Stg {
                2.0 / (1.0 + vc)
            } else if s < 1e-12 {
                1.0
            } else {
                rho / s
            };
            [(ve * k).to_degrees(), (vn * k).to_degrees()]
        };
        xy.iter().all(|x| x.is_finite()).then_some(xy)
    }

    pub fn unproject(&self, x: f64, y: f64) -> [f64; 2] {
        if self.proj.is_tan() {
            return unproject(x, y, self.center);
        }
        if self.proj == Proj::Arc {
            return unproject_arc(x, y, self.center);
        }
        let (x, y) = (x.to_radians(), y.to_radians());
        let (c, e, n) = self.basis;
        let v = if self.proj.is_cylindrical() {
            let yy = y + cylinder_y(self.proj, self.latitude0);
            let lat = match self.proj {
                Proj::Car => yy,
                Proj::Mer => libm::asin(libm::tanh(yy)),
                _ => libm::atan(yy),
            };
            if x.abs() > PI + 1e-12 || lat.abs() > PI / 2.0 + 1e-12 {
                return [f64::NAN; 2];
            }
            std::array::from_fn(|i| {
                libm::cos(lat) * (libm::cos(x) * c[i] + libm::sin(x) * e[i]) + libm::sin(lat) * n[i]
            })
        } else {
            let r = libm::hypot(x, y);
            let rho = 2.0 * libm::atan(r / 2.0);
            let k = if r < 1e-12 { 1.0 } else { libm::sin(rho) / r };
            std::array::from_fn(|i| libm::cos(rho) * c[i] + k * (x * e[i] + y * n[i]))
        };
        vector_sky(v)
    }
}

/// Batch sky-to-pixel evaluator borrowing an immutable WCS. Its projection and
/// determinant remain valid for this WCS only. The adjugate/divide form of the CD
/// inverse preserves the scalar evaluator's rounding at matching boundaries.
pub struct PreparedWcs<'a> {
    wcs: &'a Wcs,
    projection: PreparedProjection,
    det: f64,
}

impl PreparedWcs<'_> {
    /// Strict inversion with the same arithmetic, iteration cap and stopping test
    /// as `Wcs::sky_to_pixel`; only fixed projection work and SIP powers are shared.
    #[inline]
    pub fn sky_to_pixel(&self, ra: f64, dec: f64) -> Option<[f64; 2]> {
        let p = self.projection.project(ra, dec)?;
        if self.det.abs() < 1e-16 {
            return None;
        }
        let cd = &self.wcs.cd;
        let tx = (cd[1][1] * p[0] - cd[0][1] * p[1]) / self.det;
        let ty = (-cd[1][0] * p[0] + cd[0][0] * p[1]) / self.det;
        let (mut x, mut y) = (tx, ty);
        if let Some(s) = &self.wcs.sip {
            for _ in 0..20 {
                let ((fa, fb), jj) = s.eval_jacobian(x, y);
                let err = Vector2::new(x + fa - tx, y + fb - ty);
                let j = Matrix2::new(jj[0][0], jj[0][1], jj[1][0], jj[1][1]);
                let delta = j.try_inverse()? * err;
                x -= delta[0];
                y -= delta[1];
                if delta.norm() < 1e-7 {
                    break;
                }
            }
        }
        if x.is_finite() && y.is_finite() {
            Some(
                self.wcs
                    .correct_pixel([x + self.wcs.crpix[0], y + self.wcs.crpix[1]]),
            )
        } else {
            None
        }
    }

    #[inline]
    pub fn sky_to_pixel_fast(&self, ra: f64, dec: f64) -> Option<[f64; 2]> {
        let p = self.projection.project(ra, dec)?;
        if self.det.abs() < 1e-16 {
            return None;
        }
        let cd = &self.wcs.cd;
        let tx = (cd[1][1] * p[0] - cd[0][1] * p[1]) / self.det;
        let ty = (-cd[1][0] * p[0] + cd[0][0] * p[1]) / self.det;
        let (mut x, mut y) = (tx, ty);
        if let Some(s) = &self.wcs.sip {
            for _ in 0..8 {
                let ((fa, fb), jj) = s.eval_jacobian(x, y);
                let ex = x + fa - tx;
                let ey = y + fb - ty;
                let d = jj[0][0] * jj[1][1] - jj[0][1] * jj[1][0];
                if d.abs() < 1e-12 {
                    return None;
                }
                let dx = (jj[1][1] * ex - jj[0][1] * ey) / d;
                let dy = (-jj[1][0] * ex + jj[0][0] * ey) / d;
                x -= dx;
                y -= dy;
                if dx.abs() + dy.abs() < 1e-3 {
                    break;
                }
            }
        }
        if x.is_finite() && y.is_finite() {
            Some(
                self.wcs
                    .correct_pixel([x + self.wcs.crpix[0], y + self.wcs.crpix[1]]),
            )
        } else {
            None
        }
    }
}

pub fn at_epoch(mut s: CatalogStar, epoch: Option<f64>) -> CatalogStar {
    if let Some(t) = epoch {
        let dt = t - s.epoch as f64;
        let u = unit(s.ra, s.dec);
        let r = s.ra.to_radians();
        let d = s.dec.to_radians();
        let e = [-r.sin(), r.cos(), 0.0];
        let n = [-r.cos() * d.sin(), -r.sin() * d.sin(), d.cos()];
        let k = PI / (180.0 * 3_600_000.0);
        let v: Vec<f64> = (0..3)
            .map(|i| u[i] + dt * k * (s.pmra as f64 * e[i] + s.pmdec as f64 * n[i]))
            .collect();
        s.ra = v[1].atan2(v[0]).to_degrees().rem_euclid(360.0);
        s.dec = v[2].atan2(v[0].hypot(v[1])).to_degrees();
    }
    s
}
impl Wcs {
    /// Base WCS pixels to measured image pixels, after any local registration.
    pub fn correct_pixel(&self, p: [f64; 2]) -> [f64; 2] {
        self.local_grid.as_ref().map_or(p, |g| g.forward(p))
    }
    pub fn uncorrect_pixel(&self, p: [f64; 2]) -> Option<[f64; 2]> {
        self.local_grid.as_ref().map_or(Some(p), |g| g.inverse(p))
    }
    /// Projection geometry, including the native axis of a panorama.
    pub fn projection(&self) -> Projection {
        Projection {
            proj: self.proj,
            native_pole: self.native_pole,
        }
    }
    /// Prepare the fixed projection quantities for repeated catalogue transforms.
    pub fn prepare(&self) -> PreparedWcs<'_> {
        PreparedWcs {
            wcs: self,
            projection: self.projection().prepare([self.ra, self.dec]),
            det: self.determinant(),
        }
    }

    /// True angular width through the middle row, including TAN projection/SIP.
    pub fn width_deg(&self, width: u32, height: u32) -> f64 {
        let y = (height as f64 - 1.0) / 2.0;
        self.arc_deg([-0.5, y], [width as f64 - 0.5, y])
    }
    /// Angle (deg) along the straight pixel path from `a` to `b`: the sum over 64 steps, so a
    /// path longer than 180 deg (a fisheye frame) is not folded back as an end-to-end angle
    /// would be. For a frame narrower than 180 deg it equals the end-to-end angle up to the
    /// curvature of the path on the sky.
    pub fn arc_deg(&self, a: [f64; 2], b: [f64; 2]) -> f64 {
        const STEPS: usize = 64;
        let at = |t: f64| {
            let s = self.pixel_to_sky(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]));
            unit(s[0], s[1])
        };
        let mut prev = at(0.0);
        let mut total = 0.0;
        for k in 1..=STEPS {
            let p = at(k as f64 / STEPS as f64);
            total += separation(prev, p);
            prev = p;
        }
        total.to_degrees()
    }
    pub fn determinant(&self) -> f64 {
        self.cd[0][0] * self.cd[1][1] - self.cd[0][1] * self.cd[1][0]
    }
    pub fn pixel_to_sky(&self, x: f64, y: f64) -> [f64; 2] {
        let Some([x, y]) = self.uncorrect_pixel([x, y]) else {
            return [f64::NAN; 2];
        };
        let (mut dx, mut dy) = (x - self.crpix[0], y - self.crpix[1]);
        if let Some(s) = &self.sip {
            let (fa, fb) = s.eval(dx, dy);
            dx += fa;
            dy += fb;
        }
        self.projection().unproject(
            self.cd[0][0] * dx + self.cd[0][1] * dy,
            self.cd[1][0] * dx + self.cd[1][1] * dy,
            [self.ra, self.dec],
        )
    }
    /// Like [`Wcs::sky_to_pixel`] but stops the SIP inversion at 1e-3 px (for matching).
    pub fn sky_to_pixel_fast(&self, ra: f64, dec: f64) -> Option<[f64; 2]> {
        let p = self
            .projection()
            .prepare([self.ra, self.dec])
            .project(ra, dec)?;
        let det = self.determinant();
        if det.abs() < 1e-16 {
            return None;
        }
        let tx = (self.cd[1][1] * p[0] - self.cd[0][1] * p[1]) / det;
        let ty = (-self.cd[1][0] * p[0] + self.cd[0][0] * p[1]) / det;
        let (mut x, mut y) = (tx, ty);
        if let Some(s) = &self.sip {
            for _ in 0..8 {
                let (fa, fb) = s.eval(x, y);
                let ex = x + fa - tx;
                let ey = y + fb - ty;
                let jj = s.jacobian(x, y);
                let d = jj[0][0] * jj[1][1] - jj[0][1] * jj[1][0];
                if d.abs() < 1e-12 {
                    return None;
                }
                let dx = (jj[1][1] * ex - jj[0][1] * ey) / d;
                let dy = (-jj[1][0] * ex + jj[0][0] * ey) / d;
                x -= dx;
                y -= dy;
                if dx.abs() + dy.abs() < 1e-3 {
                    break;
                }
            }
        }
        if x.is_finite() && y.is_finite() {
            Some(self.correct_pixel([x + self.crpix[0], y + self.crpix[1]]))
        } else {
            None
        }
    }
    pub fn sky_to_pixel(&self, ra: f64, dec: f64) -> Option<[f64; 2]> {
        let p = self
            .projection()
            .prepare([self.ra, self.dec])
            .project(ra, dec)?;
        let det = self.determinant();
        if det.abs() < 1e-16 {
            return None;
        }
        let tx = (self.cd[1][1] * p[0] - self.cd[0][1] * p[1]) / det;
        let ty = (-self.cd[1][0] * p[0] + self.cd[0][0] * p[1]) / det;
        let (mut x, mut y) = (tx, ty);
        if let Some(s) = &self.sip {
            for _ in 0..20 {
                let (fa, fb) = s.eval(x, y);
                let err = Vector2::new(x + fa - tx, y + fb - ty);
                let jj = s.jacobian(x, y);
                let j = Matrix2::new(jj[0][0], jj[0][1], jj[1][0], jj[1][1]);
                let delta = j.try_inverse()? * err;
                x -= delta[0];
                y -= delta[1];
                if delta.norm() < 1e-7 {
                    break;
                }
            }
        }
        if x.is_finite() && y.is_finite() {
            Some(self.correct_pixel([x + self.crpix[0], y + self.crpix[1]]))
        } else {
            None
        }
    }
    /// Geometric-mean angular pixel scale at CRPIX, including the cylindrical metric.
    pub fn scale_arcsec(&self) -> f64 {
        let metric = if self.proj.is_cylindrical() {
            let p = self.projection().prepare([self.ra, self.dec]);
            let cos = libm::cos(p.latitude0);
            match self.proj {
                Proj::Car => cos,
                Proj::Mer => cos * cos,
                _ => cos * cos * cos,
            }
        } else {
            1.0
        };
        (self.determinant().abs() * metric).sqrt() * 3600.0
    }
    /// Position angle of image +y, east of celestial north at CRPIX.
    pub fn rotation_deg(&self) -> f64 {
        if !self.proj.is_cylindrical() {
            return self.cd[0][1]
                .atan2(self.cd[1][1])
                .to_degrees()
                .rem_euclid(360.0);
        }
        let p = self.projection().prepare([self.ra, self.dec]);
        let (a, e, n) = p.basis;
        let (sin, cos) = (libm::sin(p.latitude0), libm::cos(p.latitude0));
        let dy = match self.proj {
            Proj::Car => 1.0,
            Proj::Mer => cos,
            _ => cos * cos,
        };
        let v: [f64; 3] = std::array::from_fn(|i| {
            cos * e[i] * self.cd[0][1] + dy * (-sin * a[i] + cos * n[i]) * self.cd[1][1]
        });
        let (_, east, north) = panorama_basis([self.ra, self.dec]);
        let dot = |q: [f64; 3]| q.iter().zip(v).map(|(a, b)| a * b).sum::<f64>();
        libm::atan2(dot(east), dot(north))
            .to_degrees()
            .rem_euclid(360.0)
    }
    pub fn fits_header(&self) -> String {
        let mut cards = vec![
            "SIMPLE  =                    T".to_string(),
            "BITPIX  =                    8".into(),
            "NAXIS   =                    0".into(),
            "EXTEND  =                    T".into(),
        ];
        if self.local_grid.is_some() {
            cards.push(
                "COMMENT Base projection/SIP only; local pixel grid is in the JSON result.".into(),
            );
            cards.push(
                "COMMENT Do not use this header for precise astrometry near stitched edges.".into(),
            );
        }
        let suf = if self.sip.is_some() { "-SIP" } else { "" };
        let code = self.proj.code();
        cards.push(format!("CTYPE1  = 'RA---{code}{suf}'"));
        cards.push(format!("CTYPE2  = 'DEC--{code}{suf}'"));
        if self.proj.is_cylindrical() {
            let p = self.projection().prepare([self.ra, self.dec]);
            let (a, e, n) = p.basis;
            for (k, v) in [
                ("PV1_0", 1.0),
                ("PV1_1", 0.0),
                ("PV1_2", p.latitude0.to_degrees()),
                ("LONPOLE", e[2].atan2(a[2]).to_degrees()),
                ("LATPOLE", n[2].clamp(-1.0, 1.0).asin().to_degrees()),
            ] {
                cards.push(format!("{k:<8}= {v:20.12E}"));
            }
            if self.proj == Proj::Cyp {
                cards.push("PV2_1   =                  0.0".into());
                cards.push("PV2_2   =                  1.0".into());
            }
        }
        cards.push("RADESYS = 'ICRS'".into());
        cards.push("CUNIT1  = 'deg'".into());
        cards.push("CUNIT2  = 'deg'".into());
        for (k, v) in [
            ("CRVAL1", self.ra),
            ("CRVAL2", self.dec),
            ("CRPIX1", self.crpix[0] + 1.0),
            ("CRPIX2", self.crpix[1] + 1.0),
            ("CD1_1", self.cd[0][0]),
            ("CD1_2", self.cd[0][1]),
            ("CD2_1", self.cd[1][0]),
            ("CD2_2", self.cd[1][1]),
        ] {
            cards.push(format!("{k:<8}= {v:20.12E}"));
        }
        if let Some(s) = &self.sip {
            cards.push(format!("A_ORDER = {:20}", s.order));
            cards.push(format!("B_ORDER = {:20}", s.order));
            for (i, (p, q)) in Sip::terms(s.order).into_iter().enumerate() {
                cards.push(format!("{:<8}= {:20.12E}", format!("A_{p}_{q}"), s.a[i]));
                cards.push(format!("{:<8}= {:20.12E}", format!("B_{p}_{q}"), s.b[i]));
            }
        }
        cards.push("END".into());
        let mut out = cards.iter().map(|c| format!("{c:<80}")).collect::<String>();
        let n = out.len().div_ceil(2880) * 2880;
        out.extend(std::iter::repeat_n(' ', n - out.len()));
        out
    }
}

/// The pixel side of a quad fit is identical for every atlas hit. Factor it once
/// and reuse it, while still projecting each catalog hypothesis on the sphere.
#[allow(dead_code)] // used by the F3 hinted-quad fallback (milestone M5)
pub(crate) struct QuadFit {
    inverse: SMatrix<f64, 3, 4>,
    center: [f64; 2],
    scale: f64,
}
#[allow(dead_code)]
impl QuadFit {
    pub fn new(pixels: [[f64; 2]; 4]) -> Option<Self> {
        let center = [
            pixels.iter().map(|p| p[0]).sum::<f64>() / 4.0,
            pixels.iter().map(|p| p[1]).sum::<f64>() / 4.0,
        ];
        let scale = pixels
            .iter()
            .map(|p| (p[0] - center[0]).hypot(p[1] - center[1]))
            .fold(1.0, f64::max);
        let a = SMatrix::<f64, 4, 3>::from_fn(|i, j| match j {
            0 => (pixels[i][0] - center[0]) / scale,
            1 => (pixels[i][1] - center[1]) / scale,
            _ => 1.0,
        });
        let inverse = a.svd(true, true).pseudo_inverse(1e-10).ok()?;
        Some(Self {
            inverse,
            center,
            scale,
        })
    }
    pub fn fit(&self, sky: [[f64; 2]; 4]) -> Option<Wcs> {
        let center = sky[0];
        let projected: Vec<_> = sky
            .into_iter()
            .map(|s| project(s[0], s[1], center))
            .collect::<Option<_>>()?;
        let x = self.inverse * SVector::<f64, 4>::from_fn(|i, _| projected[i][0]);
        let y = self.inverse * SVector::<f64, 4>::from_fn(|i, _| projected[i][1]);
        let cd = [
            [x[0] / self.scale, x[1] / self.scale],
            [y[0] / self.scale, y[1] / self.scale],
        ];
        let offset = Matrix2::new(cd[0][0], cd[0][1], cd[1][0], cd[1][1]).try_inverse()?
            * Vector2::new(x[2], y[2]);
        Some(Wcs {
            ra: center[0],
            dec: center[1],
            crpix: [self.center[0] - offset[0], self.center[1] - offset[1]],
            cd,
            sip: None,
            proj: Proj::Tan,
            native_pole: None,
            local_grid: None,
        })
    }
}

/// Least-squares TAN from correspondences (x,y,ra,dec), with centered pixel features.
pub fn fit_wcs(pairs: &[[f64; 4]], center: [f64; 2]) -> Option<Wcs> {
    if pairs.len() < 3 {
        return None;
    }
    let mx = pairs.iter().map(|p| p[0]).sum::<f64>() / pairs.len() as f64;
    let my = pairs.iter().map(|p| p[1]).sum::<f64>() / pairs.len() as f64;
    let scale = pairs
        .iter()
        .map(|p| (p[0] - mx).hypot(p[1] - my))
        .fold(1.0, f64::max);
    let mut a = DMatrix::zeros(pairs.len(), 3);
    let mut u = DVector::zeros(pairs.len());
    let mut v = u.clone();
    for (i, p) in pairs.iter().enumerate() {
        let q = project(p[2], p[3], center)?;
        a[(i, 0)] = (p[0] - mx) / scale;
        a[(i, 1)] = (p[1] - my) / scale;
        a[(i, 2)] = 1.0;
        u[i] = q[0];
        v[i] = q[1];
    }
    let svd = a.svd(true, true);
    let x = svd.solve(&u, 1e-10).ok()?;
    let y = svd.solve(&v, 1e-10).ok()?;
    let cd = [[x[0] / scale, x[1] / scale], [y[0] / scale, y[1] / scale]];
    let inv = Matrix2::new(cd[0][0], cd[0][1], cd[1][0], cd[1][1]).try_inverse()?;
    let offset = inv * Vector2::new(x[2], y[2]);
    Some(Wcs {
        ra: center[0],
        dec: center[1],
        crpix: [mx - offset[0], my - offset[1]],
        cd,
        sip: None,
        proj: Proj::Tan,
        native_pole: None,
        local_grid: None,
    })
}
/// HEALPix RING pixel center (NSIDE must be positive).
pub fn healpix_center(nside: u32, pix: u64) -> [f64; 2] {
    let n = nside as i64;
    let p = pix as i64;
    let np = 12 * n * n;
    let cap = 2 * n * (n - 1);
    let (z, phi) = if p < cap {
        let ring = ((1.0 + (1.0 + 2.0 * p as f64).sqrt()) / 2.0).floor() as i64;
        let ip = p + 1 - 2 * ring * (ring - 1);
        (
            1.0 - (ring * ring) as f64 / (3 * n * n) as f64,
            (ip as f64 - 0.5) * PI / (2 * ring) as f64,
        )
    } else if p < np - cap {
        let ip = p - cap;
        let ring = ip / (4 * n) + n;
        let az = ip % (4 * n) + 1;
        let f = 0.5 * (1 + ((ring + n) & 1)) as f64;
        (
            (2 * n - ring) as f64 * 2.0 / (3 * n) as f64,
            (az as f64 - f) * PI / (2 * n) as f64,
        )
    } else {
        let ip = np - p;
        let ring = ((1.0 + (2.0 * ip as f64 - 1.0).sqrt()) / 2.0).floor() as i64;
        let az = 4 * ring + 1 - (ip - 2 * ring * (ring - 1));
        (
            -1.0 + (ring * ring) as f64 / (3 * n * n) as f64,
            (az as f64 - 0.5) * PI / (2 * ring) as f64,
        )
    };
    [phi.to_degrees().rem_euclid(360.0), z.asin().to_degrees()]
}

#[cfg(test)]
mod arc_tests {
    use super::*;

    /// `RA---ARC` / `RA---ARC-SIP` values computed with astropy 7 (wcslib) for the same header
    /// (CRPIX is one-based there, zero-based here).
    #[test]
    fn arc_matches_astropy_and_round_trips() {
        let mut w = Wcs {
            ra: 163.0,
            dec: 71.5,
            crpix: [999.5, 666.0],
            cd: [[-0.05, -0.007], [0.007, -0.05]],
            sip: None,
            proj: Proj::Arc,
            native_pole: None,
            local_grid: None,
        };
        for (px, want) in [
            ([-1.0, -1.0], [267.53348, 35.736778]),
            ([1998.0, 1331.0], [106.222256, 20.253008]),
            ([499.0, 899.0], [199.957084, 49.257924]),
        ] {
            let s = w.pixel_to_sky(px[0], px[1]);
            assert!(
                (s[0] - want[0]).abs() < 2e-6 && (s[1] - want[1]).abs() < 2e-6,
                "{s:?}"
            );
            let back = w.sky_to_pixel(s[0], s[1]).unwrap();
            assert!((back[0] - px[0]).hypot(back[1] - px[1]) < 1e-6);
        }
        w.sip = Some(Sip {
            order: 2,
            a: vec![1e-4, 0.0, 0.0],
            b: vec![0.0, 0.0, -1e-4],
        });
        let s = w.pixel_to_sky(1499.0, 99.0);
        assert!((s[0] - 35.674462).abs() < 2e-6 && (s[1] - 63.59072).abs() < 2e-6);
        assert!(w.fits_header().contains("'RA---ARC-SIP'"));
        // beyond 90 deg from the reference point (a 200 deg fisheye) ARC stays invertible
        let far = unproject_arc(0.0, 100.0, [10.0, 20.0]);
        let p = project_arc(far[0], far[1], [10.0, 20.0]).unwrap();
        assert!(p[0].abs() < 1e-9 && (p[1] - 100.0).abs() < 1e-9);
        assert_eq!(project_p(Proj::Tan, far[0], far[1], [10.0, 20.0]), None);
    }

    #[test]
    fn tan_wcs_json_has_no_proj_field_and_old_json_reads_as_tan() {
        let w: Wcs = serde_json::from_str(
            r#"{"ra":1.0,"dec":2.0,"crpix":[3.0,4.0],"cd":[[1e-3,0.0],[0.0,1e-3]],"sip":null}"#,
        )
        .unwrap();
        assert_eq!(w.proj, Proj::Tan);
        assert!(!serde_json::to_string(&w).unwrap().contains("proj"));
        let a = Wcs {
            proj: Proj::Arc,
            native_pole: None,
            local_grid: None,
            ..w
        };
        let j = serde_json::to_string(&a).unwrap();
        assert!(j.contains(r#""proj":"ARC""#));
        assert_eq!(serde_json::from_str::<Wcs>(&j).unwrap().proj, Proj::Arc);
    }
}

#[cfg(test)]
mod quad_fit_tests {
    use super::*;
    #[test]
    fn reusable_fit_agrees_with_full_svd_for_distinct_sky_hypotheses() {
        let pixels = [[10.0, 30.0], [983.0, 60.0], [420.0, 680.0], [170.0, 320.0]];
        let fit = QuadFit::new(pixels).unwrap();
        for center in [[359.8, 72.0], [83.0, -5.0], [123.0, -89.0]] {
            for scale in [0.0002, 0.012] {
                for parity in [-1.0, 1.0] {
                    let truth = Wcs {
                        ra: center[0],
                        dec: center[1],
                        crpix: [511.5, 383.5],
                        cd: [
                            [scale * parity, scale * 0.2],
                            [-scale * 0.2 * parity, scale],
                        ],
                        sip: None,
                        proj: Proj::Tan,
                        native_pole: None,
                        local_grid: None,
                    };
                    let sky = pixels.map(|p| truth.pixel_to_sky(p[0], p[1]));
                    let pairs: Vec<_> = pixels
                        .iter()
                        .zip(sky)
                        .map(|(p, s)| [p[0], p[1], s[0], s[1]])
                        .collect();
                    let a = fit.fit(sky).unwrap();
                    let b = fit_wcs(&pairs, sky[0]).unwrap();
                    for p in [[0.0, 0.0], [1023.0, 767.0], [512.0, 384.0]] {
                        let q = a.pixel_to_sky(p[0], p[1]);
                        let r = b.sky_to_pixel(q[0], q[1]).unwrap();
                        assert!((r[0] - p[0]).hypot(r[1] - p[1]) < 1e-6);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod prepared_tests {
    use super::*;

    fn distortion(order: u32) -> Sip {
        let terms = Sip::terms(order);
        Sip {
            order,
            a: terms
                .iter()
                .enumerate()
                .map(|(k, &(p, q))| {
                    (k as f64 + 0.3).sin() * 2e-6 / 1000f64.powi((p + q - 2) as i32)
                })
                .collect(),
            b: terms
                .iter()
                .enumerate()
                .map(|(k, &(p, q))| {
                    (k as f64 + 0.7).cos() * 2e-6 / 1000f64.powi((p + q - 2) as i32)
                })
                .collect(),
        }
    }

    #[test]
    fn prepared_projection_matches_scalar_at_poles_wrap_and_horizon() {
        for proj in [Proj::Tan, Proj::Arc] {
            for center in [[359.9, 89.99], [0.1, -89.99], [123.0, 0.0], [42.0, 67.0]] {
                let prepared = PreparedProjection::new(proj, center);
                for ra in [-0.01, 0.0, 0.1, 90.0, 180.0, 269.9999, 359.9, 360.0] {
                    for dec in [-90.0, -89.9999, -17.0, 0.0, 67.0, 89.9999, 90.0] {
                        assert_eq!(prepared.project(ra, dec), project_p(proj, ra, dec, center));
                    }
                }
                assert_eq!(
                    prepared.project(center[0], center[1]),
                    project_p(proj, center[0], center[1], center)
                );
            }
        }
    }

    #[test]
    fn prepared_wcs_preserves_matching_pixels_for_sip_and_both_parities() {
        for proj in [Proj::Tan, Proj::Arc] {
            for center in [[359.8, 72.0], [83.0, -5.0], [123.0, -89.0]] {
                for parity in [-1.0, 1.0] {
                    for order in [0, 2, 3, 5, 7] {
                        let wcs = Wcs {
                            ra: center[0],
                            dec: center[1],
                            crpix: [999.5, 749.5],
                            cd: [[-0.013 * parity, 0.002], [0.001 * parity, 0.013]],
                            sip: (order > 0).then(|| distortion(order)),
                            proj,
                            native_pole: None,
                            local_grid: None,
                        };
                        let prepared = wcs.prepare();
                        for x in [-300.0, 0.0, 200.25, 999.5, 1700.0, 1999.0, 2300.0] {
                            for y in [-200.0, 0.0, 380.75, 749.5, 1499.0, 1800.0] {
                                let sky = wcs.pixel_to_sky(x, y);
                                let a = prepared.sky_to_pixel_fast(sky[0], sky[1]);
                                let b = wcs.sky_to_pixel_fast(sky[0], sky[1]);
                                assert_eq!(a, b, "{proj:?} {center:?} {parity} {order} {x},{y}");
                                let back = a.unwrap();
                                assert!((back[0] - x).hypot(back[1] - y) < 1e-3);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn prepared_wcs_strict_matches_scalar_for_tan_arc_and_sip() {
        for proj in [Proj::Tan, Proj::Arc] {
            for center in [[359.8, 72.0], [83.0, -5.0], [123.0, -89.0]] {
                for parity in [-1.0, 1.0] {
                    for order in [0, 2, 3, 5, 7] {
                        let wcs = Wcs {
                            ra: center[0],
                            dec: center[1],
                            crpix: [999.5, 749.5],
                            cd: [[-0.013 * parity, 0.002], [0.001 * parity, 0.013]],
                            sip: (order > 0).then(|| distortion(order)),
                            proj,
                            native_pole: None,
                            local_grid: None,
                        };
                        let prepared = wcs.prepare();
                        for x in [-300.0, 0.0, 200.25, 999.5, 1700.0, 1999.0, 2300.0] {
                            for y in [-200.0, 0.0, 380.75, 749.5, 1499.0, 1800.0] {
                                let sky = wcs.pixel_to_sky(x, y);
                                let a = prepared.sky_to_pixel(sky[0], sky[1]);
                                let b = wcs.sky_to_pixel(sky[0], sky[1]);
                                assert_eq!(a, b, "{proj:?} {center:?} {parity} {order} {x},{y}");
                                let back = a.unwrap();
                                assert!((back[0] - x).hypot(back[1] - y) < 1e-6);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn prepared_wcs_strict_preserves_failed_and_iteration_limited_inversions() {
        for proj in [Proj::Tan, Proj::Arc] {
            // Choose CD so the first inverse estimate is exactly (1, 0).
            let tx = project_p(proj, 1.0, 0.0, [0.0, 0.0]).unwrap()[0];
            let mut wcs = Wcs {
                ra: 0.0,
                dec: 0.0,
                crpix: [100.0, 100.0],
                cd: [[tx, 0.0], [0.0, 1.0]],
                sip: Some(Sip {
                    order: 2,
                    a: vec![-0.5, 0.0, 0.0],
                    b: vec![0.0; 3],
                }),
                proj,
                native_pole: None,
                local_grid: None,
            };
            // x' = x - x²/2 has a singular derivative at the initial x=1.
            assert_eq!(wcs.sky_to_pixel(1.0, 0.0), None);
            assert_eq!(wcs.prepare().sky_to_pixel(1.0, 0.0), None);
            wcs.sip.as_mut().unwrap().a[0] = f64::NAN;
            assert_eq!(wcs.sky_to_pixel(1.0, 0.0), None);
            assert_eq!(wcs.prepare().sky_to_pixel(1.0, 0.0), None);
            // x' = x - x² cannot reach 1. Newton alternates 1/0. Existing strict
            // inversion returns its finite twentieth iterate, rather than None.
            wcs.sip.as_mut().unwrap().a[0] = -1.0;
            assert_eq!(wcs.sky_to_pixel(1.0, 0.0), Some([101.0, 100.0]));
            assert_eq!(
                wcs.prepare().sky_to_pixel(1.0, 0.0),
                wcs.sky_to_pixel(1.0, 0.0)
            );
            assert_eq!(
                wcs.prepare().sky_to_pixel(180.0, 0.0),
                wcs.sky_to_pixel(180.0, 0.0)
            );
            wcs.cd = [[0.01, 0.02], [0.02, 0.04]];
            assert_eq!(wcs.prepare().sky_to_pixel(0.0, 0.0), None);
        }
    }

    #[test]
    fn prepared_wcs_rejects_invalid_projection_and_singular_cd() {
        let mut w = Wcs {
            ra: 0.0,
            dec: 0.0,
            crpix: [100.0, 100.0],
            cd: [[0.01, 0.0], [0.0, 0.01]],
            sip: None,
            proj: Proj::Tan,
            native_pole: None,
            local_grid: None,
        };
        assert_eq!(w.prepare().sky_to_pixel_fast(180.0, 0.0), None);
        w.proj = Proj::Arc;
        assert_eq!(w.prepare().sky_to_pixel_fast(180.0, 0.0), None);
        w.cd = [[0.01, 0.02], [0.02, 0.04]];
        assert_eq!(w.prepare().sky_to_pixel_fast(0.0, 0.0), None);
    }

    #[test]
    fn fused_sip_matches_separate_evaluation_and_finite_difference() {
        for order in 2..=7 {
            let sip = distortion(order);
            for (u, v) in [
                (-1200.0, 700.0),
                (0.0, 0.0),
                (400.0, -1100.0),
                (23.5, 19.75),
            ] {
                let (value, jac) = sip.eval_jacobian(u, v);
                assert_eq!(value, sip.eval(u, v));
                assert_eq!(jac, sip.jacobian(u, v));
                let eps = 0.001;
                let xp = sip.eval(u + eps, v);
                let xm = sip.eval(u - eps, v);
                let yp = sip.eval(u, v + eps);
                let ym = sip.eval(u, v - eps);
                let finite = [
                    [
                        1.0 + (xp.0 - xm.0) / (2.0 * eps),
                        (yp.0 - ym.0) / (2.0 * eps),
                    ],
                    [
                        (xp.1 - xm.1) / (2.0 * eps),
                        1.0 + (yp.1 - ym.1) / (2.0 * eps),
                    ],
                ];
                for i in 0..2 {
                    for j in 0..2 {
                        assert!((finite[i][j] - jac[i][j]).abs() < 1e-8);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod panorama_tests {
    use super::*;
    #[test]
    fn panorama_projections_match_wcslib_and_round_trip() {
        // Independent oracle; regenerate with bench/panorama_wcs_fixture.py (astropy/WCSLIB).
        let data: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/panorama-wcs.json")).unwrap();
        for f in data["fixtures"].as_array().unwrap() {
            let w: Wcs = serde_json::from_value(f["wcs"].clone()).unwrap();
            let pixels: Vec<[f64; 2]> = serde_json::from_value(f["pixels"].clone()).unwrap();
            let sky: Vec<[f64; 2]> = serde_json::from_value(f["sky"].clone()).unwrap();
            for (px, want) in pixels.iter().zip(&sky) {
                let got = w.pixel_to_sky(px[0], px[1]);
                assert!(
                    separation(unit(got[0], got[1]), unit(want[0], want[1])) < 1e-10,
                    "{:?} {:?}: {got:?} != {want:?}",
                    w.proj,
                    w.native_pole
                );
                let back = w.sky_to_pixel(want[0], want[1]).unwrap();
                assert!((px[0] - back[0]).hypot(px[1] - back[1]) < 1e-6);
                assert_eq!(
                    w.prepare().sky_to_pixel(want[0], want[1]),
                    w.sky_to_pixel(want[0], want[1])
                );
                let atlas = crate::coordinates::pixel_to_sky(&w, px[0], px[1]);
                assert_eq!(atlas, got);
            }
            let header = w.fits_header();
            assert!(header.contains(&format!("RA---{}", w.proj.code())));
            for (key, want) in f["fits_parameters"].as_object().unwrap() {
                let card = header
                    .as_bytes()
                    .chunks(80)
                    .map(|c| std::str::from_utf8(c).unwrap())
                    .find(|c| c[..8].trim() == key)
                    .unwrap();
                let got: f64 = card[10..].trim().parse().unwrap();
                assert!(
                    (got - want.as_f64().unwrap()).abs() < 1e-9,
                    "{key}: {got} != {want}"
                );
            }
            if w.proj.is_cylindrical() {
                assert!(header.contains("PV1_2"));
                assert!(header.contains("LATPOLE"));
            }
            assert_eq!(
                serde_json::from_str::<Wcs>(&serde_json::to_string(&w).unwrap())
                    .unwrap()
                    .native_pole,
                w.native_pole
            );
        }
    }
    #[test]
    fn cylindrical_scale_and_rotation_use_celestial_metric() {
        for proj in [Proj::Car, Proj::Mer, Proj::Cyp] {
            let w = Wcs {
                ra: 283.,
                dec: -4.,
                crpix: [682., 1023.5],
                cd: [[-0.075, 0.002], [0.003, 0.065]],
                sip: None,
                proj,
                native_pole: Some([40., 60.]),
                local_grid: None,
            };
            let c = unit(w.ra, w.dec);
            let mut delta = [[0.; 3]; 2];
            let step = 0.001;
            for (i, axis) in [[step, 0.], [0., step]].iter().enumerate() {
                let sky = w.pixel_to_sky(w.crpix[0] + axis[0], w.crpix[1] + axis[1]);
                let q = unit(sky[0], sky[1]);
                delta[i] = std::array::from_fn(|k| (q[k] - c[k]) / step);
            }
            let (u, v) = (delta[0], delta[1]);
            let cross = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let numeric = cross
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt()
                .sqrt()
                .to_degrees()
                * 3600.;
            assert!((numeric / w.scale_arcsec() - 1.).abs() < 1e-5);
            let (_, e, n) = zenithal_basis([w.ra, w.dec]);
            let dot = |q: [f64; 3]| q.iter().zip(v).map(|(a, b)| a * b).sum::<f64>();
            let rot = dot(e).atan2(dot(n)).to_degrees().rem_euclid(360.);
            assert!((rot - w.rotation_deg()).abs() < 1e-4);
        }
    }

    #[test]
    fn panorama_domains_seams_and_transverse_axis() {
        for proj in [Proj::Car, Proj::Mer, Proj::Cyp] {
            let p = Projection {
                proj,
                native_pole: Some([0., 90.]),
            }
            .prepare([359., 0.]);
            let xy = p.project(1., 0.).unwrap();
            assert!((xy[0] - 2.).abs() < 1e-10);
            assert!(p.project(0., 90.).is_none());
            assert!(p.unproject(181., 0.)[0].is_nan());
            assert!(p.project(f64::NAN, 0.).is_none());
            let transverse = Projection {
                proj,
                native_pole: Some([0., 0.]),
            }
            .prepare([90., 0.]);
            let sky = transverse.unproject(120., 0.);
            let back = transverse.project(sky[0], sky[1]).unwrap();
            assert!((back[0] - 120.).abs() < 1e-9 && back[1].abs() < 1e-9);
        }
        assert!(project_p(Proj::Stg, 180., 0., [0., 0.]).is_none());
    }
}
