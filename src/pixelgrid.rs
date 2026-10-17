//! A small, invertible residual map after the ordinary WCS. Values are measured-image
//! minus base-WCS pixels, bilinearly interpolated and held at the border off the frame.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "GridWire")]
pub struct PixelGrid {
    pub cols: usize,
    pub rows: usize,
    pub origin: [f64; 2],
    pub step: [f64; 2],
    pub offsets: Vec<[f64; 2]>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GridWire {
    cols: usize,
    rows: usize,
    origin: [f64; 2],
    step: [f64; 2],
    offsets: Vec<[f64; 2]>,
}
impl TryFrom<GridWire> for PixelGrid {
    type Error = String;
    fn try_from(w: GridWire) -> Result<Self, Self::Error> {
        let g = Self {
            cols: w.cols,
            rows: w.rows,
            origin: w.origin,
            step: w.step,
            offsets: w.offsets,
        };
        g.validate()?;
        Ok(g)
    }
}
impl PixelGrid {
    pub fn validate(&self) -> Result<(), String> {
        if self.cols < 2
            || self.rows < 2
            || self.cols > 129
            || self.rows > 129
            || self.offsets.len() != self.cols * self.rows
            || !self.origin.iter().all(|x| x.is_finite())
            || !self.step.iter().all(|x| x.is_finite() && *x > 0.)
            || !self
                .offsets
                .iter()
                .flatten()
                .all(|x| x.is_finite() && x.abs() <= 256.)
        {
            return Err("invalid local pixel grid".into());
        }
        // A global contraction bound makes p + offset(p) one-to-one, including
        // off-frame continuation. Checking both sides bounds every bilinear cell.
        for y in 0..self.rows - 1 {
            for x in 0..self.cols - 1 {
                let a = self.offsets[y * self.cols + x];
                let b = self.offsets[y * self.cols + x + 1];
                let c = self.offsets[(y + 1) * self.cols + x];
                let d = self.offsets[(y + 1) * self.cols + x + 1];
                for k in 0..2 {
                    let dx = (b[k] - a[k]).abs().max((d[k] - c[k]).abs()) / self.step[0];
                    let dy = (c[k] - a[k]).abs().max((d[k] - b[k]).abs()) / self.step[1];
                    if dx + dy >= 0.75 {
                        return Err("folding local pixel grid".into());
                    }
                }
            }
        }
        Ok(())
    }
    #[inline]
    pub fn offset(&self, p: [f64; 2]) -> [f64; 2] {
        let u = ((p[0] - self.origin[0]) / self.step[0]).clamp(0., (self.cols - 1) as f64);
        let v = ((p[1] - self.origin[1]) / self.step[1]).clamp(0., (self.rows - 1) as f64);
        let x = (u.floor() as usize).min(self.cols - 2);
        let y = (v.floor() as usize).min(self.rows - 2);
        let (u, v) = (u - x as f64, v - y as f64);
        std::array::from_fn(|k| {
            (1. - v)
                * ((1. - u) * self.offsets[y * self.cols + x][k]
                    + u * self.offsets[y * self.cols + x + 1][k])
                + v * ((1. - u) * self.offsets[(y + 1) * self.cols + x][k]
                    + u * self.offsets[(y + 1) * self.cols + x + 1][k])
        })
    }
    pub fn forward(&self, p: [f64; 2]) -> [f64; 2] {
        let d = self.offset(p);
        [p[0] + d[0], p[1] + d[1]]
    }
    pub fn inverse(&self, p: [f64; 2]) -> Option<[f64; 2]> {
        if !p.iter().all(|x| x.is_finite()) {
            return None;
        }
        let mut q = p;
        for _ in 0..80 {
            let d = self.offset(q);
            let next = [p[0] - d[0], p[1] - d[1]];
            if (next[0] - q[0]).abs().max((next[1] - q[1]).abs()) < 1e-7 {
                return Some(next);
            }
            q = next;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn grid() -> PixelGrid {
        PixelGrid {
            cols: 3,
            rows: 3,
            origin: [0.; 2],
            step: [100.; 2],
            offsets: (0..3)
                .flat_map(|y| {
                    (0..3).map(move |x| {
                        [
                            3. + 2. * x as f64 + 4. * y as f64,
                            -5. - 3. * x as f64 + 1.5 * y as f64,
                        ]
                    })
                })
                .collect(),
        }
    }
    #[test]
    fn affine_oracle_inverse_and_border_continuation() {
        let g = grid();
        g.validate().unwrap();
        for x in [-100., 0., 13., 100., 177., 200., 300.] {
            for y in [-50., 0., 54., 120., 200., 500.] {
                let p = [x, y];
                let c = g.forward(p);
                let cx = x.clamp(0., 200.);
                let cy = y.clamp(0., 200.);
                assert!((c[0] - (x + 3. + 0.02 * cx + 0.04 * cy)).abs() < 1e-12);
                assert!((c[1] - (y - 5. - 0.03 * cx + 0.015 * cy)).abs() < 1e-12);
                let q = g.inverse(c).unwrap();
                assert!((q[0] - x).hypot(q[1] - y) < 1e-6);
            }
        }
    }
    #[test]
    fn serialization_rejects_malformed_or_folding_grids() {
        let g = grid();
        let j = serde_json::to_value(&g).unwrap();
        assert!(serde_json::from_value::<PixelGrid>(j.clone()).is_ok());
        for bad in [serde_json::json!(0), serde_json::json!(130)] {
            let mut v = j.clone();
            v["cols"] = bad;
            assert!(serde_json::from_value::<PixelGrid>(v).is_err());
        }
        let mut v = j.clone();
        v["step"][0] = serde_json::json!(0);
        assert!(serde_json::from_value::<PixelGrid>(v).is_err());
        let mut v = j;
        v["offsets"][0][0] = serde_json::json!(100);
        assert!(serde_json::from_value::<PixelGrid>(v).is_err());
    }
    #[test]
    fn every_wcs_consumer_uses_the_same_grid_on_both_sides_of_sip() {
        use crate::types::{Proj, Sip, Wcs};
        for proj in [
            Proj::Tan,
            Proj::Arc,
            Proj::Stg,
            Proj::Car,
            Proj::Mer,
            Proj::Cyp,
        ] {
            let base = Wcs {
                ra: 140.,
                dec: 30.,
                crpix: [100.; 2],
                cd: [[-0.1, 0.02], [0.01, 0.1]],
                sip: Some(Sip {
                    order: 2,
                    a: vec![1e-5, 2e-5, -1e-5],
                    b: vec![0., -1e-5, 1e-5],
                }),
                proj,
                native_pole: None,
                local_grid: None,
            };
            let mut corrected = base.clone();
            corrected.local_grid = Some(grid());
            for p in [[0., 0.], [40., 70.], [190., 180.], [250., 100.]] {
                let s = base.pixel_to_sky(p[0], p[1]);
                let actual = corrected.correct_pixel(p);
                for got in [
                    corrected.sky_to_pixel(s[0], s[1]),
                    corrected.sky_to_pixel_fast(s[0], s[1]),
                    corrected.prepare().sky_to_pixel(s[0], s[1]),
                    corrected.prepare().sky_to_pixel_fast(s[0], s[1]),
                    crate::coordinates::sky_to_pixel(&corrected, s[0], s[1]),
                ] {
                    let q = got.unwrap();
                    assert!(
                        (q[0] - actual[0]).hypot(q[1] - actual[1]) < 1e-3,
                        "{proj:?}"
                    );
                }
                for q in [
                    corrected.pixel_to_sky(actual[0], actual[1]),
                    crate::coordinates::pixel_to_sky(&corrected, actual[0], actual[1]),
                ] {
                    assert!((q[0] - s[0]).hypot(q[1] - s[1]) < 1e-6);
                }
            }
            assert!(corrected.fits_header().contains("Base projection/SIP only"));
        }
    }
}
