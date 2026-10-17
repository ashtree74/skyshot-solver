use crate::{
    index::{Index, companion_wide_dir},
    pipeline::{self, PipelineTimings},
    solve::{self, DeepIndex, Solution, SolveConfig, WideIndex},
    types::StarField,
};
use anyhow::{Result, ensure};
use serde::Serialize;
use std::{path::Path, sync::Arc, time::Instant};

/// Owns a memory-mapped index and optional wide/deep companions across many solves.
/// Open once per session. No network calls, account, agent, or activation is involved.
pub struct Engine {
    index: Index,
    deep: Option<DeepIndex>,
    wide: Option<WideIndex>,
}

/// The same JSON envelope is used for image and source-list inputs.
#[derive(Serialize)]
pub struct SolveOutput {
    pub schema_version: u32,
    pub solution: Solution,
    pub timings: PipelineTimings,
    pub n_stars: usize,
}

impl Engine {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let index = Index::open(path)?;
        let deep = DeepIndex::companion(path)?;
        let wide = companion_wide_dir(path)
            .map(|p| Index::open(&p).map(|i| WideIndex(Arc::new(i))))
            .transpose()?;
        Ok(Self { index, deep, wide })
    }

    pub fn index(&self) -> &Index {
        &self.index
    }

    fn config(&self, cfg: &SolveConfig) -> Result<SolveConfig> {
        ensure!(
            cfg.deadline_ms.is_finite() && cfg.deadline_ms > 0.0,
            "deadline_ms must be finite and positive"
        );
        ensure!(
            cfg.min_fov_deg.is_finite()
                && cfg.max_fov_deg.is_finite()
                && cfg.min_fov_deg > 0.0
                && cfg.max_fov_deg >= cfg.min_fov_deg,
            "invalid field-of-view bounds"
        );
        let cfg = SolveConfig {
            deep_index: cfg.deep_index.clone().or_else(|| self.deep.clone()),
            wide_index: cfg.wide_index.clone().or_else(|| self.wide.clone()),
            ..cfg.clone()
        };
        ensure!(
            cfg.mode != crate::api::SolveMode::Deep
                || cfg.deep_index.is_some()
                || self.index.manifest.profile == "deep",
            "DEEP mode requires a compatible index-deep companion or a DEEP main index"
        );
        Ok(cfg)
    }

    pub fn solve_path(&self, path: impl AsRef<Path>, cfg: &SolveConfig) -> Result<SolveOutput> {
        let result = pipeline::solve_path(&self.index, path.as_ref(), &self.config(cfg)?)?;
        Ok(SolveOutput {
            schema_version: 1,
            solution: result.solution,
            timings: result.timings,
            n_stars: result.n_stars,
        })
    }

    /// Source positions are zero-based native pixels; brightness is expressed as flux.
    pub fn solve_sources(&self, field: &StarField, cfg: &SolveConfig) -> Result<SolveOutput> {
        ensure!(
            (16..=50_000).contains(&field.width) && (16..=50_000).contains(&field.height),
            "source-list dimensions must be between 16 and 50000 pixels"
        );
        ensure!(
            field.stars.iter().all(|s| s.x.is_finite()
                && s.y.is_finite()
                && s.flux.is_finite()
                && s.sigma.is_finite()
                && s.sigma > 0.0),
            "source positions, flux and uncertainties must be finite; sigma must be positive"
        );
        let cfg = self.config(cfg)?;
        let start = Instant::now();
        let solution = solve::solve_field(&self.index, field, &cfg);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        Ok(SolveOutput {
            schema_version: 1,
            solution,
            timings: PipelineTimings {
                solve_ms: elapsed,
                total_ms: elapsed,
                attempts: 1,
                ..Default::default()
            },
            n_stars: field.stars.len(),
        })
    }
}
