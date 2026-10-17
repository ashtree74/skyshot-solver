//! Offline paired benchmark driver; input paths and results are supplied by the caller.
//! This example does not add a public request protocol to the CLI.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde::Deserialize;
use skyshot::{api::SolveMode, image_io, index::Index, pipeline, solve::SolveConfig};
use std::{io::Write, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    learned: Option<PathBuf>,
    #[arg(long)]
    handcrafted: Option<PathBuf>,
    /// A single-index, single-image run for fresh-process measurements.
    #[arg(long)]
    index: Option<PathBuf>,
    #[arg(long)]
    only_frame: Option<String>,
    #[arg(long, default_value_t = 4)]
    threads: usize,
    #[arg(long, default_value_t = 2000.0)]
    deadline_ms: f64,
}

#[derive(Deserialize)]
struct Frame {
    id: String,
    image: PathBuf,
}

#[derive(Deserialize)]
struct Plan {
    frames: Vec<Frame>,
    orders: Vec<Vec<usize>>,
}

fn run_frame(index: &Index, frame: &Frame, cfg: &SolveConfig) -> serde_json::Value {
    let begin = Instant::now();
    let result = (|| -> Result<serde_json::Value> {
        let decode = Instant::now();
        let (gray, mut info) = image_io::load_with_info(&frame.image)?;
        let decode_ms = decode.elapsed().as_secs_f64() * 1000.0;
        // Blind comparison: retain the decoded pixels and sample format, discard pointing/scale.
        info.hint = None;
        let mut result = pipeline::solve_gray(index, &gray, &info, cfg);
        result.timings.decode_ms = decode_ms;
        result.timings.total_ms += decode_ms;
        ensure!(
            !result.solution.stats.hint_used,
            "benchmark unexpectedly used a hint"
        );
        ensure!(
            !result.solution.stats.deep_search,
            "benchmark unexpectedly used DEEP"
        );
        let wcs_header = result.solution.wcs.as_ref().map(|w| w.fits_header());
        drop(gray);
        let call_ms = begin.elapsed().as_secs_f64() * 1000.0;
        Ok(
            serde_json::json!({"solution": result.solution, "timings": result.timings,
            "n_stars": result.n_stars, "call_ms": call_ms, "wcs_header": wcs_header}),
        )
    })();
    match result {
        Ok(value) => value,
        Err(error) => serde_json::json!({"error": format!("{error:#}"),
            "call_ms": begin.elapsed().as_secs_f64() * 1000.0}),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.threads > 0, "threads must be positive");
    ensure!(
        args.deadline_ms.is_finite() && args.deadline_ms > 0.0,
        "invalid deadline"
    );
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()?;
    let plan: Plan = serde_json::from_reader(std::fs::File::open(&args.manifest)?)?;
    let cfg = SolveConfig {
        mode: SolveMode::Standard,
        deadline_ms: args.deadline_ms,
        allow_arc: false,
        allow_panorama: false,
        ..Default::default()
    };
    let paths = if let Some(index) = args.index {
        vec![("fresh", index)]
    } else {
        vec![
            ("learned", args.learned.context("--learned required")?),
            (
                "handcrafted",
                args.handcrafted.context("--handcrafted required")?,
            ),
        ]
    };
    let mut indexes = Vec::new();
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for (arm, path) in paths {
        let begin = Instant::now();
        // Explicit main index only: no auto-discovered learned companions for either arm.
        let index = Index::open(&path)?;
        let load_ms = begin.elapsed().as_secs_f64() * 1000.0;
        indexes.push((arm, index, load_ms));
    }
    if let Some(id) = args.only_frame {
        ensure!(indexes.len() == 1, "fresh run needs --index");
        let frame = plan
            .frames
            .iter()
            .find(|f| f.id == id)
            .context("unknown frame")?;
        let mut row = run_frame(&indexes[0].1, frame, &cfg);
        row["id"] = serde_json::json!(frame.id);
        row["phase"] = serde_json::json!("fresh");
        row["index_load_ms"] = serde_json::json!(indexes[0].2);
        writeln!(out, "{}", serde_json::to_string(&row)?)?;
    } else {
        ensure!(
            indexes.len() == 2 && !plan.frames.is_empty(),
            "paired run needs both indexes and frames"
        );
        for round in 0..=plan.orders.len() {
            let warmup_order: Vec<usize> = (0..plan.frames.len()).collect();
            let order = if round == 0 {
                &warmup_order
            } else {
                &plan.orders[round - 1]
            };
            ensure!(order.len() == plan.frames.len(), "incomplete frame order");
            let mut seen = std::collections::HashSet::new();
            for (position, &i) in order.iter().enumerate() {
                ensure!(
                    i < plan.frames.len() && seen.insert(i),
                    "invalid/duplicate frame order"
                );
                let frame = &plan.frames[i];
                let arms = if (round + position) % 2 == 0 {
                    [0, 1]
                } else {
                    [1, 0]
                };
                for arm in arms {
                    let (name, index, load_ms) = &indexes[arm];
                    let mut row = run_frame(index, frame, &cfg);
                    row["id"] = serde_json::json!(frame.id);
                    row["arm"] = serde_json::json!(name);
                    row["round"] = serde_json::json!(round);
                    row["phase"] =
                        serde_json::json!(if round == 0 { "warmup" } else { "measured" });
                    row["index_load_ms"] = serde_json::json!(load_ms);
                    writeln!(out, "{}", serde_json::to_string(&row)?)?;
                    out.flush()?;
                }
                if (position + 1) % 10 == 0 {
                    eprintln!(
                        "round {round}: {}/{} paired images",
                        position + 1,
                        plan.frames.len()
                    );
                }
            }
        }
    }
    out.flush()?;
    Ok(())
}
