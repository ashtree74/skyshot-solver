//! cargo run --release --example repeated_solve -- /path/to/index frame1.fits frame2.fits
use anyhow::{Context, Result};
use skyshot::{Engine, solve::SolveConfig};

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let index = args
        .next()
        .context("usage: repeated_solve INDEX IMAGE [IMAGE ...]")?;
    let engine = Engine::open(index)?;
    let config = SolveConfig::default();
    for path in args {
        let result = engine.solve_path(path, &config)?;
        println!("{}", serde_json::to_string(&result)?);
    }
    Ok(())
}
