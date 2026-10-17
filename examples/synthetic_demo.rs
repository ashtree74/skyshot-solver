//! Generate a tiny artificial catalogue and star list for a full offline CLI demonstration.
mod support;
use anyhow::{Context, Result};
use std::path::PathBuf;

fn main() -> Result<()> {
    let root = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: synthetic_demo NEW_DIRECTORY")?;
    std::fs::create_dir(&root).context("Use a new directory with an existing parent")?;
    let (field, expected) = support::synthetic_catalog(&root.join("index"))?;
    std::fs::write(root.join("stars.json"), serde_json::to_vec_pretty(&field)?)?;
    std::fs::write(
        root.join("expected-wcs.json"),
        serde_json::to_vec_pretty(&expected)?,
    )?;
    println!("Created artificial index and stars in {}", root.display());
    Ok(())
}
