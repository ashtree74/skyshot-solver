use anyhow::{Context, Result, ensure};
use clap::{Args, Parser, Subcommand};
use skyshot::{Engine, SolveOutput, api::SolveMode, builder, image_io, solve::SolveConfig};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::ExitCode,
};

#[derive(Parser)]
#[command(
    version,
    about = "Offline plate solving for developers — CLI and Rust library"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Solve an image (FITS/PNG/JPEG/TIFF) or StarField JSON, writing JSON to stdout.
    Solve {
        input: PathBuf,
        #[command(flatten)]
        options: Options,
        /// Write a new FITS WCS header file on success (refuses existing files).
        #[arg(long)]
        wcs: Option<PathBuf>,
    },
    /// Solve several files with one open index; emit one JSON result per line.
    Batch {
        #[arg(required = true, num_args = 1..)]
        inputs: Vec<PathBuf>,
        #[command(flatten)]
        options: Options,
    },
    /// Detect stars and emit a StarField JSON document, without an index.
    Extract { input: PathBuf },
    /// Open an index and print its manifest (reader checks, not a full checksum audit).
    Inspect { index: PathBuf },
    /// Build an index from a separately obtained catalogue. Always use a NEW output directory.
    Build {
        #[arg(long)]
        d80: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "full", value_parser = ["lite", "standard", "full", "deep"])]
        profile: String,
        /// Learned coder prefix (.f32 and .json); omitted means the handcrafted v1 key.
        #[arg(long)]
        coder: Option<PathBuf>,
    },
}

#[derive(Args)]
struct Options {
    #[arg(long)]
    index: PathBuf,
    #[arg(long, value_enum, default_value = "auto")]
    mode: SolveMode,
    #[arg(long, default_value_t = 2000.0)]
    deadline_ms: f64,
    #[arg(long)]
    min_fov: Option<f64>,
    #[arg(long)]
    max_fov: Option<f64>,
    #[arg(long)]
    matches: bool,
    /// Allow an ARC projection for very wide frames.
    #[arg(long)]
    arc: bool,
    /// Opt in to STG and cylindrical panorama WCS models.
    #[arg(long)]
    panorama: bool,
}

impl Options {
    fn config(&self) -> SolveConfig {
        let defaults = SolveConfig::default();
        SolveConfig {
            mode: self.mode,
            deadline_ms: self.deadline_ms,
            min_fov_deg: self.min_fov.unwrap_or(defaults.min_fov_deg),
            max_fov_deg: self.max_fov.unwrap_or(defaults.max_fov_deg),
            keep_matches: self.matches,
            allow_arc: self.arc,
            allow_panorama: self.panorama,
            ..defaults
        }
    }
}

fn solve(engine: &Engine, path: &Path, cfg: &SolveConfig) -> Result<SolveOutput> {
    if path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
    {
        let field = serde_json::from_reader(File::open(path)?)
            .with_context(|| format!("Invalid StarField JSON: {}", path.display()))?;
        engine.solve_sources(&field, cfg)
    } else {
        engine.solve_path(path, cfg)
    }
}

fn run(cli: Cli) -> Result<u8> {
    let mut stdout = BufWriter::new(std::io::stdout().lock());
    let mut exit = 0;
    match cli.command {
        Command::Solve {
            input,
            options,
            wcs,
        } => {
            if let Some(path) = &wcs {
                ensure!(
                    !path.exists(),
                    "WCS output already exists: {}",
                    path.display()
                );
            }
            let engine = Engine::open(&options.index)?;
            let result = solve(&engine, &input, &options.config())?;
            if result.solution.status == "solved" {
                if let Some(path) = wcs {
                    let header = result
                        .solution
                        .wcs
                        .as_ref()
                        .context("Solved result has no WCS")?
                        .fits_header();
                    File::options()
                        .write(true)
                        .create_new(true)
                        .open(path)?
                        .write_all(header.as_bytes())?;
                }
            } else {
                exit = 1;
            }
            serde_json::to_writer_pretty(&mut stdout, &result)?;
            writeln!(stdout)?;
        }
        Command::Batch { inputs, options } => {
            let engine = Engine::open(&options.index)?;
            let cfg = options.config();
            for input in inputs {
                let row = match solve(&engine, &input, &cfg) {
                    Ok(result) => {
                        if result.solution.status != "solved" {
                            exit = exit.max(1);
                        }
                        serde_json::json!({"input": input, "result": result})
                    }
                    Err(error) => {
                        exit = 2;
                        serde_json::json!({"input": input, "error": format!("{error:#}")})
                    }
                };
                serde_json::to_writer(&mut stdout, &row)?;
                writeln!(stdout)?;
                stdout.flush()?;
            }
        }
        Command::Extract { input } => {
            let (gray, info) = image_io::load_with_info(&input)?;
            let field = skyshot::extract(&gray, &skyshot::ExtractOptions::for_source(&info));
            serde_json::to_writer_pretty(&mut stdout, &field)?;
            writeln!(stdout)?;
        }
        Command::Inspect { index } => {
            let index = skyshot::index::Index::open(&index)?;
            serde_json::to_writer_pretty(&mut stdout, &index.manifest)?;
            writeln!(stdout)?;
        }
        Command::Build {
            d80,
            out,
            profile,
            coder,
        } => {
            // Reserve the destination atomically. Never replace a published/mapped index.
            std::fs::create_dir(&out)
                .context("Use a new output directory with an existing parent")?;
            let config = builder::BuildConfig {
                d80,
                out,
                r7_arcmin: if profile == "deep" { 0.485 } else { 1.94 },
                max_pool_level: match profile.as_str() {
                    "lite" => 6,
                    "standard" => 7,
                    _ => 8,
                },
                dump_levels: false,
                profile,
                coder,
                ..Default::default()
            };
            let manifest = builder::build(&config)?;
            serde_json::to_writer_pretty(&mut stdout, &manifest)?;
            writeln!(stdout)?;
        }
    }
    stdout.flush()?;
    Ok(exit)
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            eprintln!("skyshot: {error:#}");
            ExitCode::from(2)
        }
    }
}
