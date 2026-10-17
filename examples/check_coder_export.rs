//! Qualify a metadata-only coder export against its original, without changing either.
use anyhow::{Result, ensure};
use clap::Parser;
use sha2::{Digest, Sha256};
use skyshot::learned::{Coder, Raw};
use std::{io::Write, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    original: PathBuf,
    #[arg(long)]
    exported: PathBuf,
    #[arg(long)]
    out: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let original = Coder::load(&args.original)?;
    let exported = Coder::load(&args.exported)?;
    ensure!(original.bits() == exported.bits());
    let mut seed = 0x5041_5249_5459_u64;
    let mut rand = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / ((1_u64 << 53) as f64)
    };
    let mut signature = Sha256::new();
    let mut vectors = Vec::new();
    for i in 0..512 {
        let len = 1.0 + 500.0 * rand();
        let angle = rand() * std::f64::consts::TAU;
        let mut xy = [[0.0; 2]; 4];
        xy[0] = [len * angle.cos(), len * angle.sin()];
        for point in &mut xy[1..] {
            *point = [
                (0.01 + 4.0 * rand() - 2.0) * len,
                (0.01 + 4.0 * rand() - 2.0) * len,
            ];
        }
        let raw = Raw {
            xy,
            d: std::array::from_fn(|_| (0.01 + 3.0 * rand()) * len),
            sigma: if i % 2 == 0 { 0.0 } else { 0.01 + rand() },
            edge: (0.01 + 2.0 * rand()) * len,
        };
        let key = original.catalogue_key(&raw);
        ensure!(
            key == exported.catalogue_key(&raw),
            "catalogue key mismatch at {i}"
        );
        for invert in [false, true] {
            let a = original.image_code(&raw, invert).unwrap();
            let b = exported.image_code(&raw, invert).unwrap();
            ensure!(
                a.key == b.key && a.feats.flip == b.feats.flip,
                "image key mismatch at {i}"
            );
            for (aa, bb) in [
                (&a.e[..], &b.e[..]),
                (&a.sigma[..], &b.sigma[..]),
                (&a.logodds[..], &b.logodds[..]),
            ] {
                for (x, y) in aa.iter().zip(bb) {
                    ensure!(x.to_bits() == y.to_bits(), "f32 output mismatch at {i}");
                    signature.update(x.to_bits().to_le_bytes());
                }
            }
            for (x, y) in a.cdf.iter().zip(b.cdf) {
                ensure!(x.to_bits() == y.to_bits(), "CDF mismatch at {i}");
                signature.update(x.to_bits().to_le_bytes());
            }
            signature.update(a.key.to_le_bytes());
        }
        let (mut a, mut b) = (Vec::new(), Vec::new());
        let pa = original.image_probes(&raw, 8, &mut a);
        let pb = exported.image_probes(&raw, 8, &mut b);
        ensure!(pa == pb && a == b, "probe mismatch at {i}");
        for (x, y) in a.iter().zip(&b) {
            ensure!(x.cost.to_bits() == y.cost.to_bits());
            signature.update(x.cost.to_bits().to_le_bytes());
        }
        if i < 16 {
            vectors.push(serde_json::json!({"xy": raw.xy, "d": raw.d, "sigma": raw.sigma,
                "edge": raw.edge, "catalogue_key": key,
                "probes": a.iter().map(|p| serde_json::json!({"key": p.key, "flip": p.flip, "cost": p.cost})).collect::<Vec<_>>()}));
        }
    }
    let report = serde_json::json!({"passed": true, "original_coder_sha256": original.sha256,
        "coder_sha256": exported.sha256, "configurations": 512, "image_branches": 1024,
        "probe_sets": 512, "comparison": "bit-identical Rust catalogue/image keys, embedding/noise/CDF/logodds and probe costs",
        "output_signature_sha256": format!("{:x}", signature.finalize()), "vectors": vectors});
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(args.out)?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    writeln!(file)?;
    println!("Metadata-only export: all 512 configurations and 1024 image branches identical");
    Ok(())
}
