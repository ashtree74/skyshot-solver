#[path = "../examples/support/mod.rs"]
mod support;
use skyshot::{Engine, solve::SolveConfig};
use std::{fs, process::Command};

#[test]
fn standalone_cli_and_reused_engine_solve_a_synthetic_sky() {
    let tmp = tempfile::tempdir().unwrap();
    let index = tmp.path().join("index");
    let (field, expected) = support::synthetic_catalog(&index).unwrap();
    let input = tmp.path().join("stars.json");
    fs::write(&input, serde_json::to_vec(&field).unwrap()).unwrap();
    let engine = Engine::open(&index).unwrap();
    let config = SolveConfig {
        deadline_ms: 10000.0,
        ..Default::default()
    };
    for _ in 0..2 {
        let result = engine.solve_sources(&field, &config).unwrap();
        assert_eq!(result.solution.status, "solved");
        assert!((result.solution.ra - expected.ra).abs() < 0.001);
        assert!((result.solution.dec - expected.dec).abs() < 0.001);
    }
    let wcs = tmp.path().join("solution.wcs");
    let output = Command::new(env!("CARGO_BIN_EXE_skyshot"))
        .arg("solve")
        .arg(&input)
        .arg("--index")
        .arg(&index)
        .args(["--deadline-ms", "10000", "--wcs"])
        .arg(&wcs)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["solution"]["status"], "solved");
    assert!(fs::read_to_string(&wcs).unwrap().contains("RA---TAN"));
    let before = fs::read(&wcs).unwrap();
    let again = Command::new(env!("CARGO_BIN_EXE_skyshot"))
        .arg("solve")
        .arg(&input)
        .arg("--index")
        .arg(&index)
        .arg("--wcs")
        .arg(&wcs)
        .output()
        .unwrap();
    assert_eq!(again.status.code(), Some(2));
    assert_eq!(fs::read(&wcs).unwrap(), before);
    let batch = Command::new(env!("CARGO_BIN_EXE_skyshot"))
        .arg("batch")
        .arg(&input)
        .arg(tmp.path().join("missing.fits"))
        .arg(&input)
        .arg("--index")
        .arg(&index)
        .args(["--deadline-ms", "10000"])
        .output()
        .unwrap();
    assert_eq!(batch.status.code(), Some(2));
    let rows: Vec<serde_json::Value> = String::from_utf8(batch.stdout)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["result"]["solution"]["status"], "solved");
    assert!(rows[1]["error"].is_string());
    assert_eq!(rows[2]["result"]["solution"]["status"], "solved");
    let mut empty = field.clone();
    empty.stars.clear();
    fs::write(&input, serde_json::to_vec(&empty).unwrap()).unwrap();
    let unsolved = Command::new(env!("CARGO_BIN_EXE_skyshot"))
        .arg("solve")
        .arg(&input)
        .arg("--index")
        .arg(&index)
        .output()
        .unwrap();
    assert_eq!(unsolved.status.code(), Some(1));
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&unsolved.stdout).unwrap()["solution"]["status"],
        "unsolved"
    );
    let mut bad = config.clone();
    bad.deadline_ms = f64::NAN;
    assert!(engine.solve_sources(&field, &bad).is_err());
}

#[test]
fn index_builder_never_replaces_an_existing_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let marker = tmp.path().join("untouched");
    fs::write(&marker, b"keep").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_skyshot"))
        .arg("build")
        .arg("--d80")
        .arg(tmp.path().join("missing"))
        .arg("--out")
        .arg(tmp.path())
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(fs::read(marker).unwrap(), b"keep");
}
