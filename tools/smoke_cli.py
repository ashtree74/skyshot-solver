#!/usr/bin/env python3
"""Exercise an unpacked candidate against synthetic_demo output (no catalogue)."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def smoke(binary, demo):
    binary = str(binary.resolve())
    demo = demo.resolve()
    expected = json.loads((demo / "expected-wcs.json").read_text())
    with tempfile.TemporaryDirectory(prefix="skyshot-smoke-") as tmp:
        wcs = Path(tmp) / "solution.wcs"
        args = [binary, "solve", str(demo / "stars.json"), "--index", str(demo / "index"),
                "--deadline-ms", "10000", "--wcs", str(wcs)]
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        row = json.loads(result.stdout)
        assert row["schema_version"] == 1 and row["solution"]["status"] == "solved"
        for coordinate in ("ra", "dec"):
            assert abs(row["solution"][coordinate] - expected[coordinate]) < 0.001
        before = wcs.read_bytes()
        assert b"RA---TAN" in before
        assert subprocess.run(args, capture_output=True).returncode == 2
        assert wcs.read_bytes() == before
        inspect = subprocess.run([binary, "inspect", str(demo / "index")],
                                 capture_output=True, text=True, check=True)
        manifest = json.loads(inspect.stdout)
        assert manifest["catalog_fingerprint"].startswith("synthetic-demo-v1;")
        assert manifest["n_records"] > 0
        batch = subprocess.run([binary, "batch", str(demo / "stars.json"), str(Path(tmp) / "missing.fits"),
                                "--index", str(demo / "index"), "--deadline-ms", "10000"],
                               capture_output=True, text=True)
        assert batch.returncode == 2
        rows = [json.loads(line) for line in batch.stdout.splitlines()]
        assert len(rows) == 2 and rows[0]["result"]["solution"]["status"] == "solved"
        assert "error" in rows[1]
    print("Packaged CLI smoke: solved at expected coordinates; WCS, inspect, batch and exit codes OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    args = parser.parse_args()
    smoke(args.binary, args.demo)
