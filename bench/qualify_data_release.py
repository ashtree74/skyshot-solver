#!/usr/bin/env python3
"""Qualify staged or installed local bundles on an existing private reference plan.

Requires NumPy/Astropy as real_images_ab.py does. Private paths/raw solutions
remain in --out. Only the sanitized aggregate receipt is eligible for packaging.
"""
import argparse
import collections
import json
import os
import subprocess
import sys
from pathlib import Path

from real_images_ab import classify, read, sha, write

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import data_bundle as data
import model_bundle as model
from prepare_data_release import no_private_text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release-dir", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--installed", type=Path, help="parent containing complete/, handcrafted/, model/ presets")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    staged = read(a.release_dir / "STAGED.json")
    ids = staged["bundle_ids"]
    if a.installed:
        indexes = {"full-lgq": a.installed / "complete/index", "deep-lgq": a.installed / "complete/index-deep",
                   "wide-lgq": a.installed / "complete/index-phone", "full-v1": a.installed / "handcrafted/index-v1",
                   "model": a.installed / "model/model"}
    else:
        indexes = {kind: a.release_dir / "staging" / bundle_id for kind, bundle_id in ids.items()}
    plan = read(a.plan)
    plan["orders"] = []  # One pass per arm; qualification, not a new timing experiment.
    write(a.out / "plan.json", plan)
    frames = {f["id"]: f for f in plan["frames"]}
    truth = {k: read(Path(f["truth"])) if f.get("truth_source") else None for k, f in frames.items()}
    inventory = {f["id"]: {"image": sha(Path(f["image"])),
                            "reference": sha(Path(f["truth"])) if truth[f["id"]] else None} for f in frames.values()}
    assert all(v["image"] == frames[k]["input_sha256"] for k, v in inventory.items())
    expected = {}
    for line in a.baseline.read_text().splitlines():
        row = json.loads(line)
        if row["phase"] == "measured":
            expected[(row["arm"], row["id"])] = row
    env = dict(os.environ, RAYON_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1")
    binary = ROOT / "target/release/skyshot"
    driver = ROOT / "target/release/examples/benchmark_images"
    raw_path = a.out / "main.jsonl"
    with raw_path.open("x") as stream, (a.out / "driver.log").open("x") as log:
        subprocess.run([str(driver), "--manifest", str(a.out / "plan.json"),
                        "--learned", str(indexes["full-lgq"]), "--handcrafted", str(indexes["full-v1"]),
                        "--threads", "4"], stdout=stream, stderr=log, env=env, check=True)
    results = collections.defaultdict(list)
    unchanged_wcs = collections.Counter()
    for line in raw_path.read_text().splitlines():
        row = json.loads(line)
        assert "error" not in row, row
        ev = classify(row, frames[row["id"]], truth[row["id"]])
        results[row["arm"]].append(ev)
        old = expected[(row["arm"], row["id"])]
        assert row["solution"]["status"] == old["solution"]["status"], row["id"]
        assert row["solution"].get("wcs") == old["solution"].get("wcs"), row["id"]
        unchanged_wcs[row["arm"]] += 1
    print("Both full index copies: 90/90 statuses and WCS equal to original benchmark", flush=True)

    def cli_case(index, frame, label, mode="auto"):
        header = a.out / f"{label}.wcs"
        proc = subprocess.run([str(binary), "solve", frame["image"], "--index", str(index),
                               "--mode", mode, "--deadline-ms", "2000", "--wcs", str(header)],
                              text=True, capture_output=True, env=env)
        row = json.loads(proc.stdout)
        row.update(id=frame["id"], exit_code=proc.returncode)
        if header.exists():
            row["wcs_header"] = header.read_text()
        row["evaluation"] = classify(row, frame, truth[frame["id"]])
        assert proc.returncode in (0, 1), proc.stderr
        assert row["evaluation"]["classification"] in ("correct", "unsolved", "unverified_solved"), row["evaluation"]
        return row

    # Test Engine companion discovery with the ordinary CLI, including FITS hints
    # where the input contains them. Main-index comparison above is strictly blind.
    if a.installed:
        main = indexes["full-lgq"]
    else:
        combo = a.out / "companion-layout"
        combo.mkdir()
        for kind, slot in (("full-lgq", "index"), ("deep-lgq", "index-deep"), ("wide-lgq", "index-phone")):
            (combo / slot).symlink_to(indexes[kind].resolve(), target_is_directory=True)
        main = combo / "index"
    complete = []
    for n, frame in enumerate(frames.values()):
        complete.append(cli_case(main, frame, f"complete-{n:03d}"))
        if (n + 1) % 15 == 0:
            print(f"Complete preset CLI: {n + 1}/90", flush=True)
    write(a.out / "complete.json", complete)
    narrow = [frames[f"user-68760-job-{n}"] for n in (39, 40, 41)]
    deep = [cli_case(main, f, f"deep-{n}", "deep") for n, f in enumerate(narrow)]
    assert all(r["evaluation"]["classification"] == "correct" and r["solution"]["stats"]["deep_search"] for r in deep)
    write(a.out / "deep.json", deep)
    widest = sorted((f for f in frames.values() if truth[f["id"]]),
                    key=lambda f: truth[f["id"]]["fov_width_deg"], reverse=True)[:3]
    wide = [cli_case(indexes["wide-lgq"], f, f"wide-{n}", "standard") for n, f in enumerate(widest)]
    assert all(r["evaluation"]["classification"] == "correct" for r in wide)
    write(a.out / "wide.json", wide)
    for k, f in frames.items():
        assert sha(Path(f["image"])) == inventory[k]["image"]
        if truth[k]:
            assert sha(Path(f["truth"])) == inventory[k]["reference"]

    def summary(rows):
        return {"classifications": dict(collections.Counter(r["classification"] for r in rows)),
                "max_inner_error_px": max((r["grid_error_px"] for r in rows if r["grid_error_px"] is not None), default=None),
                "max_edge_error_px": max((r["edge_error_px"] for r in rows if r["edge_error_px"] is not None), default=None)}

    runtime = {}
    for kind, index in indexes.items():
        names = model.FILES if kind == "model" else data.index_identity(index)[1]
        runtime[ids[kind]] = {name: sha(index / name) for name in sorted(names)}
    parity = read(a.release_dir / "coder-parity.json")
    assert parity["passed"] and parity["coder_sha256"] == staged["coder_sha256"]
    receipt = {"passed": True, "author": "Adam Jesionkiewicz", "engine_version": "0.1.0",
               "platform": "macOS ARM64", "coder_sha256": staged["coder_sha256"],
               "runtime_sha256": runtime, "coder_parity": parity,
               "binary_sha256": sha(binary), "driver_sha256": sha(driver),
               "qualifier_sha256": sha(Path(__file__)), "scorer_sha256": sha(ROOT / "bench/real_images_ab.py"),
               "source_images_unchanged": True, "references_unchanged": True,
               "frames": len(frames), "reference_frames": sum(t is not None for t in truth.values()),
               "full_blind": {k: {**summary(v), "unchanged_status_and_wcs": unchanged_wcs[k]} for k, v in results.items()},
               "complete_cli_auto": {**summary([r["evaluation"] for r in complete]),
                                     "hints_used": sum(r["solution"]["stats"].get("hint_used", False) for r in complete),
                                     "deep_solutions": sum(r["solution"]["stats"].get("deep_search", False) and r["solution"]["status"] == "solved" for r in complete)},
               "deep_explicit": summary([r["evaluation"] for r in deep]),
               "wide_standalone": {**summary([r["evaluation"] for r in wide]),
                                   "reference_fov_width_deg": [truth[f["id"]]["fov_width_deg"] for f in widest]},
               "limits": ["Historical local positive-image corpus; repeated fields, not an independent holdout.",
                          "89 reference WCS, one not independently confirmed; one image has no reference.",
                          "No global false-positive rate inferred; no negative-control corpus.",
                          "Wide tested only on three existing images up to 12.52 degrees; broader wide/phone coverage is unqualified.",
                          "CLI auto/deep preserve input FITS hints where present; full A/B removes hints.",
                          "No source images or private paths included in this receipt."]}
    assert all(v["classification"] in ("correct", "unsolved") for rows in results.values() for v in rows)
    assert all(summary(rows)["classifications"].get("correct") == 86 for rows in results.values())
    no_private_text(receipt)
    write(a.out / "qualification.json", receipt)
    print(json.dumps({"passed": True, "full": receipt["full_blind"], "complete": receipt["complete_cli_auto"],
                      "deep": receipt["deep_explicit"], "wide": receipt["wide_standalone"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
