#!/usr/bin/env python3
"""Local, read-only real-image A/B; requires numpy, astropy and psutil in Python 3.11+.

Build: cargo build --release --locked --example benchmark_images
Run: PYTHONDONTWRITEBYTECODE=1 PYTHON bench/real_images_ab.py --source SOURCE --out NEW_OUTPUT
Source images, references and indexes are opened only for reading. Results stay in --out.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import threading
import time
from pathlib import Path

import astropy
import numpy as np
import psutil
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("learned", "handcrafted")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pct(values, q):
    return float(np.percentile(values, q)) if values else None


def truth_wcs(truth):
    w = truth["wcs"]
    header = {"NAXIS": 2, "CTYPE1": w.get("ctype", ["RA---TAN", "DEC--TAN"])[0],
              "CTYPE2": w.get("ctype", ["RA---TAN", "DEC--TAN"])[1],
              "CRVAL1": w["crval1"], "CRVAL2": w["crval2"],
              "CRPIX1": w["crpix1"], "CRPIX2": w["crpix2"],
              "CD1_1": w["cd1_1"], "CD1_2": w["cd1_2"],
              "CD2_1": w["cd2_1"], "CD2_2": w["cd2_2"]}
    if w.get("sip"):
        for name in ("a", "b", "ap", "bp"):
            order = w["sip"].get(name + "_order")
            if order is not None:
                header[name.upper() + "_ORDER"] = order
                for key, value in w["sip"].get(name, {}).items():
                    header[name.upper() + "_" + key] = value
    return WCS(header, relax=True)


def classify(row, frame, truth):
    solution = row.get("solution", {})
    status = solution.get("status", "error")
    result = {"status": status, "classification": status, "grid_error_px": None,
              "edge_error_px": None, "center_error_px": None}
    if status != "solved":
        return result
    if truth is None:
        result["classification"] = "unverified_solved"
        return result
    try:
        header = row["wcs_header"]
        solved = WCS(fits.Header.fromstring(header, sep="\n" if "\n" in header else ""), relax=True)
        reference = truth_wcs(truth)
        width, height = truth["width"], truth["height"]

        def error(fractions):
            x, y = np.meshgrid(np.array(fractions) * (width - 1), np.array(fractions) * (height - 1))
            xy = np.column_stack([x.ravel(), y.ravel()])
            sky = solved.all_pix2world(xy, 0)
            projected = reference.all_world2pix(sky, 0, tolerance=1e-8, maxiter=100)
            return np.linalg.norm(projected - xy, axis=1)

        inner = error([0.1, 0.5, 0.9])
        edge = error([0.0, 0.5, 1.0])
        if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(edge)):
            raise ValueError("non-finite WCS comparison")
        result.update(grid_error_px=float(max(inner)), edge_error_px=float(max(edge)),
                      center_error_px=float(inner[4]))
        # Gross identification checks are separate from the 3 px WCS accuracy gate.
        a = np.asarray(solved.pixel_scale_matrix)
        b = np.asarray(reference.pixel_scale_matrix)
        scale_ratio = math.sqrt(abs(np.linalg.det(a) / np.linalg.det(b)))
        parity_ok = np.linalg.det(a) * np.linalg.det(b) > 0
        rotation = math.degrees(math.atan2(np.linalg.det(np.column_stack([a[:, 0], b[:, 0]])),
                                         np.dot(a[:, 0], b[:, 0])))
        result.update(scale_ratio=scale_ratio, rotation_difference_deg=rotation, parity_ok=bool(parity_ok))
        identified = inner[4] <= 50 and parity_ok and abs(scale_ratio - 1) <= 0.10 and abs(rotation) <= 5
        result["classification"] = ("wrong_field" if not identified else
                                    "correct" if max(inner) <= 3 else "inaccurate_wcs")
    except Exception as error:
        # Never count an unscorable solved result as correct or as an unsolved image.
        result.update(classification="invalid_solution", scoring_error=str(error))
    return result


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = args.source / "work/truth/manifest.json"
    entries = read(manifest)
    frames = []
    seen = {}
    excluded = []
    for entry in entries:
        if entry.get("group") == "synthetic":
            excluded.append({"id": entry["id"], "reason": "synthetic"})
            continue
        image = Path(entry["image"])
        digest = sha(image)
        if digest != entry["sha256"]:
            raise ValueError(f"image changed since reference: {entry['id']}")
        if digest in seen:
            excluded.append({"id": entry["id"], "reason": "exact duplicate", "of": seen[digest]})
            continue
        seen[digest] = entry["id"]
        frame = dict(entry, image=str(image.resolve()), input_sha256=digest)
        truth = Path(entry["truth"]) if entry.get("truth_source") else None
        frame["truth_sha256"] = sha(truth) if truth else None
        frames.append(frame)
    rng = random.Random(args.seed)
    orders = []
    for _ in range(args.repeats):
        order = list(range(len(frames)))
        rng.shuffle(order)
        orders.append(order)
    write(args.out / "plan.json", {"frames": frames, "orders": orders})
    indexes = {"learned": args.source / "work/index", "handcrafted": args.source / "work/index-v1"}
    meta = {arm: read(p / "manifest.json") for arm, p in indexes.items()}
    for key in ("profile", "r7_arcmin", "n_stars", "n_records", "n_catalog", "catalog_fingerprint", "levels"):
        if meta["learned"][key] != meta["handcrafted"][key]:
            raise ValueError(f"unmatched index setting: {key}")
    if meta["learned"].get("key_type") != "lgq-gd" or meta["handcrafted"].get("key_type", "v1") != "v1":
        raise ValueError("unexpected key types")
    inventory = {}
    for arm, index in indexes.items():
        names = {"manifest.json", *(name for name, _ in meta[arm]["files"])}
        if arm == "learned":
            names |= {"coder.json", "coder.f32"}
        inventory[arm] = {"path": str(index.resolve()), "manifest": meta[arm],
                          "files": {name: sha(index / name) for name in sorted(names)}}
    for name in ("stars.bin", "cells.bin", "deep.bin", "deep_cells.bin"):
        if inventory["learned"]["files"][name] != inventory["handcrafted"]["files"][name]:
            raise ValueError(f"catalogue geometry differs: {name}")
    provenance = {"source_manifest": str(manifest.resolve()), "source_manifest_sha256": sha(manifest),
                  "source_verification_sha256": sha(args.source / "work/truth/verification.json"),
                  "frames": len(frames), "excluded": excluded, "indexes": inventory,
                  "binary_sha256": sha(args.binary), "driver_source_sha256": sha(ROOT / "examples/benchmark_images.rs"),
                  "scorer_source_sha256": sha(Path(__file__)), "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
                  "platform": platform.platform(), "cpu_count": os.cpu_count(), "ram_bytes": psutil.virtual_memory().total,
                  "python": platform.python_version(), "numpy": np.__version__, "astropy": astropy.__version__,
                  "threads": args.threads, "repeats": args.repeats, "seed": args.seed,
                  "deadline_ms": args.deadline_ms, "mode": "standard", "hints": False,
                  "wide_companions": False, "deep_companions": False, "arc": False, "panorama": False,
                  "warm_timing": "per-image median over all measured repeats, after one untimed full-corpus warmup",
                  "fresh_timing": "new process per image and arm; OS file cache warm, not a disk-cold measurement"}
    write(args.out / "provenance.json", provenance)
    print(f"Prepared {len(frames)} real images; {sum(bool(f.get('truth_source')) for f in frames)} with reference WCS", flush=True)
    return frames, indexes, provenance


def measure(args, frames, indexes):
    stop = threading.Event()
    samples = []
    active = [None]

    def monitor():
        previous = psutil.cpu_times()
        previous_t = time.monotonic()
        previous_child = 0.0
        previous_pid = None
        while not stop.wait(1.0):
            now, ticks = time.monotonic(), psutil.cpu_times()
            busy = sum(getattr(ticks, k, 0) - getattr(previous, k, 0) for k in ("user", "system", "nice"))
            process = active[0]
            child_cpu = 0.0
            try:
                usage = psutil.Process(process.pid).cpu_times() if process else None
                current = usage.user + usage.system if usage else 0.0
                if process and process.pid == previous_pid:
                    child_cpu = max(0, current - previous_child)
                previous_child, previous_pid = current, process.pid if process else None
            except psutil.Error:
                pass
            dt = now - previous_t
            samples.append({"elapsed_s": now - started, "machine_busy_cores": busy / dt,
                            "benchmark_busy_cores": child_cpu / dt,
                            "estimated_background_cores": max(0, (busy - child_cpu) / dt),
                            "load_average": list(os.getloadavg())})
            previous, previous_t = ticks, now

    started = time.monotonic()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    base = [str(args.binary.resolve()), "--manifest", str((args.out / "plan.json").resolve()),
            "--threads", str(args.threads), "--deadline-ms", str(args.deadline_ms)]
    try:
        with (args.out / "paired.jsonl").open("x") as out, (args.out / "driver.log").open("x") as log:
            cmd = base + ["--learned", str(indexes["learned"].resolve()),
                          "--handcrafted", str(indexes["handcrafted"].resolve())]
            active[0] = subprocess.Popen(cmd, stdout=out, stderr=log)
            active[0].wait(timeout=(len(frames) * 2 * (args.repeats + 1) * (args.deadline_ms / 1000 + 2)) + 60)
            if active[0].returncode:
                raise RuntimeError(f"paired benchmark failed: {active[0].returncode}; see driver.log")
            active[0] = None
        print("Paired measurements complete; measuring fresh-process latency", flush=True)
        with (args.out / "fresh.jsonl").open("x") as out:
            for i, frame in enumerate(frames):
                for arm in (ARMS if i % 2 == 0 else ARMS[::-1]):
                    cmd = base + ["--index", str(indexes[arm].resolve()), "--only-frame", frame["id"]]
                    begin = time.perf_counter()
                    active[0] = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    stdout, stderr = active[0].communicate(timeout=args.deadline_ms / 1000 + 20)
                    wall = (time.perf_counter() - begin) * 1000
                    if active[0].returncode:
                        raise RuntimeError(f"fresh run failed: {frame['id']}, {arm}: {stderr}")
                    active[0] = None
                    row = json.loads(stdout)
                    row.update(arm=arm, process_wall_ms=wall)
                    out.write(json.dumps(row, allow_nan=False) + "\n")
                    out.flush()
                if (i + 1) % 10 == 0:
                    print(f"Fresh-process measurements: {i + 1}/{len(frames)} images", flush=True)
    finally:
        if active[0] is not None and active[0].poll() is None:
            active[0].kill()
            active[0].wait()
        stop.set()
        thread.join()
        write(args.out / "load.json", samples)


def report(args, frames, provenance):
    by_id = {f["id"]: f for f in frames}
    truths = {f["id"]: read(Path(f["truth"])) if f.get("truth_source") else None for f in frames}
    measured = []
    warmup = []
    for line in (args.out / "paired.jsonl").read_text().splitlines():
        row = json.loads(line)
        (warmup if row["phase"] == "warmup" else measured).append(row)
    if len(measured) != len(frames) * len(ARMS) * args.repeats:
        raise ValueError("incomplete paired measurements")
    fresh = [json.loads(line) for line in (args.out / "fresh.jsonl").read_text().splitlines()]
    if len(fresh) != len(frames) * len(ARMS):
        raise ValueError("incomplete fresh measurements")
    grouped = collections.defaultdict(list)
    for row in measured + fresh:
        score = classify(row, by_id[row["id"]], truths[row["id"]])
        row["evaluation"] = score
        if row["phase"] == "measured":
            grouped[(row["arm"], row["id"])].append(row)
    with (args.out / "scored.jsonl").open("x") as out:
        for row in measured + fresh:
            out.write(json.dumps(row, allow_nan=False) + "\n")
    per_image = []
    for arm in ARMS:
        for frame in frames:
            rows = grouped[(arm, frame["id"])]
            if len(rows) != args.repeats or len({r["round"] for r in rows}) != args.repeats:
                raise ValueError("missing/duplicate repeated measurement")
            kinds = [r["evaluation"]["classification"] for r in rows]
            summary = {"id": frame["id"], "arm": arm, "group": frame["group"],
                       "has_reference": truths[frame["id"]] is not None,
                       "confirmed_reference": bool(frame.get("confirmed")), "classes": dict(collections.Counter(kinds)),
                       "consistent": len(set(kinds)) == 1, "all_correct": all(k == "correct" for k in kinds),
                       "all_identified": all(k in ("correct", "inaccurate_wcs") for k in kinds),
                       "any_wrong_field": "wrong_field" in kinds, "any_invalid_solution": "invalid_solution" in kinds,
                       "all_solved": all(r["evaluation"]["status"] == "solved" for r in rows),
                       "total_ms": statistics.median(r["timings"]["total_ms"] for r in rows) if all("timings" in r for r in rows) else None,
                       "solver_ms": statistics.median(r["timings"]["solve_ms"] for r in rows) if all("timings" in r for r in rows) else None,
                       "max_grid_error_px": max((r["evaluation"]["grid_error_px"] for r in rows if r["evaluation"]["grid_error_px"] is not None), default=None),
                       "max_edge_error_px": max((r["evaluation"]["edge_error_px"] for r in rows if r["evaluation"]["edge_error_px"] is not None), default=None)}
            per_image.append(summary)
    common = {f["id"] for f in frames if all(next(r for r in per_image if r["arm"] == a and r["id"] == f["id"])["all_correct"] for a in ARMS)}
    summaries = {}
    group_summaries = {}
    for arm in ARMS:
        rows = [r for r in per_image if r["arm"] == arm]
        scored = [r for r in rows if r["has_reference"]]
        common_rows = [r for r in rows if r["id"] in common]
        ft = [r["process_wall_ms"] for r in fresh if r["arm"] == arm]
        summaries[arm] = {"images": len(rows), "reference_images": len(scored),
                          "solved_all_repeats": sum(r["all_solved"] for r in rows),
                          "identified_all_repeats": sum(r["all_identified"] for r in scored),
                          "correct_all_repeats": sum(r["all_correct"] for r in scored),
                          "wrong_field_any_repeat": sum(r["any_wrong_field"] for r in scored),
                          "invalid_solution_any_repeat": sum(r["any_invalid_solution"] for r in scored),
                          "inaccurate_wcs_images": sum("inaccurate_wcs" in r["classes"] for r in scored),
                          "unstable_classification": sum(not r["consistent"] for r in rows),
                          "classifications": dict(sum((collections.Counter(r["classes"]) for r in rows), collections.Counter())),
                          "warm_all_total_p50_ms": pct([r["total_ms"] for r in rows if r["total_ms"] is not None], 50),
                          "warm_all_total_p95_ms": pct([r["total_ms"] for r in rows if r["total_ms"] is not None], 95),
                          "warm_common_total_p50_ms": pct([r["total_ms"] for r in common_rows], 50),
                          "warm_common_total_p95_ms": pct([r["total_ms"] for r in common_rows], 95),
                          "warm_common_solver_p50_ms": pct([r["solver_ms"] for r in common_rows], 50),
                          "warm_common_solver_p95_ms": pct([r["solver_ms"] for r in common_rows], 95),
                          "fresh_process_all_p50_ms": pct(ft, 50), "fresh_process_all_p95_ms": pct(ft, 95)}
        group_summaries[arm] = {}
        for group in sorted({r["group"] for r in rows}):
            rr = [r for r in rows if r["group"] == group]
            group_summaries[arm][group] = {"images": len(rr), "reference_images": sum(r["has_reference"] for r in rr),
                                          "correct": sum(r["all_correct"] for r in rr),
                                          "identified": sum(r["all_identified"] for r in rr),
                                          "wrong_field": sum(r["any_wrong_field"] for r in rr),
                                          "inaccurate_wcs": sum("inaccurate_wcs" in r["classes"] for r in rr)}
    load = read(args.out / "load.json")
    result = {"arms": summaries, "groups": group_summaries, "common_correct_images": len(common),
              "common_correct_ids": sorted(common),
              "background_busy_cores_p50": pct([r["estimated_background_cores"] for r in load], 50),
              "background_busy_cores_p95": pct([r["estimated_background_cores"] for r in load], 95),
              "paired_common_speedup_median": statistics.median(
                  next(r["solver_ms"] for r in per_image if r["arm"] == "handcrafted" and r["id"] == fid) /
                  next(r["solver_ms"] for r in per_image if r["arm"] == "learned" and r["id"] == fid)
                  for fid in common) if common else None}
    write(args.out / "summary.json", result)
    write(args.out / "per-image.json", per_image)
    with (args.out / "per-image.csv").open("x", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    lines = ["# Local real-sky A/B: learned vs handcrafted", "",
             f"{len(frames)} real images; {sum(t is not None for t in truths.values())} reference WCS; seven synthetic images excluded.",
             f"{args.repeats} measured repeats per image/arm after one full untimed warmup; {args.threads} Rayon threads; {args.deadline_ms:g} ms solve budget.",
             "Blind in-memory decoding: FITS position/scale hints removed; original pixels unchanged; full main indexes only, no DEEP/wide companions.",
             "Matched catalogue/levels/record counts; the same stars/cell/deep-star bytes. Independent Astropy/WCSLIB scoring against pre-existing WCS references.", "",
             "| Arm | Solved / all | Identified / reference | Accurate ≤3 px / reference | Wrong field (any repeat) | Inaccurate WCS | Unstable |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, s in summaries.items():
        lines.append(f"| {arm} | {s['solved_all_repeats']}/{s['images']} | {s['identified_all_repeats']}/{s['reference_images']} | {s['correct_all_repeats']}/{s['reference_images']} | {s['wrong_field_any_repeat']} | {s['inaccurate_wcs_images']} | {s['unstable_classification']} |")
    lines += ["", "Accuracy: maximum reference-pixel disagreement on a 3×3 grid at 10/50/90% of width/height, ≤3 px in every repeat.",
              "Identification: center ≤50 px, matching parity, scale within 10%, local CD orientation within 5°. Identified-but-inaccurate is separate from a wrong field.",
              "Outer-edge 0/50/100% grid errors are retained separately per image. Solved results without a reference remain unverified.", "",
              f"Timings below use a per-image median, then p50/p95 across images; never the minimum of repeats. Common accurate subset: {len(common)} images.", "",
              "| Arm | Warm all total p50 / p95 ms | Warm common total p50 / p95 ms | Warm common solver p50 / p95 ms | Fresh process all p50 / p95 ms |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for arm, s in summaries.items():
        pair = lambda a, b: f"{s[a]:.2f} / {s[b]:.2f}" if s[a] is not None else "n/a"
        lines.append(f"| {arm} | {pair('warm_all_total_p50_ms', 'warm_all_total_p95_ms')} | {pair('warm_common_total_p50_ms', 'warm_common_total_p95_ms')} | {pair('warm_common_solver_p50_ms', 'warm_common_solver_p95_ms')} | {pair('fresh_process_all_p50_ms', 'fresh_process_all_p95_ms')} |")
    lines += ["", "Warm total covers decode/detect/solve; solver includes search/refinement and retries, excluding decode/extraction. Index mapping is excluded from warm timings.",
              "Fresh process covers process startup, index mapping, decode/detect/solve and JSON output; OS file cache is warm. This is not a disk-cold benchmark.",
              f"Shared machine: estimated background busy cores p50={result['background_busy_cores_p50']:.2f}, p95={result['background_busy_cores_p95']:.2f}; paired arm order alternated, input order shuffled per round.", "",
              "| Group | Images / references | Learned identified / accurate | Handcrafted identified / accurate |",
              "| --- | ---: | ---: | ---: |"]
    for group in group_summaries["learned"]:
        a, b = group_summaries["learned"][group], group_summaries["handcrafted"][group]
        lines.append(f"| {group} | {a['images']} / {a['reference_images']} | {a['identified']} / {a['correct']} | {b['identified']} / {b['correct']} |")
    lines += ["", "Limitations: real corpus includes repeated M66/NGC6888 fields; image counts are not independent sky-field counts.",
              "References share Gaia/D80 and historical detector/refit ancestry; they are not absolute ground truth from an independent catalogue. One referenced user frame is not independently confirmed; one real frame has no reference.",
              "This corpus was present in historical research; it is not a newly withheld generalization set. No negative controls are included, so no global false-positive probability is inferred.",
              "No image or index binaries were copied; original image, metadata/reference and index files were not modified. The input plan retains evaluation metadata locally. Raw results and report stay local; nothing was published.", "",
              "See provenance.json for exact binary/source/data hashes and settings, per-image.csv for results, paired.jsonl/fresh.jsonl/scored.jsonl for raw evidence."]
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def verify_unchanged(args, frames, provenance):
    for frame in frames:
        if sha(Path(frame["image"])) != frame["input_sha256"]:
            raise ValueError(f"input changed during benchmark: {frame['id']}")
        if frame.get("truth_source") and sha(Path(frame["truth"])) != frame["truth_sha256"]:
            raise ValueError(f"reference changed during benchmark: {frame['id']}")
    for record in provenance["indexes"].values():
        for name, expected in record["files"].items():
            if sha(Path(record["path"]) / name) != expected:
                raise ValueError(f"index changed during benchmark: {name}")
    write(args.out / "integrity-after.json", {"all_image_reference_and_index_hashes_unchanged": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/examples/benchmark_images")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--deadline-ms", type=float, default=2000)
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.out = args.out.resolve()
    if args.out.is_relative_to(args.source) or not args.out.is_relative_to(ROOT / "outputs"):
        raise ValueError("output must be a new directory under this checkout's ignored outputs/")
    if args.repeats < 1 or args.threads < 1:
        raise ValueError("positive repeats/threads required")
    frames, indexes, provenance = prepare(args)
    measure(args, frames, indexes)
    verify_unchanged(args, frames, provenance)
    report(args, frames, provenance)


if __name__ == "__main__":
    main()
