#!/usr/bin/env python3
"""Selectively stage and package local data candidates without modifying their source.

Stage first, qualify the new copies, then pack with the qualification receipt.
All generated data and archives belong outside Git. Python 3.11+, stdlib only.
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import data_bundle as data
import model_bundle as model

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = "Adam Jesionkiewicz"
PROFILES = {"full-lgq": ("index", "index"), "deep-lgq": ("index-deep", "index-deep"),
            "wide-lgq": ("index-phone", "index-phone"), "full-v1": ("index-v1", "index-v1")}
CODER_KEYS = {"format", "version", "dtype", "arch", "bits_per_dim", "tensors", "n_params", "variant"}
INDEX_KEYS = {"format", "version", "profile", "r7_arcmin", "key_spec", "levels", "n_records",
              "n_stars", "n_catalog", "star_cell_n", "catalog_fingerprint", "build_seconds",
              "files", "stop_words", "key_type", "coder_sha256"}


def no_private_text(value):
    text = json.dumps(value, ensure_ascii=False)
    data.require(not re.search(r"/Users/|/home/|/mnt/|[A-Za-z]:\\\\|(?:agents|claude|claud)\.md", text, re.I),
                 "private path or agent instructions in release metadata")


def copy_file(source, destination):
    data.require(source.is_file() and not source.is_symlink(), f"not a regular source file: {source.name}")
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)


def source_inventory(source):
    result = {}
    for name, _ in PROFILES.values():
        root = source / "work" / name
        _, files = data.index_identity(root)
        result[f"work/{name}"] = {n: {"bytes": (root / n).stat().st_size, "sha256": data.digest(root / n)}
                                  for n in sorted(files)}
    return result


def ids(version):
    return {**{k: f"skyshot-{k}-{version}" for k in PROFILES}, "model": f"skyshot-lgq-gd-{version}"}


def metadata(bundle_id, kind, source_files, coder_sha):
    return {"bundle_id": bundle_id, "name": f"Skyshot {kind}", "author": AUTHOR,
            "contributors": [{"name": AUTHOR, "roles": ["solver author", "model author", "bundle maintainer"]}],
            "engine_versions": ["0.1.0"], "intended_use": "non-commercial",
            "redistribution": {"status": "pending", "license": None,
                               "evidence": "Local preparation authorized by the project owner. External terms are inventoried in LICENSE-DATA.txt; publication is separate."},
            "sources": [{"name": "ASTAP D80 / Gaia DR3 / epoch 2025 / 1476-5",
                         "upstream": "https://www.hnsky.org/astap.htm",
                         "gaia": "https://www.cosmos.esa.int/web/gaia/dr3",
                         "input_inventory": "provenance/d80-local-inputs.json",
                         "input_inventory_sha256": data.digest(ROOT / "provenance/d80-local-inputs.json"),
                         "archive_sha256": None, "terms_evidence": "LICENSE-DATA.txt"}],
            "build": {"operation": "selective copy and metadata-only sanitized export; no index records recomputed",
                      "original_build_command": None, "original_build_revision": None,
                      "recipe": "tools/prepare_data_release.py stage --source SOURCE --out NEW_OUTPUT --version VERSION",
                      "recipe_sha256": data.digest(Path(__file__)), "source_files": source_files,
                      "coder_sha256": coder_sha, "qualification": "Awaiting new-copy qualification; must be supplied before pack."}}


LICENSE_DATA = """Skyshot data candidate — local preparation, no publication

Solver / LGQ-GD model / bundle preparation: Adam Jesionkiewicz.
The Apache-2.0 implementation license does not by itself license external data
or establish a license for these weight and catalogue distributions.

Catalogue source: ASTAP D80, 1476-5 tiles, Gaia DR3, epoch 2025.
ASTAP compilation: Han Kleijn / ASTAP project, https://www.hnsky.org/astap.htm
Gaia: ESA Gaia mission and the Gaia Data Processing and Analysis Consortium
(DPAC), https://www.cosmos.esa.int/web/gaia/dr3
This work has made use of data from the European Space Agency (ESA) mission
Gaia, processed by the Gaia Data Processing and Analysis Consortium (DPAC).
Funding for the DPAC has been provided by national institutions, in particular
the institutions participating in the Gaia Multilateral Agreement.

This candidate targets non-commercial data use as instructed by the owner.
That intended-use statement is not a substitute for upstream terms or a new
license grant. Exact downloaded D80 archive identity, included terms and complete
training/checkpoint lineage were not recovered in this extraction. Do not infer
the catalogue's license from the ASTAP executable license. Existing applicable
upstream conditions remain in force independently of the code license.

LGQ-GD weights are the author's existing export. Only descriptive metadata was
sanitized; numeric tensors and quantile tables are unchanged. Declared training
statistics are reported, not independently reproduced. Full training inputs,
checkpoint provenance and explicit weight-distribution terms remain to be
documented before publication. No third-party license has been invented or
replaced, and no legal clearance is claimed by the checksum tooling.
"""


def evidence(directory, meta, model_id, coder_sha, public_preview=False):
    directory.mkdir()
    terms = LICENSE_DATA
    if public_preview:
        terms = terms.replace("Skyshot data candidate — local preparation, no publication",
                              "Skyshot data — public preview, non-commercial intended use")
        terms = terms.replace("remain to be\ndocumented before publication", "remain to be\ndocumented")
        terms += "\nPublic availability was authorized by the project owner on 2026-10-17.\nThis authorization does not establish any missing upstream license grant.\n"
    (directory / "LICENSE-DATA.txt").write_text(terms)
    data.write_json(directory / "PROVENANCE.json", meta)
    (directory / "INSTALL.md").write_text("""# Install Skyshot data

Author and bundle maintainer: Adam Jesionkiewicz.

Current public installation guide: https://skyshot.app/downloads/
Use the release-level INSTALL.md and release.json with tools/install_data.py.
Verify SHA256SUMS first. Install into a NEW directory; do not merge an old index.
Every learned index already contains its exact coder.f32/coder.json pair.
The standalone model archive is needed only for building new learned indexes.

For standalone use after checking the archive and unpacked manifest:

```sh
skyshot inspect /path/to/THIS_UNPACKED_INDEX
skyshot solve your-image.fits --index /path/to/THIS_UNPACKED_INDEX --wcs new-solution.wcs
```

For the model bundle, pass its filename prefix (without .f32/.json):

```sh
skyshot build --d80 /path/to/d80 --out /path/to/NEW_INDEX --profile full --coder /path/to/THIS_MODEL/coder
```

The release installer uses index, index-deep and index-phone sibling names for
the qualified complete learned preset. Its optional index-v1 baseline is
installed alone so it does not silently use learned companions. Check status
and exit code; not every image is solvable. WCS output must be a new path.
""")
    if coder_sha:
        (directory / "MODEL-CARD.md").write_text(f"""# Skyshot LGQ-GD

Author: **{AUTHOR}**. Export: `{model_id}`.
Coder pair SHA256 (coder.f32 followed by coder.json): `{coder_sha}`.

This is a descriptive-metadata-only export of the existing LGQ-GD coder. The
weights and quantile tables are byte-identical to the original. Absolute
training paths were omitted using an operational-field allowlist. Author,
model name and export ID are embedded in coder.json. The new JSON changes the
pair digest; only requalified index copies name the new digest. Do not put this
JSON into an old index or mix coder pairs between builds.

Architecture: DeepSets, 16 element / 2 global / 3 noise features, hidden widths
64 and 128, embedding width 8. Quantization bits: 4/3/3/3/4/5/5/5 (32 total).
32,424 parameters plus 8 x 4,097 quantile values, float32 little endian.

Original metadata declares 3,210,419 training pairs, seed 0, 10,000 steps,
batch 8,192, learning rate 0.002, weight decay 0.0001, RTX 4090. Inspected
training code generates simulated catalogue-linked fields and pairs. These
statistics and complete checkpoint/shard ancestry were not reproduced here.
Catalogue source is ASTAP D80 / Gaia DR3 / epoch 2025; see PROVENANCE.json and
LICENSE-DATA.txt for source identity, attribution and remaining documentation.

Purpose: CPU retrieval of candidate star neighbourhoods for geometric
verification. This is not a direct coordinate predictor. Public training
reproduction and full input/split inventory are still pending. Do not assume
accuracy outside qualified images, field widths, noise and distortion regimes.
The model is not flight-qualified. Non-commercial data preparation is the
owner's present scope; explicit weight distribution terms remain inventoried
as pending. Apache-2.0 on solver code is separate.

Qualification of these exact new bytes, including catalogue/image keys and
probes, installed real-image solves and reference-WCS errors, is recorded in
the release QUALIFICATION.json. Private images and their paths are excluded.
""")


def stage(source, out, version, public_preview=False):
    data.require(re.fullmatch(r"\d{4}\.\d{2}\.\d{2}\.\d+", version), "version must be YYYY.MM.DD.N")
    data.require(not out.resolve().is_relative_to(source.resolve()), "output must be outside source")
    out.mkdir(parents=True, exist_ok=False)
    inventory = source_inventory(source)
    data.write_json(out / "SOURCE-INTEGRITY.json", inventory)
    bundle_ids = ids(version)
    staging = out / "staging"
    staging.mkdir()
    model_dir = staging / bundle_ids["model"]
    model_dir.mkdir()
    original = source / "work/index"
    old = data.read_json(original / "coder.json")
    clean = {key: old[key] for key in sorted(CODER_KEYS)}
    clean.update(author=AUTHOR, model_name="Skyshot LGQ-GD", export_id=bundle_ids["model"])
    no_private_text(clean)
    copy_file(original / "coder.f32", model_dir / "coder.f32")
    data.write_json(model_dir / "coder.json", clean)
    coder_sha = model.identity(model_dir)["coder_sha256"]
    model_meta = metadata(bundle_ids["model"], "LGQ-GD coder", inventory["work/index"], coder_sha)
    if public_preview:
        model_meta["availability"] = "public-preview"
        model_meta["redistribution"]["evidence"] = "Owner authorized public non-commercial data downloads on skyshot.app on 2026-10-17. Upstream terms and provenance gaps remain recorded in LICENSE-DATA.txt; no missing grant is inferred."
    data.write_json(out / f"{bundle_ids['model']}.metadata.json", model_meta)
    evidence(out / f"{bundle_ids['model']}.evidence", model_meta, bundle_ids["model"], coder_sha, public_preview)
    for kind, (name, _) in PROFILES.items():
        src = source / "work" / name
        dst = staging / bundle_ids[kind]
        dst.mkdir()
        identity, runtime = data.index_identity(src)
        learned = identity["key_type"] == "lgq-gd"
        if learned:
            data.require(identity["coder_sha256"] == data.index_identity(original)[0]["coder_sha256"],
                         "source coder differs; do not combine unqualified exports")
        for filename in sorted(runtime - {"manifest.json", "coder.json", "coder.f32"}):
            copy_file(src / filename, dst / filename)
        manifest = {key: value for key, value in data.read_json(src / "manifest.json").items() if key in INDEX_KEYS}
        manifest.update(catalog="ASTAP D80 / Gaia DR3 / epoch 2025", author=AUTHOR,
                        bundle_id=bundle_ids[kind])
        if learned:
            manifest["coder_sha256"] = coder_sha
            for filename in model.FILES:
                copy_file(model_dir / filename, dst / filename)
        no_private_text(manifest)
        data.write_json(dst / "manifest.json", manifest)
        data.index_identity(dst)
        meta = metadata(bundle_ids[kind], kind, inventory[f"work/{name}"], coder_sha if learned else None)
        if public_preview:
            meta["availability"] = "public-preview"
            meta["redistribution"]["evidence"] = model_meta["redistribution"]["evidence"]
        meta["build"]["key_type"] = identity["key_type"]
        data.write_json(out / f"{bundle_ids[kind]}.metadata.json", meta)
        evidence(out / f"{bundle_ids[kind]}.evidence", meta, bundle_ids["model"], coder_sha if learned else None, public_preview)
    data.require(inventory == source_inventory(source), "source changed during staging")
    data.write_json(out / "STAGED.json", {"version": version, "bundle_ids": bundle_ids,
                                          "coder_sha256": coder_sha, "author": AUTHOR,
                                          "public_preview": public_preview})
    print(f"Staged {len(PROFILES)} indexes and coder; source hashes unchanged", flush=True)


def pack(source, out, qualification):
    staged = data.read_json(out / "STAGED.json")
    inventory = data.read_json(out / "SOURCE-INTEGRITY.json")
    data.require(source_inventory(source) == inventory, "source changed since staging")
    q = data.read_json(qualification)
    data.require(q.get("passed") is True and q.get("coder_sha256") == staged["coder_sha256"],
                 "passing qualification for the exact coder required")
    no_private_text(q)
    data.write_json(out / "QUALIFICATION.json", q)
    artifacts = []
    for kind, bundle_id in staged["bundle_ids"].items():
        meta_path = out / f"{bundle_id}.metadata.json"
        meta = data.read_json(meta_path)
        meta["build"]["qualification"] = {"report": "QUALIFICATION.json", "sha256": data.digest(out / "QUALIFICATION.json")}
        no_private_text(meta)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        ev = out / f"{bundle_id}.evidence"
        (ev / "PROVENANCE.json").write_text(json.dumps(meta, indent=2) + "\n")
        root = out / "staging" / bundle_id
        manifest = out / f"{bundle_id}.bundle.json"
        module = model if kind == "model" else data
        module.create(root, meta_path, manifest)
        # Qualification lists all runtime hashes; refuse changed stage bytes.
        expected = {f["path"]: f["sha256"] for f in data.read_json(manifest)["files"]}
        data.require(expected == q["runtime_sha256"][bundle_id], "staged files changed after qualification")
        archive = out / f"{bundle_id}.tar.gz"
        print(f"Packing {bundle_id}", flush=True)
        module.pack(root, manifest, "0.1.0", ev, archive)
        artifacts.append({"bundle_id": bundle_id, "kind": "model" if kind == "model" else "index",
                          "variant": kind, "archive": archive.name, "bytes": archive.stat().st_size,
                          "sha256": data.digest(archive), "manifest": manifest.name,
                          "manifest_sha256": data.digest(manifest),
                          "coder_sha256": staged["coder_sha256"] if kind != "full-v1" else None})
    b = staged["bundle_ids"]
    release = {"schema_version": 1, "release_id": f"skyshot-data-{staged['version']}", "author": AUTHOR,
               "status": "public-preview" if staged.get("public_preview") else "local-candidate-unpublished",
               "engine_versions": ["0.1.0"], "artifacts": artifacts,
               "presets": {"full": {"index": b["full-lgq"]},
                           "full-deep": {"index": b["full-lgq"], "index-deep": b["deep-lgq"]},
                           "complete": {"index": b["full-lgq"], "index-deep": b["deep-lgq"], "index-phone": b["wide-lgq"]},
                           "handcrafted": {"index-v1": b["full-v1"]}, "model": {"model": b["model"]}},
               "qualification": {"file": "QUALIFICATION.json", "sha256": data.digest(out / "QUALIFICATION.json")}}
    if staged.get("public_preview"):
        release["base_url"] = f"https://skyshot.app/downloads/releases/{staged['version']}/"
    no_private_text(release)
    data.write_json(out / "release.json", release)
    (out / "tools").mkdir()
    for name in ("install_data.py", "data_bundle.py", "model_bundle.py", "archive_utils.py"):
        copy_file(ROOT / "tools" / name, out / "tools" / name)
    copy_file(ROOT / "docs/DATA-INSTALL.md", out / "INSTALL.md")
    data.require(source_inventory(source) == inventory, "source changed during packaging")
    data.write_json(out / "SOURCE-UNCHANGED.json", {"passed": True, "files": inventory})
    checksummed = [out / "release.json", out / "QUALIFICATION.json", out / "INSTALL.md",
                   *sorted(out.glob("*.tar.gz")), *sorted(out.glob("*.bundle.json")), *sorted((out / "tools").glob("*.py"))]
    with (out / "SHA256SUMS").open("x") as stream:
        for path in checksummed:
            stream.write(f"{data.digest(path)}  {path.relative_to(out).as_posix()}\n")
    print("Local candidates complete; no publication performed", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("stage", "pack"))
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--version", default="2026.10.17.1")
    p.add_argument("--qualification", type=Path)
    p.add_argument("--public-preview", action="store_true", help="record owner-authorized public preview availability; does not clear upstream terms")
    a = p.parse_args()
    try:
        if a.command == "stage":
            stage(a.source, a.out, a.version, a.public_preview)
        else:
            data.require(a.qualification is not None, "--qualification required")
            pack(a.source, a.out, a.qualification)
    except (ValueError, OSError, KeyError, TypeError) as error:
        p.exit(2, f"prepare_data_release: {error}\n")


if __name__ == "__main__":
    main()
