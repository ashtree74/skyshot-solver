#!/usr/bin/env python3
"""Verify and package a standalone LGQ-GD coder; Python 3.11+, standard library."""
import argparse
import hashlib
import re
import tarfile
from pathlib import Path

from data_bundle import digest, portable_info, read_json, require, validate_metadata, write_json

FILES = {"coder.f32", "coder.json"}
EVIDENCE = {"LICENSE-DATA.txt", "PROVENANCE.json", "MODEL-CARD.md", "INSTALL.md"}


def identity(root):
    for name in FILES:
        p = root / name
        require(p.is_file() and not p.is_symlink(), f"missing/linked model file: {name}")
    meta = read_json(root / "coder.json")
    require(meta["format"] == "skyanchor2-learned-coder-lgq" and meta["version"] == 1,
            "unsupported model format")
    h = hashlib.sha256()
    for name in ("coder.f32", "coder.json"):
        h.update((root / name).read_bytes())
    return {"format": meta["format"], "version": meta["version"], "key_type": "lgq-gd",
            "coder_sha256": h.hexdigest()}


def create(root, metadata, output):
    require(not output.resolve().is_relative_to(root.resolve()), "write manifest outside model")
    meta = read_json(metadata)
    validate_metadata(meta)
    write_json(output, {"schema_version": 1, "metadata": meta, "model": identity(root),
                        "files": [{"path": name, "bytes": (root / name).stat().st_size,
                                   "sha256": digest(root / name)} for name in sorted(FILES)]})


def verify(root, manifest, engine_version):
    bundle = read_json(manifest)
    require(bundle["schema_version"] == 1, "unsupported model bundle schema")
    validate_metadata(bundle["metadata"])
    require(engine_version in bundle["metadata"]["engine_versions"], "engine version not qualified")
    require(bundle["model"] == identity(root), "model identity mismatch")
    names = set()
    for item in bundle["files"]:
        name = item["path"]
        require(name in FILES and name not in names, "unsafe/duplicate model filename")
        require(type(item["bytes"]) is int and item["bytes"] >= 0, "invalid size")
        require(re.fullmatch(r"[a-f0-9]{64}", item["sha256"]), "invalid SHA256")
        names.add(name)
        require((root / name).stat().st_size == item["bytes"] and digest(root / name) == item["sha256"],
                f"model checksum mismatch: {name}")
    require(names == FILES, "incomplete model inventory")
    return bundle


def pack(root, manifest, engine_version, evidence, output):
    bundle = verify(root, manifest, engine_version)
    require(not output.resolve().is_relative_to(root.resolve()), "write archive outside model")
    for name in EVIDENCE:
        p = evidence / name
        require(p.is_file() and not p.is_symlink() and p.stat().st_size > 0, f"missing evidence: {name}")
    with output.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            prefix = bundle["metadata"]["bundle_id"]
            for name in sorted(FILES):
                archive.add(root / name, f"{prefix}/{name}", recursive=False, filter=portable_info)
            archive.add(manifest, f"{prefix}/bundle.json", recursive=False, filter=portable_info)
            for name in sorted(EVIDENCE):
                archive.add(evidence / name, f"{prefix}/{name}", recursive=False, filter=portable_info)
    verify(root, manifest, engine_version)
    output.with_name(output.name + ".sha256").write_text(f"{digest(output)}  {output.name}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("create", "verify", "pack"))
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--metadata", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--evidence", type=Path)
    p.add_argument("--engine-version", default="0.1.0")
    a = p.parse_args()
    try:
        if a.command == "create":
            require(a.metadata is not None and a.out is not None, "create needs --metadata and --out")
            create(a.model, a.metadata, a.out)
        else:
            require(a.manifest is not None, "--manifest required")
            if a.command == "verify":
                verify(a.model, a.manifest, a.engine_version)
            else:
                require(a.evidence is not None and a.out is not None, "pack needs --evidence and --out")
                pack(a.model, a.manifest, a.engine_version, a.evidence, a.out)
    except (ValueError, OSError, KeyError, TypeError) as error:
        p.exit(2, f"model_bundle: {error}\n")
    print(f"{a.command}: OK")


if __name__ == "__main__":
    main()
