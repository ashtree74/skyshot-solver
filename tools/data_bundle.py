#!/usr/bin/env python3
"""Create, verify and package immutable index bundles; Python 3.11+, no dependencies.

Never writes into the index directory. A checksum proves identity, not permission.
"""
import argparse
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path

if sys.version_info < (3, 11):
    raise SystemExit("data_bundle requires Python 3.11+ (check python3 --version)")

RUNTIME = {"manifest.json", "codes.bin", "keys.bin", "stars.bin", "cells.bin",
           "deep.bin", "deep_cells.bin", "coder.f32", "coder.json"}
BASE = {"manifest.json", "codes.bin", "keys.bin", "stars.bin", "cells.bin"}


def portable_info(info):
    """Do not export local account names, ownership, timestamps or extended headers."""
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o644
    info.pax_headers = {}
    return info


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def index_identity(root):
    m = read_json(root / "manifest.json")
    key = m.get("key_type", "v1")
    require(m["format"] == "skyanchor2-index", "unsupported index format")
    require(key in ("v1", "lgq-gd"), "unsupported key type")
    require(m["version"] in (1, 2, 3), "unsupported index version")
    require(m["version"] != 3 or m["profile"] == "deep", "v3 requires deep profile")
    require(m["version"] != 2 or key == "lgq-gd", "v2 requires learned key")
    present = {name for name in RUNTIME if (root / name).exists()}
    require(BASE <= present, "missing required index files")
    require(("deep.bin" in present) == ("deep_cells.bin" in present), "incomplete deep pair")
    for name in present:
        p = root / name
        require(p.is_file() and not p.is_symlink(), f"not a regular file: {name}")
    declared = {}
    for name, size in m["files"]:
        require(name in RUNTIME - {"manifest.json"}, f"unexpected runtime filename: {name}")
        require(name not in declared, f"duplicate index filename: {name}")
        require(type(size) is int and size >= 0, "invalid declared file size")
        declared[name] = size
        require(name in present and (root / name).stat().st_size == size,
                f"manifest size mismatch: {name}")
    require(BASE - {"manifest.json"} <= declared.keys(), "incomplete index file inventory")
    if key == "lgq-gd":
        require({"coder.f32", "coder.json"} <= present, "missing learned coder")
        h = hashlib.sha256()
        for name in ("coder.f32", "coder.json"):
            with (root / name).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
        require(h.hexdigest() == m.get("coder_sha256"), "coder identity mismatch")
    else:
        require(not ({"coder.f32", "coder.json"} & present), "coder files in a v1 index")
    return {"format": m["format"], "version": m["version"], "profile": m["profile"],
            "key_type": key, "coder_sha256": m.get("coder_sha256"),
            "catalog_fingerprint": m["catalog_fingerprint"],
            "r7_arcmin": m["r7_arcmin"]}, present


def validate_metadata(meta):
    require(re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,95}", meta["bundle_id"]), "invalid bundle_id")
    compat = meta["engine_versions"]
    require(isinstance(compat, list) and compat and all(
        isinstance(v, str) and re.fullmatch(r"\d+\.\d+\.\d+(?:-[a-zA-Z0-9.-]+)?", v)
        for v in compat), "engine_versions must be an exact nonempty version list")
    require(meta["redistribution"]["status"] in ("pending", "cleared"), "invalid terms inventory status")
    require(isinstance(meta["sources"], list) and meta["sources"], "source provenance required")
    require(isinstance(meta["build"], dict) and meta["build"], "build provenance required")


def create(root, metadata, output):
    require(not output.resolve().is_relative_to(root.resolve()), "write manifest outside index")
    meta = read_json(metadata)
    validate_metadata(meta)
    identity, names = index_identity(root)
    if "key_type" in meta["build"]:
        require(meta["build"]["key_type"] == identity["key_type"], "build key type mismatch")
    files = [{"path": name, "bytes": (root / name).stat().st_size,
              "sha256": digest(root / name)} for name in sorted(names)]
    write_json(output, {"schema_version": 1, "metadata": meta, "index": identity, "files": files})


def verify(root, manifest, engine_version):
    bundle = read_json(manifest)
    require(bundle["schema_version"] == 1, "unsupported bundle schema")
    meta = bundle["metadata"]
    validate_metadata(meta)
    require(engine_version in meta["engine_versions"], "engine version not qualified for bundle")
    identity, present = index_identity(root)
    require(identity == bundle["index"], "index identity mismatch")
    names = set()
    for item in bundle["files"]:
        name = item["path"]
        require(name in RUNTIME and name not in names, f"unsafe/duplicate filename: {name}")
        names.add(name)
        require(type(item["bytes"]) is int and item["bytes"] >= 0, "invalid file size")
        require(re.fullmatch(r"[a-f0-9]{64}", item["sha256"]), "invalid SHA256")
        path = root / name
        require(path.stat().st_size == item["bytes"], f"size mismatch: {name}")
        require(digest(path) == item["sha256"], f"SHA256 mismatch: {name}")
    require(names == present, "bundle must cover every runtime file")
    return bundle


def pack(root, manifest, engine_version, evidence, output):
    bundle = verify(root, manifest, engine_version)
    meta = bundle["metadata"]
    # Local packaging is not publication or a legal approval workflow. Preserve
    # the terms inventory verbatim, including unresolved entries, in bundle.json.
    required = {"LICENSE-DATA.txt", "PROVENANCE.json"}
    if bundle["index"]["key_type"] == "lgq-gd":
        required.add("MODEL-CARD.md")
    for name in required:
        p = evidence / name
        require(p.is_file() and not p.is_symlink() and p.stat().st_size > 0,
                f"missing release evidence: {name}")
    if (evidence / "INSTALL.md").is_file():
        require(not (evidence / "INSTALL.md").is_symlink(), "installation guide must be regular")
        required.add("INSTALL.md")
    require(not output.resolve().is_relative_to(root.resolve()), "write archive outside index")
    # Only selected files, never a recursive archive of the source checkout.
    with output.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            prefix = meta["bundle_id"]
            for item in bundle["files"]:
                archive.add(root / item["path"], f"{prefix}/{item['path']}", recursive=False, filter=portable_info)
            archive.add(manifest, f"{prefix}/bundle.json", recursive=False, filter=portable_info)
            for name in sorted(required):
                archive.add(evidence / name, f"{prefix}/{name}", recursive=False, filter=portable_info)
    # Catch accidental input changes during packaging before producing a checksum.
    verify(root, manifest, engine_version)
    with output.with_name(output.name + ".sha256").open("x", encoding="utf-8") as stream:
        stream.write(f"{digest(output)}  {output.name}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("create", "verify", "pack"):
        p = sub.add_parser(command)
        p.add_argument("--index", required=True, type=Path)
        if command == "create":
            p.add_argument("--metadata", required=True, type=Path)
            p.add_argument("--out", required=True, type=Path)
        else:
            p.add_argument("--manifest", required=True, type=Path)
            p.add_argument("--engine-version", required=True)
        if command == "pack":
            p.add_argument("--evidence", required=True, type=Path)
            p.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "create":
            create(args.index, args.metadata, args.out)
        elif args.command == "verify":
            verify(args.index, args.manifest, args.engine_version)
        else:
            pack(args.index, args.manifest, args.engine_version, args.evidence, args.out)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"data_bundle: {error}\n")
    print(f"{args.command}: OK (integrity is not a license or a solve qualification)")


if __name__ == "__main__":
    main()
