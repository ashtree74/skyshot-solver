#!/usr/bin/env python3
"""Install a selected data preset from local, checksummed archives into a NEW directory."""
import argparse
import re
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import data_bundle
import model_bundle
from archive_utils import safe_relative, unpack_tar
from data_bundle import digest, read_json, require, write_json


def simple_name(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}", value),
            "invalid release filename/directory")
    return value


def install(release_path, archive_dir, destination, preset, engine_version):
    release = read_json(release_path)
    require(release["schema_version"] == 1, "unsupported release schema")
    require(engine_version in release["engine_versions"], "engine version not qualified")
    require(preset in release["presets"], "unknown preset")
    selected = release["presets"][preset]
    require(isinstance(selected, dict) and selected, "empty preset")
    artifacts = {a["bundle_id"]: a for a in release["artifacts"]}
    require(len(artifacts) == len(release["artifacts"]), "duplicate artifact ID")
    validated = []
    for slot, bundle_id in selected.items():
        simple_name(slot)
        simple_name(bundle_id)
        a = artifacts[bundle_id]
        path = archive_dir / simple_name(a["archive"])
        require(a["kind"] in ("index", "model"), "unsupported artifact kind")
        require(re.fullmatch(r"[a-f0-9]{64}", a["sha256"]), "invalid archive SHA256")
        require(type(a["bytes"]) is int and a["bytes"] > 0, "invalid archive size")
        require(path.is_file() and not path.is_symlink(), "archive must be a regular file")
        require(path.stat().st_size == a["bytes"] and digest(path) == a["sha256"],
                f"archive checksum mismatch: {path.name}")
        validated.append((slot, a, path))
    # Reservation is exclusive. Failed installs retain .incomplete for diagnosis;
    # existing directories and mapped files are never merged or replaced.
    destination.mkdir(parents=True, exist_ok=False)
    marker = destination / ".incomplete"
    marker.write_text("Installation incomplete. Do not use this directory.\n")
    receipts = []
    for slot, a, path in validated:
        with tempfile.TemporaryDirectory(prefix=".unpack-", dir=destination) as temporary:
            stage = Path(temporary)
            # Inspect the complete member set before writing any archive content.
            with tarfile.open(path) as archive:
                members = archive.getmembers()
                names = set()
                for member in members:
                    parts = safe_relative(member.name).parts
                    require(len(parts) == 2 and parts[0] == a["bundle_id"] and member.isfile(),
                            "archive must contain one flat root of regular files")
                    require(member.name not in names, "duplicate archive member")
                    names.add(member.name)
                root = PurePosixPath(a["bundle_id"])
                m = archive.extractfile(str(root / "bundle.json"))
                require(m is not None, "missing bundle.json")
                import json
                bundle = json.load(m)
                allowed_runtime = data_bundle.RUNTIME if a["kind"] == "index" else model_bundle.FILES
                runtime = {f["path"] for f in bundle["files"]}
                require(runtime <= allowed_runtime, "unexpected bundle runtime filename")
                evidence = {"LICENSE-DATA.txt", "PROVENANCE.json", "INSTALL.md"}
                if a["kind"] == "model" or bundle.get("index", {}).get("key_type") == "lgq-gd":
                    evidence.add("MODEL-CARD.md")
                expected = {str(root / n) for n in runtime | evidence | {"bundle.json"}}
                require(names == expected, "unexpected or missing archive members")
            unpack_tar(path, stage)
            unpacked = stage / a["bundle_id"]
            verifier = data_bundle if a["kind"] == "index" else model_bundle
            checked = verifier.verify(unpacked, unpacked / "bundle.json", engine_version)
            require(checked["metadata"]["bundle_id"] == a["bundle_id"], "bundle ID mismatch")
            require(digest(unpacked / "bundle.json") == a["manifest_sha256"], "manifest mismatch")
            if a["kind"] == "index" and a.get("coder_sha256") is not None:
                require(checked["index"]["coder_sha256"] == a["coder_sha256"], "release coder mismatch")
            shutil.move(str(unpacked), str(destination / slot))
            receipts.append({"directory": slot, "bundle_id": a["bundle_id"], "archive_sha256": a["sha256"]})
    write_json(destination / "INSTALLATION.json", {"schema_version": 1, "release_id": release["release_id"],
               "release_sha256": digest(release_path), "engine_version": engine_version,
               "preset": preset, "artifacts": receipts})
    marker.unlink()
    return receipts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--archive-dir", type=Path)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--preset", choices=("full", "full-deep", "complete", "handcrafted", "model"), default="full")
    p.add_argument("--engine-version", required=True)
    a = p.parse_args()
    try:
        installed = install(a.release, a.archive_dir or a.release.parent, a.dest, a.preset, a.engine_version)
    except (ValueError, OSError, KeyError, TypeError, tarfile.TarError) as error:
        p.exit(2, f"install_data: {error}\n")
    for item in installed:
        print(f"installed {item['bundle_id']} -> {a.dest / item['directory']}")


if __name__ == "__main__":
    main()
