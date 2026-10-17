#!/usr/bin/env python3
"""Verify an archive made by package_cli, unpack safely and exercise its binary."""
import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

from archive_utils import safe_relative, unpack_tar
from data_bundle import digest, require
from smoke_cli import smoke


def check(directory, demo):
    archives = list(directory.glob("*.tar.gz")) + list(directory.glob("*.zip"))
    require(len(archives) == 1, "expected exactly one candidate archive")
    archive = archives[0]
    expected, name = archive.with_name(archive.name + ".sha256").read_text().split()
    require(name == archive.name and digest(archive) == expected, "archive SHA256 mismatch")
    with tempfile.TemporaryDirectory(prefix="skyshot-unpack-") as tmp:
        dest = Path(tmp)
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as stream:
                for info in stream.infolist():
                    safe_relative(info.filename)
                    require((info.external_attr >> 16) & 0o170000 != 0o120000, "ZIP symlink refused")
                stream.extractall(dest)
        else:
            unpack_tar(archive, dest)
        roots = list(dest.iterdir())
        require(len(roots) == 1 and roots[0].is_dir(), "expected single archive root")
        root = roots[0]
        build = json.loads((root / "build.json").read_text())
        source = json.loads((root / "source-manifest.json").read_text())
        source_digest = hashlib.sha256(json.dumps(source["files"], sort_keys=True,
                                                  separators=(",", ":")).encode()).hexdigest()
        require(source_digest == source["sha256"] == build["source_sha256"], "source identity mismatch")
        binary = root / ("skyshot.exe" if "windows" in build["target"] else "skyshot")
        require(digest(binary) == build["binary_sha256"], "binary SHA256 mismatch")
        smoke(binary, demo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    args = parser.parse_args()
    check(args.directory, args.demo)
