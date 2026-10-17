#!/usr/bin/env python3
"""Stage a local CLI candidate with exact source identity and dependency notices.

Python 3.11+; run after cargo build --release --locked --target TARGET.
Creates no tag, GitHub release, registry package or network upload.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

if sys.version_info < (3, 11):
    raise SystemExit("package_cli requires Python 3.11+ (check python3 --version)")
import tomllib

from data_bundle import digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("src", "tests", "examples", "bench", "docs", "provenance", "tools",
               "third_party", "research", ".github")
SOURCE_FILES = ("Cargo.toml", "Cargo.lock", "README.md", "LICENSE", "NOTICE",
                "THIRD_PARTY_NOTICES.md", "CONTRIBUTING.md", "CITATION.cff", ".gitignore", ".gitattributes")


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def source_files():
    files = [ROOT / name for name in SOURCE_FILES if (ROOT / name).is_file()]
    for directory in SOURCE_DIRS:
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                # Explicitly exclude agent instruction files, including nested variants.
                if path.name.lower() not in {"agents.md", "claude.md", "claud.md"}:
                    require(not path.is_symlink(), f"source symlink refused: {path}")
                    files.append(path)
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())


def source_identity():
    entries = [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in source_files()]
    h = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"sha256": h, "algorithm": "SHA256 of compact sorted-key JSON files array",
            "files": entries}


def dependency_notices(target, destination):
    meta = json.loads(command("cargo", "metadata", "--locked", "--offline", "--format-version", "1",
                              "--filter-platform", target))
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    root_id = meta["resolve"]["root"]
    visited = set()

    def visit(node_id):
        if node_id in visited:
            return
        visited.add(node_id)
        for dep in nodes[node_id]["deps"]:
            if any(k["kind"] != "dev" for k in dep["dep_kinds"]):
                visit(dep["pkg"])

    visit(root_id)
    packages = sorted((p for p in meta["packages"] if p["id"] in visited and p["id"] != root_id),
                      key=lambda p: (p["name"], p["version"]))
    inventory = []
    for package in packages:
        name = f"{package['name']}-{package['version']}"
        root = Path(package["manifest_path"]).parent
        texts = [p for p in root.rglob("*") if p.is_file() and
                 p.name.upper().startswith(("LICENSE", "LICENCE", "COPYING", "NOTICE", "COPYRIGHT"))]
        if not texts:
            root = ROOT / "third_party" / name
            texts = [p for p in root.glob("*") if p.name.startswith(("LICENSE", "NOTICE", "COPYRIGHT"))]
            if texts and (root / "README.md").is_file():
                texts.append(root / "README.md")
        require(texts, f"no license text for {name}; add a reviewed version-specific override")
        dest = destination / name
        dest.mkdir(parents=True)
        licenses = []
        for path in sorted(texts):
            relative = path.relative_to(root)
            require(not path.is_symlink(), f"license symlink: {path}")
            out = dest / relative
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, out)
            licenses.append({"path": relative.as_posix(), "sha256": digest(out)})
        inventory.append({"name": package["name"], "version": package["version"],
                          "license": package["license"], "source": package["source"],
                          "repository": package["repository"], "texts": licenses})
    write_json(destination / "inventory.json", {"target": target, "packages": inventory,
               "scope": "Cargo normal/build dependency graph, excluding dev-only dependencies; "
                        "includes host build tools and all supplied license alternatives."})


def rust_notices(destination):
    root = Path(command("rustc", "--print", "sysroot"))
    candidates = [root / "share/doc/rust/html", root / "share/doc/rust", root / "share/doc/rustc", root]
    version = command("rustc", "--version").split()[1]
    supplement = ROOT / "third_party" / f"rust-{version}"
    candidates.append(supplement)
    out = destination / "rust-standard-library"
    out.mkdir()
    for name in ("COPYRIGHT-library.html", "LICENSE-APACHE", "LICENSE-MIT"):
        paths = [p / name for p in candidates if (p / name).is_file()]
        require(paths, f"missing Rust {name}; install rust-docs for the build toolchain")
        shutil.copyfile(paths[0], out / name)
    if (supplement / "README.md").is_file():
        shutil.copyfile(supplement / "README.md", out / "README.md")


def package(target, binary, output):
    require(re_target(target), "unsupported target")
    require(binary.is_file() and not binary.is_symlink(), "binary is not a regular file")
    with binary.open("rb") as stream:
        header = stream.read(4096)
    if "apple" in target:
        machine = int.from_bytes(header[4:8], "little")
        require(header[:4] == b"\xcf\xfa\xed\xfe" and machine ==
                (0x100000C if target.startswith("aarch64") else 0x1000007), "Mach-O target mismatch")
    elif "linux" in target:
        machine = int.from_bytes(header[18:20], "little")
        require(header[:6] == b"\x7fELF\x02\x01" and machine ==
                (183 if target.startswith("aarch64") else 62), "ELF target mismatch")
    else:
        offset = int.from_bytes(header[60:64], "little")
        require(header[:2] == b"MZ" and header[offset:offset + 6] == b"PE\0\0\x64\x86",
                "PE target mismatch")
    package_meta = tomllib.loads((ROOT / "Cargo.toml").read_text())["package"]
    version = package_meta["version"]
    require(command(str(binary.resolve()), "--version") == f"{package_meta['name']} {version}",
            "binary version mismatch")
    output.mkdir(parents=True, exist_ok=True)
    name = f"skyshot-{version}-{target}"
    archive = output / (name + (".zip" if "windows" in target else ".tar.gz"))
    require(not archive.exists() and not archive.with_name(archive.name + ".sha256").exists(),
            "candidate already exists; use a new output directory")
    with tempfile.TemporaryDirectory(prefix="skyshot-package-") as tmp:
        stage = Path(tmp) / name
        stage.mkdir()
        shutil.copyfile(binary, stage / ("skyshot.exe" if "windows" in target else "skyshot"))
        (stage / ("skyshot.exe" if "windows" in target else "skyshot")).chmod(0o755)
        for path in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "Cargo.lock",
                     "docs/CLI.md", "docs/DATA.md", "docs/INSTALL.md", "docs/DATA-INSTALL.md", "docs/DATA-RELEASE.md",
                     "docs/DATA-PROVENANCE.md", "docs/RELEASE.md", "docs/verification-2026-10-17.md"):
            shutil.copyfile(ROOT / path, stage / Path(path).name)
        notices = stage / "third_party"
        notices.mkdir()
        dependency_notices(target, notices)
        rust_notices(notices)
        identity = source_identity()
        write_json(stage / "source-manifest.json", identity)
        write_json(stage / "build.json", {"schema_version": 1, "version": version, "target": target,
                   "rustc": command("rustc", "-Vv"), "cargo": command("cargo", "-V"),
                   "source_sha256": identity["sha256"], "binary_sha256": digest(binary),
                   "qualification": "Local/CI candidate; see platform acceptance report before publication."})
        if "windows" in target:
            with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as stream:
                for path in sorted(stage.rglob("*")):
                    if path.is_file():
                        stream.write(path, path.relative_to(stage.parent))
        else:
            with archive.open("xb") as out:
                with tarfile.open(fileobj=out, mode="w:gz") as stream:
                    stream.add(stage, arcname=name)
    with archive.with_name(archive.name + ".sha256").open("x", encoding="utf-8") as stream:
        stream.write(f"{digest(archive)}  {archive.name}\n")
    print(archive)


def re_target(target):
    return target in {"aarch64-apple-darwin", "x86_64-apple-darwin", "x86_64-unknown-linux-gnu",
                      "aarch64-unknown-linux-gnu", "x86_64-pc-windows-msvc"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--binary", type=Path, help="defaults to the native target/release executable")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.target == "host":
            args.target = next(line.removeprefix("host: ") for line in command("rustc", "-Vv").splitlines()
                               if line.startswith("host: "))
        if args.binary is None:
            args.binary = ROOT / "target/release" / ("skyshot.exe" if "windows" in args.target else "skyshot")
        package(args.target, args.binary, args.out)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"package_cli: {error}\n")


if __name__ == "__main__":
    main()
