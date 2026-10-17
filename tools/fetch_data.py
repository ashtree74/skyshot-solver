#!/usr/bin/env python3
"""Download pinned Skyshot data over HTTPS, resume transfers and install into a NEW directory.

Python 3.11+, standard library only. The committed release catalogue pins every
archive. Index files are never downloaded into or merged with a live index.
"""
import argparse
import http.client
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from data_bundle import digest, read_json, require
from install_data import install, simple_name

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "provenance/data-releases/2026.10.17.2/release.json"


def fetch_file(url, path, expected_bytes, expected_sha256, *, allow_http=False, progress=None):
    parsed = urllib.parse.urlsplit(url)
    require(parsed.scheme == "https" or (allow_http and parsed.scheme == "http" and
            parsed.hostname in ("127.0.0.1", "localhost", "::1")), "HTTPS is required")
    require(parsed.username is None and parsed.password is None, "credentials in download URL refused")
    require(type(expected_bytes) is int and expected_bytes > 0, "invalid expected byte size")
    require(re.fullmatch(r"[a-f0-9]{64}", expected_sha256), "invalid expected SHA256")
    require(not path.is_symlink(), "download destination must not be a symlink")
    if path.exists():
        require(path.is_file() and path.stat().st_size == expected_bytes and digest(path) == expected_sha256,
                f"cached file is corrupt; use a new cache or remove this cache file: {path.name}")
        return
    partial = path.with_name(path.name + ".part")
    require(not partial.is_symlink() and (not partial.exists() or partial.is_file()), "invalid partial download")
    have = partial.stat().st_size if partial.exists() else 0
    require(have <= expected_bytes, "partial download exceeds expected size")
    if have < expected_bytes:
        headers = {"Accept-Encoding": "identity", "User-Agent": "Skyshot-data-installer/1"}
        if have:
            headers["Range"] = f"bytes={have}-"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            final = urllib.parse.urlsplit(response.geturl())
            require(final.scheme == parsed.scheme, "download redirected to a different URL scheme")
            require(response.headers.get("Content-Encoding", "identity") == "identity", "encoded download refused")
            status = response.status
            if status == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                require(match is not None and tuple(map(int, match.groups())) ==
                        (have, expected_bytes - 1, expected_bytes), "invalid resumed Content-Range")
            else:
                require(status == 200, f"unexpected HTTP status {status}")
                have = 0  # A server ignoring Range sends the whole object; restart only our cache partial.
            length = response.headers.get("Content-Length")
            require(length is not None and int(length) == expected_bytes - have, "unexpected Content-Length")
            mode = "ab" if have else ("wb" if partial.exists() else "xb")
            with partial.open(mode) as out:
                while True:
                    chunk = response.read(min(1024 * 1024, expected_bytes - have + 1))
                    if not chunk:
                        break
                    have += len(chunk)
                    require(have <= expected_bytes, "download exceeds expected size")
                    out.write(chunk)
                    if progress:
                        progress(path.name, have, expected_bytes)
    require(have == expected_bytes, "incomplete download; rerun to resume")
    require(digest(partial) == expected_sha256, f"download SHA256 mismatch: {path.name}")
    require(not path.exists(), "download destination appeared during transfer")
    partial.rename(path)


def fetch_and_install(release_path, cache, destination, preset, engine_version, *, progress=None):
    trusted = read_json(release_path)
    require(trusted["schema_version"] == 1, "unsupported release schema")
    require(engine_version in trusted["engine_versions"], "engine version not qualified")
    require(preset in trusted["presets"], "unknown preset")
    require(not destination.exists() and not destination.is_symlink(), "installation destination must be NEW")
    base = trusted["base_url"]
    require(base.startswith("https://") and base.endswith("/"), "release base_url must use HTTPS and end with /")
    cache.mkdir(parents=True, exist_ok=True)
    lock = cache / ".download.lock"
    with lock.open("x") as stream:
        stream.write(f"{os.getpid()}\n")
    try:
        pinned = cache / "release.json"
        fetch_file(base + "release.json", pinned, release_path.stat().st_size, digest(release_path), progress=progress)
        artifacts = {a["bundle_id"]: a for a in trusted["artifacts"]}
        require(len(artifacts) == len(trusted["artifacts"]), "duplicate artifact IDs")
        for bundle_id in trusted["presets"][preset].values():
            artifact = artifacts[bundle_id]
            name = simple_name(artifact["archive"])
            print(f"Checking/downloading {name}", flush=True)
            fetch_file(base + name, cache / name, artifact["bytes"], artifact["sha256"], progress=progress)
        return install(pinned, cache, destination, preset, engine_version)
    finally:
        lock.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    p.add_argument("--preset", choices=("full", "full-deep", "complete", "handcrafted", "model"), default="full")
    p.add_argument("--cache", type=Path)
    p.add_argument("--dest", required=True, type=Path)
    p.add_argument("--engine-version", default="0.1.0")
    a = p.parse_args()
    last = [0.0]

    def progress(name, have, total):
        now = time.monotonic()
        if now - last[0] > 10 or have == total:
            print(f"{name}: {have / 1e6:.1f}/{total / 1e6:.1f} MB ({have * 100 / total:.0f}%)", flush=True)
            last[0] = now

    try:
        release = read_json(a.release)
        cache = a.cache or Path("data/download-cache") / simple_name(release["release_id"])
        receipts = fetch_and_install(a.release, cache, a.dest, a.preset, a.engine_version, progress=progress)
    except (ValueError, OSError, KeyError, TypeError, urllib.error.URLError, http.client.HTTPException) as error:
        p.exit(2, f"fetch_data: {error}\n")
    for item in receipts:
        print(f"Installed {item['bundle_id']} -> {a.dest / item['directory']}")
    print("Installation verified. See https://skyshot.app/downloads/ for your first solve.")


if __name__ == "__main__":
    main()
