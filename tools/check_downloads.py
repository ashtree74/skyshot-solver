#!/usr/bin/env python3
"""Read-only smoke check of public Downloads routing, links and artifact identity.

No full archive download. Use --archive-dir for comparison of actual range bytes;
fetch_data.py performs the separate complete archive/runtime SHA256 verification.
"""
import argparse
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from data_bundle import digest, read_json, require
from fetch_data import DEFAULT_RELEASE
from install_data import simple_name

ROOT_URL = "https://skyshot.app/downloads/"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        attribute = "src" if tag == "script" else "href"
        if tag in ("a", "link", "script") and attribute in attrs:
            url = urllib.parse.urljoin(ROOT_URL, attrs[attribute])
            if url.startswith(ROOT_URL) and "#" not in url:
                self.links.add(url)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def check(release_path, archive_dir=None):
    release = read_json(release_path)
    base = release["base_url"]
    require(base.startswith(ROOT_URL + "releases/"), "unexpected public base URL")
    with urllib.request.urlopen(base + "release.json", timeout=30) as r:
        require(hashlib.sha256(r.read()).hexdigest() == digest(release_path), "public release differs from pinned catalogue")
    try:
        urllib.request.build_opener(NoRedirect()).open(
            urllib.request.Request(ROOT_URL.rstrip("/"), method="HEAD"), timeout=30)
        raise ValueError("missing canonical redirect")
    except urllib.error.HTTPError as r:
        require(r.code == 308 and r.headers.get("Location") == "/downloads/", "incorrect canonical redirect")
    for url in (base, ROOT_URL + "staging/", base + "SOURCE-INTEGRITY.json"):
        try:
            urllib.request.urlopen(url, timeout=30)
            raise ValueError("unexpected directory listing or preparation file")
        except urllib.error.HTTPError as r:
            require(r.code == 404, "unexpected non-public path response")
    with urllib.request.urlopen(ROOT_URL, timeout=30) as r:
        page = r.read().decode("utf-8")
    parser = Links()
    parser.feed(page)
    require(parser.links, "no public download links")
    for url in sorted(parser.links):
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as r:
            expected = "public, max-age=31536000, immutable" if "/releases/" in url else "public, max-age=300"
            require(r.status == 200 and r.headers.get("Cache-Control") == expected, f"link/cache failure: {url}")
    for a in release["artifacts"]:
        name = simple_name(a["archive"])
        url = base + name
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as r:
            require(int(r.headers["Content-Length"]) == a["bytes"], f"archive length mismatch: {name}")
        offset = min(1048576, a["bytes"] // 2)
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{offset + 1023}", "Accept-Encoding": "gzip"})
        with urllib.request.urlopen(request, timeout=30) as r:
            require(r.status == 206 and r.headers.get("Content-Range") == f"bytes {offset}-{offset + 1023}/{a['bytes']}"
                    and r.headers.get("Content-Encoding") is None, f"range/encoding failure: {name}")
            body = r.read(1025)
            require(len(body) == 1024, f"range length mismatch: {name}")
        if archive_dir is not None:
            with (archive_dir / name).open("rb") as source:
                source.seek(offset)
                require(source.read(1024) == body, f"range bytes mismatch: {name}")
        with urllib.request.urlopen(base + simple_name(a["manifest"]), timeout=30) as r:
            require(hashlib.sha256(r.read()).hexdigest() == a["manifest_sha256"], f"manifest mismatch: {name}")
    return {"passed": True, "release_id": release["release_id"], "page_links": len(parser.links),
            "archives": len(release["artifacts"]), "range_bytes_compared": archive_dir is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--archive-dir", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(check(args.release, args.archive_dir), indent=2))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"check_downloads: {error}\n")


if __name__ == "__main__":
    main()
