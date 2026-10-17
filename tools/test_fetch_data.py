"""Real HTTP transfer tests with artificial bytes; no external downloads."""
import hashlib
import http.server
import tempfile
import threading
import unittest
from pathlib import Path

from fetch_data import fetch_file


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.payload = b"synthetic catalogue bytes\n" * 1000
        self.sha = hashlib.sha256(self.payload).hexdigest()
        payload = self.payload
        self.ranges = []
        ranges = self.ranges
        self.behavior = {}
        behavior = self.behavior

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                header = self.headers.get("Range")
                ranges.append(header)
                resumed = header and not behavior.get("ignore_range")
                start = int(header.split("=")[1].split("-")[0]) if resumed else 0
                self.send_response(206 if resumed else 200)
                if resumed:
                    claimed_start = start + 1 if behavior.get("bad_range") else start
                    self.send_header("Content-Range", f"bytes {claimed_start}-{len(payload)-1}/{len(payload)}")
                if behavior.get("encoded"):
                    self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(payload) - start))
                self.end_headers()
                self.wfile.write(payload[start:])

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.url = f"http://127.0.0.1:{server.server_port}/fixture.bin"
        self.path = self.root / "fixture.bin"

    def fetch(self, sha=None):
        fetch_file(self.url, self.path, len(self.payload), sha or self.sha, allow_http=True)

    def test_fresh_then_verified_cache_does_not_download_again(self):
        self.fetch()
        self.assertEqual(self.path.read_bytes(), self.payload)
        self.fetch()
        self.assertEqual(len(self.ranges), 1)

    def test_resume_partial_and_verify_complete_content(self):
        self.path.with_suffix(".bin.part").write_bytes(self.payload[:123])
        self.fetch()
        self.assertEqual(self.ranges, ["bytes=123-"])
        self.assertEqual(self.path.read_bytes(), self.payload)
        self.assertFalse(self.path.with_suffix(".bin.part").exists())

    def test_tampered_download_is_never_promoted(self):
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.fetch("0" * 64)
        self.assertFalse(self.path.exists())

    def test_server_ignoring_range_restarts_our_partial(self):
        self.behavior["ignore_range"] = True
        self.path.with_suffix(".bin.part").write_bytes(self.payload[:123])
        self.fetch()
        self.assertEqual(self.ranges, ["bytes=123-"])
        self.assertEqual(self.path.read_bytes(), self.payload)

    def test_bad_content_range_does_not_append_to_partial(self):
        self.behavior["bad_range"] = True
        partial = self.path.with_suffix(".bin.part")
        partial.write_bytes(self.payload[:123])
        with self.assertRaisesRegex(ValueError, "invalid resumed Content-Range"):
            self.fetch()
        self.assertEqual(partial.read_bytes(), self.payload[:123])
        self.assertFalse(self.path.exists())

    def test_content_encoding_cannot_change_archive_bytes(self):
        self.behavior["encoded"] = True
        with self.assertRaisesRegex(ValueError, "encoded download refused"):
            self.fetch()
        self.assertFalse(self.path.exists())
        self.assertFalse(self.path.with_suffix(".bin.part").exists())

    def test_corrupt_existing_cache_is_refused_without_overwrite(self):
        self.path.write_bytes(b"x" * len(self.payload))
        with self.assertRaisesRegex(ValueError, "cached file is corrupt"):
            self.fetch()
        self.assertEqual(self.path.read_bytes(), b"x" * len(self.payload))
        self.assertFalse(self.ranges)

    def test_public_transfer_requires_https(self):
        with self.assertRaisesRegex(ValueError, "HTTPS is required"):
            fetch_file(self.url, self.path, len(self.payload), self.sha)


if __name__ == "__main__":
    unittest.main()
