"""Run with: python3 -m unittest discover -s tools -p 'test_*.py'."""
import json
import tempfile
import unittest
from pathlib import Path

import data_bundle as data
from archive_utils import unpack_tar


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.index = self.root / "index"
        self.index.mkdir()
        files = ["codes.bin", "keys.bin", "stars.bin", "cells.bin"]
        for name in files:
            (self.index / name).write_bytes(b"fixture")
        self.internal = {"format": "skyanchor2-index", "version": 1, "profile": "synthetic",
                         "catalog_fingerprint": "synthetic", "r7_arcmin": 1.94,
                         "files": [[name, 7] for name in files]}
        self.meta = {"bundle_id": "synthetic-1", "engine_versions": ["0.1.0"],
                     "sources": [{"generator": "test fixture; no real stars"}],
                     "build": {"command": "test fixture"},
                     "redistribution": {"status": "pending"}}
        self.metadata = self.root / "metadata.json"
        self.manifest = self.root / "bundle.json"
        self.save_inputs()

    def save_inputs(self):
        (self.index / "manifest.json").write_text(json.dumps(self.internal))
        self.metadata.write_text(json.dumps(self.meta))

    def create(self):
        data.create(self.index, self.metadata, self.manifest)

    def verify(self):
        return data.verify(self.index, self.manifest, "0.1.0")

    def test_roundtrip_and_same_size_corruption(self):
        self.create()
        self.verify()
        (self.index / "codes.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.verify()

    def test_compatibility_and_incomplete_inventory(self):
        self.create()
        with self.assertRaisesRegex(ValueError, "not qualified"):
            data.verify(self.index, self.manifest, "0.2.0")
        bundle = data.read_json(self.manifest)
        bundle["files"].pop()
        self.manifest.write_text(json.dumps(bundle))
        with self.assertRaisesRegex(ValueError, "every runtime file"):
            self.verify()

    def test_path_traversal_and_duplicate_are_refused(self):
        self.create()
        bundle = data.read_json(self.manifest)
        bundle["files"][0]["path"] = "../secret"
        self.manifest.write_text(json.dumps(bundle))
        with self.assertRaisesRegex(ValueError, "unsafe/duplicate"):
            self.verify()

    def test_symlinks_and_partial_deep_pair_are_refused(self):
        (self.index / "deep.bin").write_bytes(b"deep")
        with self.assertRaisesRegex(ValueError, "incomplete deep pair"):
            self.create()
        (self.index / "deep.bin").unlink()
        (self.index / "codes.bin").unlink()
        (self.index / "codes.bin").symlink_to(self.index / "keys.bin")
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            self.create()

    def test_pending_terms_do_not_block_local_packaging_or_become_cleared(self):
        self.create()
        (self.root / "LICENSE-DATA.txt").write_text("External terms recorded separately; no new license grant")
        data.write_json(self.root / "PROVENANCE.json", self.meta)
        out = self.root / "data.tar.gz"
        data.pack(self.index, self.manifest, "0.1.0", self.root, out)
        stage = self.root / "unpacked-pending"
        unpack_tar(out, stage)
        bundle = data.verify(stage / "synthetic-1", stage / "synthetic-1/bundle.json", "0.1.0")
        self.assertEqual(bundle["metadata"]["redistribution"], {"status": "pending"})
        self.assertTrue((stage / "synthetic-1/LICENSE-DATA.txt").is_file())

    def test_local_packaging_still_requires_attribution_documents(self):
        self.create()
        out = self.root / "missing-notices.tar.gz"
        with self.assertRaisesRegex(ValueError, "missing release evidence"):
            data.pack(self.index, self.manifest, "0.1.0", self.root, out)
        self.assertFalse(out.exists())

    def test_cleared_synthetic_fixture_archive_can_be_verified(self):
        self.meta["redistribution"] = {"status": "cleared", "license": "Apache-2.0",
                                       "evidence": "artificial test bytes created here"}
        self.save_inputs()
        self.create()
        (self.root / "LICENSE-DATA.txt").write_text("Artificial test bytes only")
        data.write_json(self.root / "PROVENANCE.json", self.meta)
        out = self.root / "data.tar.gz"
        data.pack(self.index, self.manifest, "0.1.0", self.root, out)
        stage = self.root / "unpacked"
        unpack_tar(out, stage)
        data.verify(stage / "synthetic-1", stage / "synthetic-1/bundle.json", "0.1.0")
        self.assertIn(data.digest(out), out.with_name(out.name + ".sha256").read_text())

    def test_coder_hash_covers_metadata_too(self):
        self.internal.update(version=2, key_type="lgq-gd", coder_sha256="0" * 64)
        (self.index / "coder.f32").write_bytes(b"weights")
        (self.index / "coder.json").write_text("{}")
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, "coder identity mismatch"):
            self.create()

    def test_never_write_into_or_replace_index(self):
        with self.assertRaisesRegex(ValueError, "outside index"):
            data.create(self.index, self.metadata, self.index / "bundle.json")
        self.create()
        with self.assertRaises(FileExistsError):
            self.create()


if __name__ == "__main__":
    unittest.main()
