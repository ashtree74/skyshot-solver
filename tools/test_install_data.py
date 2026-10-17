"""Exercise archive/install boundaries with tiny artificial bytes, no real data."""
import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import data_bundle as data
import install_data
import model_bundle as model
from prepare_data_release import no_private_text


class InstallationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.index = self.root / "source"
        self.index.mkdir()
        files = ["codes.bin", "keys.bin", "stars.bin", "cells.bin"]
        for name in files:
            (self.index / name).write_bytes(b"artificial")
        data.write_json(self.index / "manifest.json", {"format": "skyanchor2-index", "version": 1,
                        "profile": "synthetic", "r7_arcmin": 1.94, "catalog_fingerprint": "synthetic",
                        "files": [[n, 10] for n in files]})
        meta = {"bundle_id": "fixture-1", "engine_versions": ["0.1.0"], "sources": [{"synthetic": True}],
                "build": {"test": True}, "author": "Adam Jesionkiewicz", "redistribution": {"status": "pending"}}
        data.write_json(self.root / "metadata.json", meta)
        data.create(self.index, self.root / "metadata.json", self.root / "bundle.json")
        for name in ("LICENSE-DATA.txt", "PROVENANCE.json", "INSTALL.md"):
            (self.root / name).write_text("Artificial test bytes")
        self.archive = self.root / "fixture-1.tar.gz"
        data.pack(self.index, self.root / "bundle.json", "0.1.0", self.root, self.archive)
        self.release = {"schema_version": 1, "release_id": "fixture-release", "engine_versions": ["0.1.0"],
                        "presets": {"handcrafted": {"index-v1": "fixture-1"}}, "artifacts": [
                            {"bundle_id": "fixture-1", "kind": "index", "archive": self.archive.name,
                             "bytes": self.archive.stat().st_size, "sha256": data.digest(self.archive),
                             "manifest_sha256": data.digest(self.root / "bundle.json")}]}
        self.release_path = self.root / "release.json"
        self.save_release()

    def save_release(self):
        self.release_path.write_text(json.dumps(self.release))

    def install(self, dest=None):
        return install_data.install(self.release_path, self.root, dest or self.root / "installed",
                                    "handcrafted", "0.1.0")

    def test_roundtrip_no_source_or_existing_install_changes(self):
        before = data.digest(self.index / "manifest.json")
        with tarfile.open(self.archive) as archive:
            for member in archive:
                self.assertEqual((member.uid, member.gid, member.uname, member.gname, member.mtime),
                                 (0, 0, "", "", 0))
        self.install()
        target = self.root / "installed"
        self.assertTrue((target / "INSTALLATION.json").is_file())
        self.assertFalse((target / ".incomplete").exists())
        self.assertEqual(data.read_json(target / "index-v1/bundle.json")["metadata"]["author"], "Adam Jesionkiewicz")
        with self.assertRaises(FileExistsError):
            self.install()
        self.assertEqual(before, data.digest(self.index / "manifest.json"))

    def test_corruption_refused_before_destination_created(self):
        self.archive.write_bytes(self.archive.read_bytes() + b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.install()
        self.assertFalse((self.root / "installed").exists())

    def test_extra_file_rejected_even_when_outer_hash_matches(self):
        # A new valid tar with an unexpected agent file is never unpacked.
        other = self.root / "changed.tar.gz"
        (self.root / "AGENTS.md").write_text("untrusted")
        with tarfile.open(self.archive) as src, tarfile.open(other, "w:gz") as dst:
            for member in src:
                dst.addfile(member, src.extractfile(member))
            dst.add(self.root / "AGENTS.md", "fixture-1/AGENTS.md", recursive=False)
        self.release["artifacts"][0].update(archive=other.name, bytes=other.stat().st_size, sha256=data.digest(other))
        self.save_release()
        with self.assertRaisesRegex(ValueError, "archive members"):
            self.install()
        self.assertTrue((self.root / "installed/.incomplete").exists())
        self.assertFalse((self.root / "installed/index-v1").exists())

    def test_unsafe_slot_and_unqualified_version(self):
        self.release["presets"]["handcrafted"] = {"../escape": "fixture-1"}
        self.save_release()
        with self.assertRaisesRegex(ValueError, "filename/directory"):
            self.install()
        with self.assertRaisesRegex(ValueError, "not qualified"):
            install_data.install(self.release_path, self.root, self.root / "different", "handcrafted", "0.2.0")

    def test_standalone_model_metadata_is_part_of_identity(self):
        root = self.root / "model"
        root.mkdir()
        (root / "coder.f32").write_bytes(b"artificial weights")
        data.write_json(root / "coder.json", {"format": "skyanchor2-learned-coder-lgq", "version": 1,
                                              "author": "Adam Jesionkiewicz"})
        manifest = self.root / "model.bundle.json"
        model.create(root, self.root / "metadata.json", manifest)
        model.verify(root, manifest, "0.1.0")
        expected = hashlib.sha256((root / "coder.f32").read_bytes() + (root / "coder.json").read_bytes()).hexdigest()
        self.assertEqual(model.identity(root)["coder_sha256"], expected)
        meta = data.read_json(root / "coder.json")
        meta["author"] = "changed"
        (root / "coder.json").write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            model.verify(root, manifest, "0.1.0")

    def test_metadata_path_screen(self):
        no_private_text({"author": "Adam Jesionkiewicz", "path": "work/index"})
        for value in ("/Users/private/data", "/mnt/data/train", "AGENTS.md", "claud.md"):
            with self.assertRaisesRegex(ValueError, "private path"):
                no_private_text({"field": value})


if __name__ == "__main__":
    unittest.main()
