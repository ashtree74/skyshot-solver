import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from archive_utils import unpack_tar


class ArchiveTests(unittest.TestCase):
    def test_traversal_is_refused_without_writing_outside_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                info = tarfile.TarInfo("../escaped")
                info.size = 4
                stream.addfile(info, io.BytesIO(b"oops"))
            with self.assertRaisesRegex(ValueError, "unsafe archive path"):
                unpack_tar(archive, root / "unpack")
            self.assertFalse((root / "escaped").exists())

    def test_links_cannot_redirect_a_later_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "link.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                info = tarfile.TarInfo("bundle/link")
                info.type = tarfile.SYMTYPE
                info.linkname = "../.."
                stream.addfile(info)
            with self.assertRaisesRegex(ValueError, "links/special files refused"):
                unpack_tar(archive, root / "unpack")
            self.assertFalse((root / "unpack/bundle/link").exists())


if __name__ == "__main__":
    unittest.main()
