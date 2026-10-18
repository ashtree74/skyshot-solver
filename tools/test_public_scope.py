"""Keep website assets and local archives out of the public engine source tree."""
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PublicScopeTests(unittest.TestCase):
    def test_tracked_source_excludes_website_and_local_archives(self):
        # Read the index rather than ignored local files or generated Rust docs.
        tracked = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
        forbidden = []
        for name in filter(None, tracked):
            path = Path(name)
            if (path.suffix.lower() in {".html", ".css", ".js", ".mjs", ".svg", ".caddy"}
                    or path.name.lower() in {"agents.md", "claude.md", "claud.md"}
                    or path.parts[0] in {".local-site", "outputs", "data", "work"}):
                forbidden.append(name)
        self.assertEqual([], forbidden, "Website and local-only files must stay outside public source")

    def test_local_website_archive_is_ignored(self):
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", ".local-site/website.html"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
