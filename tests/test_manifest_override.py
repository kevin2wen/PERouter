from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ComP.per.manifest_utils import resolve_manifest_path, sha256_file


class ManifestOverrideTest(unittest.TestCase):
    def test_same_manifest_resolves_to_same_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.csv"
            path.write_text("sample_id,utterance_index,availability\ns,0,\"[1,0,1]\"\n")
            first = resolve_manifest_path(str(path), "/unused")
            second = resolve_manifest_path(str(path), "/different")
            self.assertEqual(first, second)
            self.assertEqual(sha256_file(first), sha256_file(second))

    def test_missing_override_fails(self):
        with self.assertRaises(FileNotFoundError):
            resolve_manifest_path("/tmp/does-not-exist-mask.csv", "fallback.csv")


if __name__ == "__main__":
    unittest.main()
