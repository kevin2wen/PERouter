from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ComP.per.manifest_utils import audit_manifest


class MaskAuditTest(unittest.TestCase):
    def test_counts_observed_and_missing_distribution(self):
        rows = [
            "seed,sample_id,utterance_index,utterance_id,availability,mask_rate,sampler_version",
            '66,s,0,u0,"[1,1,1]",0.5,categorical',
            '66,s,1,u1,"[1,0,1]",0.5,categorical',
            '66,t,0,u2,"[0,1,0]",0.5,categorical',
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.csv"
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            report = audit_manifest(path)
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["utterance_count"], 3)
        self.assertAlmostEqual(report["full_visible_rate"], 1 / 3)
        self.assertAlmostEqual(report["one_missing_rate"], 1 / 3)
        self.assertAlmostEqual(report["two_missing_rate"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
