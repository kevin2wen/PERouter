from __future__ import annotations

import unittest

import numpy as np

from ComP.knowledge.msc.card_diversity import near_duplicate_report


class CardSemanticDiversityTest(unittest.TestCase):
    def test_near_duplicate_thresholds(self):
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        report = near_duplicate_report(embeddings)
        self.assertEqual(report["pairs"], 3)
        self.assertGreater(report["near_duplicate_rate_0.98"], 0.0)
        self.assertLess(report["near_duplicate_rate_0.98"], 1.0)


if __name__ == "__main__":
    unittest.main()
