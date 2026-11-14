from __future__ import annotations

import unittest

from ComP.knowledge.msc.card_text_quality import card_text_quality


class CardTextQualityTest(unittest.TestCase):
    def test_duplicate_and_distribution(self):
        cards = [
            {"evidence_text": "same text", "trigger_text": "", "relation_type": "r"},
            {"evidence_text": "same text", "trigger_text": "", "relation_type": "r"},
            {"evidence_text": "other text", "trigger_text": "", "relation_type": "r"},
        ]
        report = card_text_quality(cards)
        self.assertEqual(report["cards"], 3)
        self.assertEqual(report["unique_texts"], 2)
        self.assertGreater(report["duplicate_text_rate"], 0.3)
        self.assertIn("r", report["relation_distribution"])


if __name__ == "__main__":
    unittest.main()
