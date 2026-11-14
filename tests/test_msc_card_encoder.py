from __future__ import annotations

import unittest

from ComP.knowledge.msc.card_encoder import SentenceCardEncoder


class MSCCardEncoderTest(unittest.TestCase):
    def test_card_text_composition(self):
        text = SentenceCardEncoder.card_text(
            {
                "evidence_text": "the speaker sounds excited",
                "trigger_text": "first utterance",
                "relation_type": "retrieved_positive_context",
            }
        )
        self.assertIn("excited", text)
        self.assertIn("first utterance", text)
        self.assertIn("positive", text)

    def test_card_text_skips_empty_parts(self):
        text = SentenceCardEncoder.card_text(
            {"evidence_text": "hello", "trigger_text": None, "relation_type": ""}
        )
        self.assertEqual(text, "hello")


if __name__ == "__main__":
    unittest.main()
