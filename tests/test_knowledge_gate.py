from __future__ import annotations

import unittest

from ComP.knowledge.gate import evaluate_ek3, evaluate_ek4


class KnowledgeGateTest(unittest.TestCase):
    def test_ek3_requires_real_card_content_gain(self):
        results = {
            "empty": {"mae": 1.0},
            "raw_feature": {"mae": 0.99},
            "card": {"mae": 0.95},
            "random_card": {"mae": 0.98},
            "shuffled_card": {"mae": 0.98},
            "source_only": {"mae": 0.98},
            "oracle_card": {"mae": 0.90},
        }
        decision = evaluate_ek3(results)
        self.assertTrue(decision["passed"])

    def test_source_only_can_reject_ek3(self):
        results = {
            "empty": {"mae": 1.0},
            "card": {"mae": 0.95},
            "random_card": {"mae": 0.98},
            "shuffled_card": {"mae": 0.98},
            "source_only": {"mae": 0.94},
            "oracle_card": {"mae": 0.90},
        }
        decision = evaluate_ek3(results)
        self.assertFalse(decision["passed"])
        self.assertEqual(decision["reason"], "SOURCE_ONLY_EXPLAINS_GAIN")

    def test_ek4_requires_filter_to_beat_controls(self):
        results = {
            "card": {"acc": 0.75},
            "confidence": {"acc": 0.70},
            "random_action": {"acc": 0.68},
            "learned_filter": {"acc": 0.74},
        }
        decision = evaluate_ek4(results, lower_is_better=False)
        self.assertTrue(decision["passed"])
