from __future__ import annotations

import unittest

import torch

from ComP.per.evaluation_audit import EvaluationAuditAccumulator


class EvaluationAuditTest(unittest.TestCase):
    def test_pattern_counts_exclude_zero_label_and_padding(self):
        labels = torch.tensor([[0.0, 1.0, -1.0]])
        predictions = torch.tensor([[0.1, 0.8, -0.7]])
        availability = torch.tensor(
            [
                [
                    [True, True, True],
                    [True, False, True],
                    [False, True, True],
                ]
            ],
            dtype=torch.bool,
        )
        sequence = torch.tensor([[True, True, False]])
        accumulator = EvaluationAuditAccumulator()
        accumulator.add(labels, predictions, availability, sequence)
        result = accumulator.summary("CMUMOSI")

        self.assertEqual(result["pattern_full_count"], 0.0)
        self.assertEqual(result["pattern_text_missing_count"], 1.0)
        self.assertEqual(result["pattern_audio_missing_count"], 0.0)
        self.assertEqual(result["pattern_text_missing_acc"], 1.0)
        self.assertAlmostEqual(
            result["pattern_text_missing_mae"], 0.2, places=5
        )


if __name__ == "__main__":
    unittest.main()
