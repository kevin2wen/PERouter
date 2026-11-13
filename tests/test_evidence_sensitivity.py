import unittest

import torch

from ComP.per.evidence_sensitivity import (
    masked_mean,
    masked_rate,
    paired_logit_deltas,
)


class EvidenceSensitivityTest(unittest.TestCase):
    def test_identical_evidence_has_zero_delta(self):
        logits = torch.tensor([[[0.2], [1.0], [-0.5]]])
        mask = torch.tensor([[True, True, False]])
        result = paired_logit_deltas(
            logits, logits.clone(), torch.zeros(1, 3), mask, "regression"
        )
        self.assertEqual(masked_mean(result["logits_delta_l2"], mask), 0.0)
        self.assertEqual(masked_rate(result["sign_flip"], mask), 0.0)

    def test_sign_flip_and_prediction_delta_are_detected(self):
        reference = torch.tensor([[[0.2], [1.0]]])
        comparison = torch.tensor([[[-0.3], [1.25]]])
        mask = torch.tensor([[True, True]])
        result = paired_logit_deltas(
            reference, comparison, torch.zeros(1, 2), mask, "regression"
        )
        self.assertTrue(result["sign_flip"][0, 0])
        self.assertAlmostEqual(float(result["prediction_delta"][0, 0]), 0.5, places=6)

    def test_padding_is_excluded(self):
        values = torch.tensor([[1.0, 100.0]])
        mask = torch.tensor([[True, False]])
        self.assertAlmostEqual(masked_mean(values, mask), 1.0)

    def test_same_classification_prediction_has_no_sign_flip(self):
        reference = torch.tensor([[[3.0, 1.0], [1.0, 2.0]]])
        comparison = torch.tensor([[[2.0, 0.5], [4.0, 0.1]]])
        mask = torch.tensor([[True, True]])
        result = paired_logit_deltas(
            reference, comparison, torch.zeros(1, 2), mask, "classification"
        )
        self.assertEqual(result["sign_flip"].tolist(), [[False, True]])
