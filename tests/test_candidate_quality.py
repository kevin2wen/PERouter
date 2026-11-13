import unittest

import torch

from ComP.per.evidence_quality import (
    binary_label_consistency,
    candidate_statistics,
    rank_histogram,
    recall_at_k,
)


class CandidateQualityTest(unittest.TestCase):
    def test_best_candidate_and_gain(self):
        utility = torch.tensor([[[[0.1, 0.5, -0.2], [0.0, 0.1, 0.2], [9.0, 9.0, 9.0]]]])
        prompt = torch.tensor([[[0.2, 0.0, 0.0]]])
        active = torch.tensor([[[True, True, False]]])
        result = candidate_statistics(utility, prompt, active)
        self.assertEqual(result["best_rank"].tolist(), [[[1, 2, 0]]])
        self.assertAlmostEqual(float(result["oracle_gain"][0, 0, 0]), 0.3, places=6)
        self.assertFalse(bool(result["has_helpful_candidate"][0, 0, 2]))

    def test_recall_prefix_is_monotonic(self):
        utility = torch.tensor([[[[0.0, 0.3, 0.4], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]]])
        prompt = torch.tensor([[[0.2, 0.1, 0.0]]])
        active = torch.tensor([[[True, True, False]]])
        recall = recall_at_k(utility, prompt, active)
        self.assertLessEqual(recall[1], recall[2])
        self.assertLessEqual(recall[2], recall[3])
        self.assertEqual(recall[3], 1.0)

    def test_rank_histogram_excludes_inactive(self):
        best = torch.tensor([[[0, 2, 0]]])
        active = torch.tensor([[[True, True, False]]])
        self.assertEqual(rank_histogram(best, active, 3), {0: 1, 1: 0, 2: 1})

    def test_binary_label_consistency(self):
        candidate = torch.tensor([[[1.0, -1.0, 0.5]]])
        target = torch.tensor([[[0.5, -0.5, -0.5]]])
        active = torch.tensor([[[True, True, False]]])
        self.assertEqual(binary_label_consistency(candidate, target, active), 1.0)
