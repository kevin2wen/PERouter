from __future__ import annotations

import unittest

import torch

from ComP.knowledge.msc.single_scorer import MSCSingleScorer


class MSCSingleScorerTest(unittest.TestCase):
    def test_output_shape_and_mask(self):
        scorer = MSCSingleScorer(8, 16, 4, hidden_dim=32)
        n, k = 3, 5
        out = scorer(
            torch.randn(n, 16), torch.randn(n, 4), torch.randn(n, k, 8),
            torch.tensor([[1, 1, 0, 0, 0]] * n, dtype=torch.bool),
            torch.rand(n, k), torch.rand(n, k),
        )
        self.assertEqual(tuple(out.shape), (n, k))
        self.assertTrue(torch.isinf(out[:, 2:]).all())
        self.assertTrue(torch.isfinite(out[:, :2]).all())

    def test_semantic_toggle_changes_scores(self):
        torch.manual_seed(0)
        full = MSCSingleScorer(8, 4, 4, hidden_dim=32, use_semantic=True)
        meta = MSCSingleScorer(8, 4, 4, hidden_dim=32, use_semantic=False)
        meta.load_state_dict(full.state_dict())
        args = (
            torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 5, 8),
            torch.ones(2, 5, dtype=torch.bool),
            torch.rand(2, 5), torch.rand(2, 5),
        )
        self.assertFalse(torch.allclose(full(*args), meta(*args)))


if __name__ == "__main__":
    unittest.main()
