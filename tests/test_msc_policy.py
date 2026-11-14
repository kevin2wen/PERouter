from __future__ import annotations

import unittest

import torch

from ComP.knowledge.msc.cache import subset_masks_batch
from ComP.knowledge.msc.policy import MSCActionSelector, MSCPolicy


class MSCPolicyTest(unittest.TestCase):
    def test_policy_outputs_and_selector_skip_rules(self):
        policy = MSCPolicy(candidate_dim=8, state_dim=16, top_k=5, hidden_dim=32)
        cv = torch.randn(2, 5, 8)
        cm = torch.tensor(
            [[True, True, True, False, False], [True, False, False, False, False]]
        )
        state = torch.randn(2, 16)
        masks, _ = subset_masks_batch(cm, 5)
        out = policy(cv, cm, state, masks)
        self.assertEqual(tuple(out["subset_scores"].shape), (2, masks.size(1)))
        self.assertEqual(tuple(out["skip_logit"].shape), (2,))
        selector = MSCActionSelector(tau_skip=0.5)
        selected = selector.select(out["subset_scores"], masks, out["skip_logit"])
        self.assertEqual(selected["subset_mask"].shape, (2, 5))
        self.assertEqual(selected["skip"].shape, (2,))
        self.assertIn(selected["skip"].dtype, (torch.bool,))

    def test_selector_skips_when_no_positive_score(self):
        selector = MSCActionSelector(tau_skip=0.99)
        scores = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
        masks = torch.ones(2, 2, 3, dtype=torch.bool)
        skip_logit = torch.tensor([-5.0, -5.0])
        selected = selector.select(scores, masks, skip_logit)
        self.assertTrue(bool(selected["skip"].all()))


if __name__ == "__main__":
    unittest.main()
