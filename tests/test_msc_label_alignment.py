from __future__ import annotations

import unittest

import torch

from ComP.knowledge.msc.audit import audit_single_labels


class MSCLabelAlignmentTest(unittest.TestCase):
    def test_skip_and_action_labels(self):
        utility = torch.tensor([[0.5, -0.2, 0.0], [-0.1, -0.3, -0.2]])
        mask = torch.ones(2, 3, dtype=torch.bool)
        cache = {
            "single_utility": utility,
            "candidate_mask": mask,
            "skip_label": torch.tensor([0.0, 1.0]),
            "target_action": torch.tensor([1, 0]),
            "target_modality": torch.tensor([[1, 0, 0], [0, 1, 0]]),
            "missing_pattern": torch.tensor([1, 2]),
        }
        report = audit_single_labels(cache)
        self.assertEqual(report["skip_label_mismatch"], 0)
        self.assertEqual(report["target_action_mismatch"], 0)
        self.assertAlmostEqual(report["query_any_positive_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
