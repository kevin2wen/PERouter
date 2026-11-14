from __future__ import annotations

import unittest

import torch

from ComP.knowledge.msc.subsets import (
    nonempty_subset_masks,
    valid_subset_masks,
)


class MSCSubsetsTest(unittest.TestCase):
    def test_five_candidates_have_31_nonempty_subsets(self):
        self.assertEqual(len(nonempty_subset_masks(5)), 31)
        self.assertEqual(nonempty_subset_masks(5), list(range(1, 32)))

    def test_valid_subset_masks_use_only_valid_positions(self):
        valid = torch.tensor([True, False, True, True, False])
        masks = valid_subset_masks(valid, 5)
        self.assertEqual(masks.size(0), 2 ** 3 - 1)
        self.assertFalse(bool(masks[:, 1].any()))
        self.assertFalse(bool(masks[:, 4].any()))
        self.assertTrue(bool(masks[:, 0].any()))

    def test_empty_candidate_set_yields_no_subsets(self):
        masks = valid_subset_masks(torch.zeros(5, dtype=torch.bool), 5)
        self.assertEqual(masks.numel(), 0)


if __name__ == "__main__":
    unittest.main()
