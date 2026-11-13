from __future__ import annotations

import unittest

import torch

from ComP.per.teacher_diagnostics import compute_route_quality, route_quality_groups, valid_route_mask


class RouteMetricsMaskTest(unittest.TestCase):
    def test_padding_observed_and_unavailable_evidence_are_excluded(self):
        availability = torch.tensor(
            [[[False, False, True], [True, True, True], [True, True, True]]],
            dtype=torch.bool,
        )
        sequence = torch.tensor([[True, True, False]])
        evidence = torch.ones(1, 3, 3, dtype=torch.bool)
        evidence[0, 0, 0] = False
        valid = valid_route_mask(availability, sequence, evidence)
        self.assertTrue(valid[0, 0, 1])
        self.assertFalse(valid[0, 0, 0])
        self.assertFalse(valid[0, 1, 0])
        self.assertFalse(valid[0, 2].any())

    def test_correlation_and_target_conflicts(self):
        availability = torch.tensor([[[False, False, True]]], dtype=torch.bool)
        sequence = torch.tensor([[True]])
        evidence = torch.ones(1, 1, 3, dtype=torch.bool)
        route = torch.tensor([[[0.9, 0.1, 0.0]]])
        target = torch.tensor([[[0.8, 0.2, 0.5]]])
        gap = torch.tensor([[[0.5, -0.5, 0.0]]])
        stats = compute_route_quality(
            route, target, gap, availability, sequence, evidence
        )
        self.assertEqual(stats["count"], 2)
        self.assertAlmostEqual(stats["route_target_pearson"], 1.0, places=5)
        self.assertAlmostEqual(stats["route_target_spearman"], 1.0, places=5)


    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_group_diagnostics_run_on_cuda(self):
        device = torch.device("cuda:0")
        availability = torch.tensor(
            [[[False, False, True]]], dtype=torch.bool, device=device
        )
        sequence = torch.tensor([[True]], dtype=torch.bool, device=device)
        evidence = torch.ones(1, 1, 3, dtype=torch.bool, device=device)
        route = torch.tensor([[[0.9, 0.1, 0.0]]], device=device)
        target = torch.tensor([[[0.8, 0.2, 0.5]]], device=device)
        gap = torch.tensor([[[0.5, -0.5, 0.0]]], device=device)
        rows = route_quality_groups(
            route, target, gap, availability, sequence, evidence
        )
        self.assertTrue(rows)


if __name__ == "__main__":
    unittest.main()
