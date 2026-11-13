from __future__ import annotations

import unittest

import torch

from ComP.per.losses import build_route_target, route_distillation_loss


class RouteLossEvidenceMaskTest(unittest.TestCase):
    def test_no_evidence_positions_have_zero_gradient(self):
        logits = torch.randn(1, 2, 3, requires_grad=True)
        target = torch.full((1, 2, 3), 0.8)
        missing = torch.ones(1, 2, 3, dtype=torch.bool)
        evidence = torch.zeros(1, 2, 3, dtype=torch.bool)
        evidence[:, 0, 0] = True
        loss = route_distillation_loss(
            logits, target, missing, evidence_available=evidence
        )
        loss.backward()
        self.assertTrue((logits.grad[~evidence].abs() == 0).all())
        self.assertTrue((logits.grad[evidence].abs() > 0).any())

    def test_target_is_neutral_without_evidence(self):
        prompt = torch.zeros(1, 1, 3)
        candidate = torch.ones(1, 1, 3)
        missing = torch.ones(1, 1, 3, dtype=torch.bool)
        evidence = torch.zeros(1, 1, 3, dtype=torch.bool)
        target = build_route_target(
            prompt,
            candidate,
            missing,
            torch.tensor(1.0),
            torch.ones(1, 1, 3),
            evidence_available=evidence,
        )
        torch.testing.assert_close(target, torch.full_like(target, 0.5))

    def test_old_interface_remains_valid(self):
        logits = torch.zeros(1, 1, 3, requires_grad=True)
        target = torch.full((1, 1, 3), 0.5)
        loss = route_distillation_loss(
            logits, target, torch.ones(1, 1, 3, dtype=torch.bool)
        )
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
