from __future__ import annotations

import unittest

import torch

from ComP.per.losses import build_route_target, route_distillation_loss


class RouteMaskingTest(unittest.TestCase):
    def test_loss_only_uses_missing_and_sequence_valid_positions(self):
        availability = torch.tensor(
            [
                [[True, False, False], [True, True, True], [False, True, True]],
                [[True, True, False], [False, False, True], [True, True, True]],
            ],
            dtype=torch.bool,
        )
        missing = ~availability
        sequence_mask = torch.tensor(
            [[True, True, False], [True, False, True]]
        )
        target = torch.full((2, 3, 3), 0.5)
        target[missing] = 0.9
        logits_a = torch.zeros(2, 3, 3, requires_grad=True)
        loss_a = route_distillation_loss(
            logits_a, target, missing, sequence_mask
        )
        changed = target.clone()
        changed[~missing] = 0.0
        loss_b = route_distillation_loss(
            logits_a, changed, missing, sequence_mask
        )
        torch.testing.assert_close(loss_a, loss_b)
        loss_a.backward()
        self.assertTrue((logits_a.grad[~missing] == 0).all())

    def test_no_missing_positions_returns_differentiable_zero(self):
        availability = torch.ones(1, 2, 3, dtype=torch.bool)
        logits = torch.randn(1, 2, 3, requires_grad=True)
        loss = route_distillation_loss(
            logits, torch.full_like(logits, 0.5), ~availability
        )
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue((logits.grad == 0).all())

    def test_route_target_is_detached_and_confidence_shrinks_to_half(self):
        prompt = torch.zeros(1, 1, 3, requires_grad=True)
        evidence = torch.ones(1, 1, 3, requires_grad=True)
        missing = torch.ones(1, 1, 3, dtype=torch.bool)
        confidence = torch.zeros(1, 1, 3)
        target = build_route_target(
            prompt,
            evidence,
            missing,
            torch.tensor(1.0),
            confidence,
        )
        self.assertFalse(target.requires_grad)
        torch.testing.assert_close(target, torch.full_like(target, 0.5))


if __name__ == "__main__":
    unittest.main()
