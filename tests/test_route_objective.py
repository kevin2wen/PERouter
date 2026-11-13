from __future__ import annotations

import unittest

import torch

from ComP.per.losses import route_distillation_loss


class RouteObjectiveTest(unittest.TestCase):
    def test_bce_and_smooth_l1_are_finite_and_differentiable(self):
        logits = torch.randn(2, 3, 3, requires_grad=True)
        target = torch.full((2, 3, 3), 0.5)
        valid = torch.ones(2, 3, 3, dtype=torch.bool)
        sequence = torch.tensor([[True, True, False], [True, True, True]])
        for objective in ("bce", "smooth_l1"):
            loss = route_distillation_loss(
                logits, target, valid, sequence, objective=objective
            )
            self.assertTrue(torch.isfinite(loss))
            gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertTrue((gradient[0, 2].abs() == 0).all())

    def test_invalid_route_objective_fails(self):
        logits = torch.randn(1, 1, 3)
        target = torch.full_like(logits, 0.5)
        valid = torch.ones_like(logits, dtype=torch.bool)
        with self.assertRaises(ValueError):
            route_distillation_loss(logits, target, valid, objective="bad")


if __name__ == "__main__":
    unittest.main()
