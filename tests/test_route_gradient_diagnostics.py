from __future__ import annotations

import unittest

import torch

from ComP.per.gradient_diagnostics import (
    component_gradient_norms,
    parameter_grad_norm,
)


class RouteGradientDiagnosticsTest(unittest.TestCase):
    def test_component_ratio_and_parameter_norm_are_finite(self):
        parameter = torch.tensor(1.0, requires_grad=True)
        route_loss = parameter.pow(2)
        task_loss = (parameter - 3.0).pow(2)
        result = component_gradient_norms(route_loss, task_loss, [parameter])
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in result.values()))
        self.assertGreater(result["route_component_grad_norm"], 0)
        self.assertGreater(result["task_component_grad_norm"], 0)
        (route_loss + task_loss).backward()
        self.assertGreater(parameter_grad_norm([parameter]), 0)


if __name__ == "__main__":
    unittest.main()
