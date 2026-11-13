from __future__ import annotations

import unittest

import torch

from ComP.per.model import PERStudent
from ComP.per.types import EvidenceBatch
from tests.helpers import text_availability, tiny_args, tiny_features


class FixedSourceRouterTest(unittest.TestCase):
    def test_fixed_router_uses_constant_weight_only_on_missing_evidence_positions(self):
        args = tiny_args()
        args.per_mode = "fixed"
        args.fixed_source_weight = 0.5
        model = PERStudent(
            args,
            5,
            6,
            7,
            evidence_dim=1,
            top_k=2,
            task="regression",
            use_router=True,
            fix_private_branch=True,
        ).eval()
        availability = text_availability()
        evidence = EvidenceBatch(
            ids=[[["e0", "e1"] for _ in range(4)] for _ in range(2)],
            features=torch.randn(2, 4, 2, 1),
            mask=torch.ones(2, 4, 2, dtype=torch.bool),
            source_confidence=torch.full((2, 4, 2), 0.8),
            conflict_score=torch.zeros(2, 4, 2),
        )
        with torch.no_grad():
            output = model(tiny_features(), availability, evidence)
        self.assertIsNone(model.router)
        self.assertTrue(torch.allclose(
            output.route[availability],
            torch.zeros_like(output.route[availability]),
        ))
        self.assertTrue(torch.allclose(
            output.route[~availability],
            torch.full_like(output.route[~availability], 0.5),
        ))


if __name__ == "__main__":
    unittest.main()
