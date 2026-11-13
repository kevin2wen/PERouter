from __future__ import annotations

import unittest

import torch

from ComP.per.fusion_diagnostics import FusionDiagnosticsAccumulator
from ComP.per.types import EvidenceBatch, PERBatch, PEROutput


class FusionDiagnosticsTest(unittest.TestCase):
    def test_route_and_coordinator_are_compared_only_on_active_missing_evidence(self):
        batch = PERBatch(
            features={},
            availability=torch.tensor(
                [[[True, False, True], [True, True, True]]], dtype=torch.bool
            ),
            sequence_mask=torch.tensor([[True, True]]),
            labels=torch.tensor([[1.0, 1.0]]),
            sample_ids=["s"],
            utterance_ids=[["u0", "u1"]],
            evidence=EvidenceBatch(
                ids=[[["e", ""], ["", ""]]],
                features=torch.ones(1, 2, 2, 1),
                mask=torch.tensor([[[True, False], [False, False]]]),
                source_confidence=torch.ones(1, 2, 2),
                conflict_score=torch.zeros(1, 2, 2),
            ),
        )
        output = PEROutput(
            logits=torch.zeros(1, 2, 1),
            route_logits=torch.zeros(1, 2, 3),
            route=torch.tensor([[[0.0, 0.8, 0.0], [0.2, 0.2, 0.2]]]),
            prompt_candidates={},
            evidence={},
            fused={},
            coordinator_weights=torch.tensor(
                [[[0.6, 0.2, 0.2], [0.4, 0.4, 0.2]]]
            ),
        )
        accumulator = FusionDiagnosticsAccumulator()
        accumulator.add(output, batch)
        result = accumulator.summary()
        self.assertEqual(result["fusion_count"], 1.0)
        self.assertEqual(result["fusion_pattern_010_count"], 1.0)
        self.assertTrue(torch.isfinite(torch.tensor(result["fusion_coordinator_entropy_mean"])))


if __name__ == "__main__":
    unittest.main()
