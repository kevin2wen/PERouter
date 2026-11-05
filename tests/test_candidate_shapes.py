from __future__ import annotations

import unittest

import torch

from ComP.per.types import EvidenceBatch
from tests.helpers import text_availability, tiny_features, tiny_model


class CandidateShapeTest(unittest.TestCase):
    def test_prompt_evidence_and_fused_shapes_are_batch_time_hidden(self):
        model = tiny_model(use_router=True, evidence_dim=1).eval()
        features = tiny_features()
        availability = text_availability()
        evidence = EvidenceBatch(
            ids=[[["e0", "e1"] for _ in range(4)] for _ in range(2)],
            features=torch.randn(2, 4, 2, 1),
            mask=torch.ones(2, 4, 2, dtype=torch.bool),
            source_confidence=torch.full((2, 4, 2), 0.8),
            conflict_score=torch.zeros(2, 4, 2),
        )
        with torch.no_grad():
            output = model(features, availability, evidence=evidence)
        self.assertEqual(tuple(output.logits.shape), (2, 4, 1))
        self.assertEqual(tuple(output.route_logits.shape), (2, 4, 3))
        self.assertEqual(tuple(output.route.shape), (2, 4, 3))
        self.assertEqual(tuple(output.coordinator_weights.shape), (2, 4, 3))
        for name in ("audio", "text", "video"):
            self.assertEqual(tuple(output.prompt_candidates[name].shape), (2, 4, 8))
            self.assertEqual(tuple(output.evidence[name].shape), (2, 4, 8))
            self.assertEqual(tuple(output.fused[name].shape), (2, 4, 8))
            self.assertTrue(torch.isfinite(output.prompt_candidates[name]).all())
            self.assertTrue(torch.isfinite(output.fused[name]).all())
        self.assertTrue(torch.isfinite(output.logits).all())

    def test_router_zeroes_route_on_complete_modalities(self):
        model = tiny_model(use_router=True, evidence_dim=1).eval()
        features = tiny_features()
        availability = torch.zeros(2, 4, 3, dtype=torch.bool)
        availability[:, :, 0] = True
        evidence = EvidenceBatch(
            ids=[[["e"] for _ in range(4)] for _ in range(2)],
            features=torch.randn(2, 4, 1, 1),
            mask=torch.ones(2, 4, 1, dtype=torch.bool),
            source_confidence=torch.ones(2, 4, 1),
            conflict_score=torch.zeros(2, 4, 1),
        )
        with torch.no_grad():
            output = model(features, availability, evidence=evidence)
        self.assertTrue((output.route[:, :, 0] == 0).all())
        self.assertTrue((output.route > 0).any())

    def test_coordinator_on_fixed_and_off_are_all_finite(self):
        for mode in ("on", "fixed", "off"):
            model = tiny_model(
                use_router=False, evidence_dim=1, coordinator_mode=mode
            ).eval()
            with torch.no_grad():
                output = model(tiny_features(), text_availability(), None)
            self.assertTrue(torch.isfinite(output.logits).all(), mode)
            if mode == "off":
                self.assertIsNone(output.coordinator_weights)
            else:
                self.assertEqual(tuple(output.coordinator_weights.shape), (2, 4, 3))
                self.assertTrue(
                    torch.allclose(
                        output.coordinator_weights.sum(dim=-1),
                        torch.ones(2, 4),
                        atol=1e-5,
                    )
                )


if __name__ == "__main__":
    unittest.main()
