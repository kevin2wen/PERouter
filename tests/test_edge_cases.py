from __future__ import annotations

import unittest

import torch

from ComP.per.evidence import FeatureNeighborRetriever
from tests.helpers import text_availability, tiny_features, tiny_model


class EdgeCaseTest(unittest.TestCase):
    def test_single_sample_empty_evidence(self):
        model = tiny_model(use_router=True, evidence_dim=1).eval()
        with torch.no_grad():
            output = model(
                tiny_features(batch_size=1),
                text_availability(batch_size=1),
                evidence=None,
            )
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertTrue((output.route == 0).all())

    def test_all_padding_is_finite(self):
        model = tiny_model(use_router=True, evidence_dim=1).eval()
        sequence = torch.zeros(1, 4, 3, dtype=torch.bool)
        with torch.no_grad():
            output = model(tiny_features(batch_size=1), sequence, None)
        self.assertTrue(torch.isfinite(output.logits).all())

    def test_missing_nonfinite_values_are_not_read(self):
        model = tiny_model(use_router=False, evidence_dim=1).eval()
        features = tiny_features(batch_size=1)
        features["text"][0, 0, 0] = float("nan")
        availability = torch.ones(1, 4, 3, dtype=torch.bool)
        availability[:, :, 1] = False
        with torch.no_grad():
            output = model(features, availability, None)
        self.assertTrue(torch.isfinite(output.logits).all())

    def test_nonfinite_observed_value_fails_loudly(self):
        model = tiny_model(use_router=False, evidence_dim=1).eval()
        features = tiny_features(batch_size=1)
        features["audio"][0, 0, 0] = float("inf")
        with self.assertRaises(FloatingPointError):
            with torch.no_grad():
                model(features, torch.ones(1, 4, 3, dtype=torch.bool), None)

    def test_float64_model_and_features(self):
        model = tiny_model(use_router=False, evidence_dim=1).double().eval()
        features = {name: value.double() for name, value in tiny_features().items()}
        with torch.no_grad():
            output = model(features, text_availability(), None)
        self.assertEqual(output.logits.dtype, torch.float64)
        self.assertTrue(torch.isfinite(output.logits).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_single_sample(self):
        device = torch.device("cuda")
        model = tiny_model(use_router=False, evidence_dim=1, device=device).eval()
        with torch.no_grad():
            output = model(
                tiny_features(batch_size=1, device=device),
                text_availability(batch_size=1, device=device),
                None,
            )
        self.assertEqual(output.logits.device.type, "cuda")
        self.assertTrue(torch.isfinite(output.logits).all())

    def test_feature_retriever_excludes_same_utterance(self):
        retriever = FeatureNeighborRetriever(
            {"audio": 2, "text": 2, "video": 2}, top_k=1
        )
        features = {
            "audio": torch.tensor([[1.0, 0.0]]),
            "text": torch.tensor([[1.0, 0.0]]),
            "video": torch.tensor([[1.0, 0.0]]),
        }
        retriever.add("same", ["u0"], features, torch.ones(1, dtype=torch.bool))
        other = {
            "audio": torch.tensor([[0.0, 1.0]]),
            "text": torch.tensor([[0.0, 1.0]]),
            "video": torch.tensor([[0.0, 1.0]]),
        }
        retriever.add("other", ["u1"], other, torch.ones(1, dtype=torch.bool))
        batch_features = {name: value.unsqueeze(0) for name, value in features.items()}
        availability = torch.tensor([[[True, False, False]]])
        evidence = retriever.retrieve(
            batch_features,
            availability,
            torch.ones(1, 1, dtype=torch.bool),
            ["same"],
            [["u0"]],
        )
        self.assertEqual(evidence.ids[0][0][0], "other:u1")
        self.assertTrue(evidence.mask[0, 0, 0])


if __name__ == "__main__":
    unittest.main()
