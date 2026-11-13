from __future__ import annotations

import unittest

import torch

from ComP.per.types import EvidenceBatch


class EvidenceBatchContractTest(unittest.TestCase):
    def test_non_finite_features_are_rejected(self):
        evidence = EvidenceBatch(
            ids=[[["a"]]],
            features=torch.tensor([[[[float("nan")]]]]),
            mask=torch.tensor([[[True]]]),
            source_confidence=torch.tensor([[[0.5]]]),
        )
        with self.assertRaises(FloatingPointError):
            evidence.validate()

    def test_confidence_and_conflict_ranges_are_enforced(self):
        evidence = EvidenceBatch(
            ids=[[["a"]]],
            features=torch.zeros(1, 1, 1, 2),
            mask=torch.tensor([[[True]]]),
            source_confidence=torch.tensor([[[1.1]]]),
            conflict_score=torch.tensor([[[-0.1]]]),
        )
        with self.assertRaises(ValueError):
            evidence.validate()
