import unittest

import torch

from ComP.per.evidence_oracle import (
    apply_selected_ranks,
    isolate_rank_evidence,
    select_confidence_ranks,
    select_strong_ranks,
    validate_evidence_oracle_split,
)
from ComP.per.types import EvidenceBatch


def make_evidence():
    return EvidenceBatch(
        ids=[[["a:0", "b:0"]]],
        features=torch.tensor([[[[1.0, 0.0], [9.0, 0.0]]]]),
        mask=torch.tensor([[[True, True]]]),
        source_confidence=torch.tensor([[[0.2, 0.8]]]),
        conflict_score=torch.zeros(1, 1, 2),
    )


class EvidenceOracleV2Test(unittest.TestCase):
    def test_strong_oracle_rejects_test_split(self):
        with self.assertRaises(PermissionError):
            validate_evidence_oracle_split("oracle_strong", "test")
        validate_evidence_oracle_split("oracle_strong", "validation")
        validate_evidence_oracle_split("oracle_weak", "test")

    def test_confidence_oracle_selects_highest_confidence(self):
        reference = make_evidence()
        ranks = select_confidence_ranks(reference)
        selected = apply_selected_ranks(reference, ranks)
        self.assertEqual(ranks.tolist(), [[1]])
        self.assertTrue(selected.mask.all())
        self.assertEqual(selected.features.shape, reference.features.shape)
        self.assertTrue(torch.equal(selected.features[..., 0], torch.full_like(selected.features[..., 0], 9.0)))
        self.assertTrue(torch.equal(selected.features[..., 1], torch.zeros_like(selected.features[..., 1])))

    def test_strong_oracle_selects_best_utilities(self):
        utilities = torch.tensor([[[[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]]]])
        availability = torch.tensor([[[False, True, False]]])
        sequence_mask = torch.tensor([[True]])
        evidence_mask = torch.tensor([[[True, True]]])
        ranks, valid = select_strong_ranks(
            utilities, availability, sequence_mask, evidence_mask
        )
        self.assertTrue(valid.all())
        self.assertEqual(ranks.tolist(), [[1]])
        selected = apply_selected_ranks(make_evidence(), ranks)
        self.assertTrue(torch.equal(selected.features[..., 0], torch.full_like(selected.features[..., 0], 9.0)))
        self.assertTrue(torch.equal(selected.features[..., 1], torch.zeros_like(selected.features[..., 1])))

    def test_isolate_rank_preserves_single_rank_only(self):
        isolated = isolate_rank_evidence(make_evidence(), 1)
        self.assertFalse(isolated.mask[..., 0].any())
        self.assertTrue(isolated.mask[..., 1].all())
        self.assertEqual(isolated.ids[0][0][1], "b:0")
