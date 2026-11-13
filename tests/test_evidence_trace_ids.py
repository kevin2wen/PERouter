from __future__ import annotations

import unittest

import torch

from ComP.per.evidence import EvidenceBatch
from ComP.per.evidence_audit import evidence_trace_rows


class EvidenceTraceIdsTest(unittest.TestCase):
    def test_trace_contains_ids_without_text(self):
        evidence = EvidenceBatch(
            ids=[[["neighbor:utt", ""]]],
            features=torch.ones(1, 1, 2, 1),
            mask=torch.tensor([[[True, False]]]),
            source_confidence=torch.tensor([[[0.7, 0.0]]]),
            conflict_score=torch.zeros(1, 1, 2),
        )
        rows = evidence_trace_rows(
            evidence,
            torch.tensor([[[False, True, True]]], dtype=torch.bool),
            torch.ones(1, 1, dtype=torch.bool),
            split="validation",
            sample_ids=["query"],
            utterance_ids=[["query_utt"]],
            evidence_mode="feature",
            mask_rate=0.5,
            seed=66,
            sampler_version="categorical",
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["target_modality"] == "audio" for row in rows))
        self.assertEqual(rows[0]["neighbor_sample_id"], "neighbor")
        self.assertEqual(rows[0]["neighbor_utterance_id"], "utt")
        self.assertTrue(rows[0]["evidence_available"])
        self.assertFalse(rows[1]["rank_available"])
        self.assertTrue(all("raw_text" not in row for row in rows))


if __name__ == "__main__":
    unittest.main()
