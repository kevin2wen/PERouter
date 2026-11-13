from __future__ import annotations

import unittest

import torch

from ComP.per.evidence import EvidenceBatch
from ComP.per.evidence_audit import compute_evidence_coverage


class EvidenceCoverageTest(unittest.TestCase):
    def test_coverage_excludes_padding_and_complete_modalities(self):
        evidence = EvidenceBatch(
            ids=[[["s:u", ""], ["", ""], ["", ""]]],
            features=torch.ones(1, 3, 2, 1),
            mask=torch.tensor([[[True, False], [False, False], [False, False]]]),
            source_confidence=torch.tensor([[[0.8, 0.0], [0.0, 0.0], [0.0, 0.0]]]),
            conflict_score=torch.zeros(1, 3, 2),
        )
        availability = torch.tensor(
            [[[False, False, True], [True, True, True], [False, False, True]]],
            dtype=torch.bool,
        )
        sequence = torch.tensor([[True, True, False]])
        result = compute_evidence_coverage(evidence, availability, sequence)
        # The same evidence row is shared by both missing target modalities.
        self.assertEqual(result["coverage_all_numerator"], 2.0)
        self.assertEqual(result["coverage_all_denominator"], 2.0)
        self.assertEqual(result["coverage_all_fraction"], 1.0)
        self.assertEqual(result["coverage_audio_denominator"], 1.0)
        self.assertEqual(result["coverage_text_denominator"], 1.0)


if __name__ == "__main__":
    unittest.main()
