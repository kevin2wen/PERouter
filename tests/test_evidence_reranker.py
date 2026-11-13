import unittest
import torch

from ComP.per.evidence_reranker import EvidenceReranker


class EvidenceRerankerTest(unittest.TestCase):
    def test_output_shape_and_skip_slot(self):
        model = EvidenceReranker(candidate_dim=8, prompt_dim=6, top_k=4)
        logits = model(
            torch.randn(3, 4, 8),
            torch.randn(3, 6),
            torch.rand(3, 4),
            torch.tensor([0, 1, 2]),
            torch.randint(0, 2, (3, 3)),
            torch.tensor([[1, 1, 0, 1], [1, 1, 1, 1], [0, 0, 1, 0]], dtype=torch.bool),
        )
        self.assertEqual(tuple(logits.shape), (3, 5))

    def test_unavailable_ranks_are_masked(self):
        model = EvidenceReranker(candidate_dim=4, prompt_dim=4, top_k=3)
        logits = model(
            torch.randn(1, 3, 4), torch.randn(1, 4), torch.rand(1, 3),
            torch.tensor([1]), torch.zeros(1, 3, dtype=torch.long),
            torch.tensor([[True, False, True]]),
        )
        self.assertLess(float(logits[0, 2]), -1e20)
