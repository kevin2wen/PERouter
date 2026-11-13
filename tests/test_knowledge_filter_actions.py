from __future__ import annotations

import inspect
import unittest

import torch

from ComP.knowledge.filter import KnowledgeEvidenceFilter


class KnowledgeFilterActionsTest(unittest.TestCase):
    def test_forward_excludes_labels_and_utility(self):
        parameters = set(
            inspect.signature(KnowledgeEvidenceFilter.forward).parameters
        )
        self.assertFalse(
            parameters & {"labels", "targets", "utility", "teacher_utility"}
        )

    def test_skip_and_rank_shapes(self):
        model = KnowledgeEvidenceFilter(4, 3, 3)
        output = model(
            torch.randn(2, 3, 4),
            torch.randn(2, 3),
            torch.rand(2, 3),
            torch.tensor([0, 1]),
            torch.tensor([[0, 1, 1], [1, 0, 0]]),
            torch.tensor([[True, False, True], [True, True, True]]),
        )
        self.assertEqual(tuple(output.action_logits.shape), (2, 4))
        self.assertEqual(tuple(output.candidate_scores.shape), (2, 3))
        self.assertEqual(tuple(output.selected_rank.shape), (2,))
