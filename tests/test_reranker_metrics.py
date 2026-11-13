import unittest
import torch

from ComP.per.reranker_metrics import (
    action_metrics,
    dialogue_folds,
    listwise_loss,
    pairwise_loss,
    utility_regression_loss,
)


class RerankerMetricsTest(unittest.TestCase):
    def test_listwise_and_pairwise_are_finite(self):
        logits = torch.randn(2, 4, requires_grad=True)
        targets = torch.tensor([0, 2])
        mask = torch.tensor([[True, True, True], [True, True, True]])
        self.assertTrue(torch.isfinite(listwise_loss(logits, targets)))
        self.assertTrue(torch.isfinite(pairwise_loss(logits, targets, mask)))

    def test_utility_loss_ignores_unavailable_rank(self):
        logits = torch.zeros(1, 4, requires_grad=True)
        targets = torch.zeros(1, 4)
        mask = torch.tensor([[True, False, True]])
        loss = utility_regression_loss(logits, targets, mask)
        self.assertEqual(float(loss), 0.0)

    def test_dialogue_folds_do_not_split_dialogue(self):
        ids = ["a", "a", "b", "b", "c", "d", "e"]
        folds = dialogue_folds(ids, 3, 66)
        dialogue_sets = [{ids[index] for index in group} for group in folds]
        for left in range(len(dialogue_sets)):
            for right in range(left + 1, len(dialogue_sets)):
                self.assertFalse(dialogue_sets[left] & dialogue_sets[right])
        self.assertEqual(set().union(*dialogue_sets), set(ids))
        self.assertEqual(sorted(index for group in folds for index in group), list(range(len(ids))))

    def test_action_metrics(self):
        logits = torch.tensor([[0.0, 2.0, 1.0], [3.0, 0.0, 1.0]])
        delta = torch.tensor([[0.0, 0.4, 0.2], [0.0, 0.3, 0.1]])
        mask = torch.tensor([[True, True], [True, True]])
        result = action_metrics(logits, delta, mask, torch.tensor([1, 0]))
        self.assertAlmostEqual(result["top1_action_agreement"], 1.0)
