from __future__ import annotations

import unittest

import torch

from ComP.knowledge.statistics import (
    cluster_bootstrap_mean,
    cluster_deltas,
    holm_adjust,
)
from ComP.knowledge.utility import oracle_utility, regression_benefit


def row(mode, dialogue, utterance, error, label=1.0, prediction=0.0):
    return {
        "mode": mode,
        "sample_id": dialogue,
        "dialogue_id": dialogue,
        "utterance_id": utterance,
        "abs_error": error,
        "label": label,
        "prediction": prediction,
    }


class StatisticsTest(unittest.TestCase):
    def test_cluster_deltas_pair_modes(self):
        rows = [
            row("card", "d1", "u1", 1.0),
            row("random_card", "d1", "u1", 1.5),
            row("card", "d2", "u2", 2.0),
            row("random_card", "d2", "u2", 2.5),
        ]
        grouped = cluster_deltas(rows, "random_card", "card")
        self.assertEqual(sorted(grouped), ["d1", "d2"])
        self.assertAlmostEqual(grouped["d1"][0], -0.5)

    def test_bootstrap_negative_effect_has_negative_upper_bound(self):
        grouped = {
            f"d{index}": [-0.5 - 0.01 * index, -0.4 + 0.01 * index]
            for index in range(10)
        }
        stats = cluster_bootstrap_mean(grouped, n_boot=500, seed=1)
        self.assertLess(stats["ci_high"], 0.0)
        self.assertLess(stats["ci_low"], stats["delta"])

    def test_holm_adjust_is_monotone_and_bounded(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))
        self.assertGreaterEqual(adjusted[1], adjusted[0])


class UtilityTest(unittest.TestCase):
    def test_regression_benefit_sign(self):
        self.assertGreater(regression_benefit(0.5, 1.0), 0.0)
        self.assertLess(regression_benefit(1.5, 1.0), 0.0)
        self.assertEqual(regression_benefit(1.0, 1.0), 0.0)

    def test_accuracy_oracle_prefers_correct_prediction(self):
        labels = torch.tensor([1.0, -1.0])
        skip = torch.tensor([-1.0, -1.0])
        correct = torch.tensor([1.0, -1.0])
        wrong = torch.tensor([0.1, 0.2])
        good = oracle_utility(
            correct, labels, skip, task="regression", objective="acc"
        )
        bad = oracle_utility(
            wrong, labels, skip, task="regression", objective="acc"
        )
        self.assertTrue(bool((good > bad).all()))

    def test_composite_penalises_sign_flip(self):
        labels = torch.tensor([1.0])
        skip = torch.tensor([1.0])
        same_sign = torch.tensor([0.9])
        flipped = torch.tensor([-0.9])
        good = oracle_utility(
            same_sign, labels, skip, task="regression", objective="composite"
        )
        bad = oracle_utility(
            flipped, labels, skip, task="regression", objective="composite"
        )
        self.assertGreater(float(good), float(bad))


if __name__ == "__main__":
    unittest.main()
