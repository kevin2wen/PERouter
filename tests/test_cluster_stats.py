from __future__ import annotations

import unittest

from ComP.knowledge.cluster_stats import (
    metric_value,
    paired_cluster_bootstrap,
    per_cluster_metric_delta,
    wild_cluster_bootstrap,
)


def row(mode, cluster, utterance, label, prediction):
    return {
        "mode": mode,
        "sample_id": cluster,
        "dialogue_id": cluster,
        "utterance_id": utterance,
        "label": label,
        "prediction": prediction,
        "abs_error": abs(prediction - label),
    }


class ClusterStatsTest(unittest.TestCase):
    def test_metric_values(self):
        labels = [1.0, -1.0, 1.0]
        predictions = [0.5, -0.5, -0.5]
        self.assertAlmostEqual(metric_value(labels, predictions, "acc"), 2 / 3)
        self.assertAlmostEqual(
            metric_value(labels, predictions, "mae"), (0.5 + 0.5 + 1.5) / 3
        )
        self.assertGreater(metric_value(labels, predictions, "corr"), 0.0)

    def test_paired_bootstrap_detects_negative_shift(self):
        rows = []
        for cluster in ("d1", "d2", "d3"):
            for index in range(4):
                rows.append(row("base", cluster, f"{cluster}_{index}", 1.0, 1.0 + index))
                rows.append(row("comp", cluster, f"{cluster}_{index}", 1.0, 1.0))
        stats = paired_cluster_bootstrap(
            rows, "comp", "base", metric="mae", n_boot=200, seed=1
        )
        self.assertLess(stats["ci_high"], 0.0)
        self.assertEqual(stats["clusters"], 3.0)

    def test_wild_and_loo(self):
        clusters = {"d1": -0.5, "d2": -0.3, "d3": -0.7}
        wild = wild_cluster_bootstrap(clusters, n_boot=500, seed=2)
        self.assertLess(wild["wild_ci_high"], 0.0)
        self.assertAlmostEqual(wild["raw_delta"], -0.5)
        self.assertLess(wild["wild_p_value"], 0.05)
        rows = []
        for cluster, delta in clusters.items():
            rows.append(row("base", cluster, f"{cluster}_0", 1.0, 1.0))
            rows.append(row("comp", cluster, f"{cluster}_0", 1.0, 1.0 + delta))
        per_cluster = per_cluster_metric_delta(rows, "comp", "base", metric="mae")
        self.assertEqual(set(per_cluster), {"d1", "d2", "d3"})


if __name__ == "__main__":
    unittest.main()
