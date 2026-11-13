from __future__ import annotations

import unittest

from ComP.knowledge.power_analysis import (
    cluster_mde,
    power_report,
    required_clusters,
)


class PowerAnalysisTest(unittest.TestCase):
    def test_cluster_mde_shrinks_with_clusters(self):
        small = cluster_mde([-0.02, -0.01, -0.03, -0.02, -0.01])
        large = cluster_mde(
            [-0.02, -0.01, -0.03, -0.02, -0.01] * 4
        )
        self.assertLess(large["mde"], small["mde"])
        self.assertEqual(large["clusters"], 20.0)

    def test_required_clusters_grows_for_small_effect(self):
        few = required_clusters(0.03, 0.03)
        many = required_clusters(0.03, 0.01)
        self.assertGreaterEqual(many, few)

    def test_power_report_flags_underpowered_target(self):
        report = power_report(
            {"mae": [-0.05, 0.03, -0.02, 0.04, -0.01]},
            {"mae": 0.02},
        )
        self.assertIn("mae", report)
        self.assertFalse(report["mae"]["detects_target"])
        self.assertGreater(report["mae"]["required_clusters"], 5)


if __name__ == "__main__":
    unittest.main()
