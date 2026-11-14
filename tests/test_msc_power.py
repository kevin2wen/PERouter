from __future__ import annotations

import unittest

from ComP.knowledge.msc.power import msc_power_report


class MSCPowerTest(unittest.TestCase):
    def test_report(self):
        report = msc_power_report(
            {
                "utility_mae": [0.03, 0.02, 0.04, 0.01, 0.03],
                "top1_regret_reduction": [0.01, 0.02, 0.0, 0.015, 0.02],
            }
        )
        self.assertIn("utility_mae", report)
        self.assertIn("required_clusters", report["utility_mae"])
        self.assertGreater(report["utility_mae"]["clusters"], 0)


if __name__ == "__main__":
    unittest.main()
