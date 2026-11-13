from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "paired_bootstrap.py"
    spec = importlib.util.spec_from_file_location("paired_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PairedBootstrapTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.rows = [
            {"sample_id": "a", "label": "1", "prediction_control": "0.5", "prediction_method": "0.9"},
            {"sample_id": "a", "label": "-1", "prediction_control": "-0.8", "prediction_method": "-0.9"},
            {"sample_id": "b", "label": "1", "prediction_control": "0.2", "prediction_method": "0.8"},
            {"sample_id": "c", "label": "-1", "prediction_control": "0.2", "prediction_method": "-0.7"},
        ]

    def test_metric_delta_is_correct(self):
        self.assertAlmostEqual(
            self.module.compute_metric(self.rows, "mae_delta"),
            -0.5,
        )
        self.assertAlmostEqual(
            self.module.compute_metric(self.rows, "acc_delta"), 0.25
        )

    def test_bootstrap_is_reproducible(self):
        first = self.module.clustered_bootstrap(self.rows, "mae_delta", 100, 66)
        second = self.module.clustered_bootstrap(self.rows, "mae_delta", 100, 66)
        self.assertEqual(first, second)
        self.assertLessEqual(first["ci_low"], first["ci_high"])


if __name__ == "__main__":
    unittest.main()
