from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ComP.knowledge.msc.calibration import (
    brier_score,
    expected_calibration_error,
    fit_calibration,
    fit_threshold,
    load_calibration,
    save_calibration,
)


class MSCCalibrationTest(unittest.TestCase):
    def test_metrics_and_threshold(self):
        probabilities = [0.1, 0.2, 0.8, 0.9]
        labels = [0, 0, 1, 1]
        self.assertLess(expected_calibration_error(probabilities, labels), 0.2)
        self.assertLess(brier_score(probabilities, labels), 0.2)
        threshold = fit_threshold(probabilities, labels)
        self.assertGreaterEqual(threshold["balanced_accuracy"], 0.9)

    def test_calibration_bound_to_scorer_hash(self):
        payload = fit_calibration(
            [0.1, -0.2, 0.3], [0, 0, 1], utility_scorer_hash="abc"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            save_calibration(payload, path)
            loaded = load_calibration(path, expected_scorer_hash="abc")
            self.assertEqual(loaded["utility_scorer_hash"], "abc")
            with self.assertRaises(ValueError):
                load_calibration(path, expected_scorer_hash="different")


if __name__ == "__main__":
    unittest.main()
