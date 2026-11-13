from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SeedDeltaAggregateTest(unittest.TestCase):
    def test_delta_matches_manual_calculation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows_path = root / "rows.csv"
            rows = [
                {"per_mode": "off", "coordinator_mode": "on", "mask_rate": "0.5", "seed": "66", "test_acc": "0.7", "test_f1": "0.6", "test_mae": "1.0", "test_corr": "0.5"},
                {"per_mode": "task_router", "coordinator_mode": "on", "mask_rate": "0.5", "seed": "66", "test_acc": "0.8", "test_f1": "0.7", "test_mae": "0.9", "test_corr": "0.6"},
            ]
            with rows_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            output = root / "out"
            subprocess.run([
                sys.executable, "scripts/aggregate_seed_deltas.py",
                "--rows", str(rows_path), "--output-dir", str(output),
            ], check=True, cwd=Path(__file__).resolve().parents[1])
            with (output / "per_seed_delta.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                result = list(csv.DictReader(handle))
            self.assertAlmostEqual(float(result[0]["delta_acc"]), 0.1)
            self.assertAlmostEqual(float(result[0]["delta_mae"]), -0.1)


if __name__ == "__main__":
    unittest.main()
