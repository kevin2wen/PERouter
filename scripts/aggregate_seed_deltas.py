from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(Path(args.rows).open(encoding="utf-8")))
    baseline = {
        (row["mask_rate"], row["seed"]): row
        for row in rows
        if row["per_mode"] == "off" and row["coordinator_mode"] == "on"
    }
    output_rows = []
    for row in rows:
        key = (row["mask_rate"], row["seed"])
        if row["per_mode"] == "off" or key not in baseline:
            continue
        base = baseline[key]
        output = {
            "per_mode": row["per_mode"],
            "coordinator_mode": row["coordinator_mode"],
            "mask_rate": row["mask_rate"],
            "seed": row["seed"],
        }
        for metric in ("acc", "f1", "mae", "corr"):
            output[f"delta_{metric}"] = (
                float(row.get(f"test_{metric}", "nan"))
                - float(base.get(f"test_{metric}", "nan"))
            )
        output_rows.append(output)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_rows:
        with (output_dir / "per_seed_delta.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
    grouped = defaultdict(list)
    for row in output_rows:
        grouped[(row["per_mode"], row["coordinator_mode"], row["mask_rate"])].append(row)
    stats = []
    for key, values in sorted(grouped.items()):
        result = {
            "per_mode": key[0],
            "coordinator_mode": key[1],
            "mask_rate": key[2],
            "runs": len(values),
        }
        for metric in ("acc", "f1", "mae", "corr"):
            series = [row[f"delta_{metric}"] for row in values]
            series = [value for value in series if math.isfinite(value)]
            mean = sum(series) / len(series) if series else 0.0
            variance = (
                sum((value - mean) ** 2 for value in series) / (len(series) - 1)
                if len(series) > 1 else 0.0
            )
            result[f"delta_{metric}_mean"] = mean
            result[f"delta_{metric}_std"] = math.sqrt(variance)
        stats.append(result)
    with (output_dir / "per_seed_delta_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
