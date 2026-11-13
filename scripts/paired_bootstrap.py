from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_rows(path: str):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def clustered_bootstrap(rows, metric: str, replicates: int, seed: int):
    clusters = defaultdict(list)
    for row in rows:
        clusters[str(row["sample_id"])].append(row)
    cluster_ids = sorted(clusters)
    rng = random.Random(seed)
    observed = compute_metric(rows, metric)
    samples = []
    for _ in range(replicates):
        selected = [rng.choice(cluster_ids) for _ in cluster_ids]
        resampled = [row for cluster in selected for row in clusters[cluster]]
        samples.append(compute_metric(resampled, metric))
    array = np.asarray(samples, dtype=np.float64)
    return {
        "metric": metric,
        "observed": observed,
        "ci_low": float(np.quantile(array, 0.025)),
        "ci_high": float(np.quantile(array, 0.975)),
        "replicates": replicates,
        "clusters": len(cluster_ids),
    }


def compute_metric(rows, metric: str) -> float:
    labels = np.asarray([float(row["label"]) for row in rows])
    control = np.asarray([float(row["prediction_control"]) for row in rows])
    method = np.asarray([float(row["prediction_method"]) for row in rows])
    if metric == "mae_delta":
        return float(np.mean(np.abs(method - labels) - np.abs(control - labels)))
    if metric == "acc_delta":
        return float(
            np.mean((method > 0) == (labels > 0))
            - np.mean((control > 0) == (labels > 0))
        )
    raise ValueError(f"unknown metric: {metric}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=66)
    args = parser.parse_args()
    rows = read_rows(args.rows)
    if not rows:
        raise ValueError("prediction rows are empty")
    required = {"sample_id", "label", "prediction_control", "prediction_method"}
    if not required.issubset(rows[0]):
        raise ValueError(f"missing columns: {sorted(required - set(rows[0]))}")
    reports = [
        clustered_bootstrap(rows, metric, args.replicates, args.seed)
        for metric in ("mae_delta", "acc_delta")
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
