from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.knowledge.cluster_stats import per_cluster_metric_delta
from ComP.knowledge.power_analysis import cluster_mde, required_clusters

COMPARISONS = (
    ("card", "empty"),
    ("card", "random_card"),
    ("card", "shuffled_card"),
    ("card", "source_only"),
    ("oracle_card", "empty"),
)
TARGETS = {"mae": 0.02, "acc": 0.01}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True,
                        help="mode=path/to/per_sample.csv")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in args.input:
        mode, path = item.split("=", 1)
        for row in csv.DictReader(Path(path).open(newline="")):
            row["mode"] = mode
            row["label"] = float(row["label"])
            row["prediction"] = float(row["prediction"])
            row["abs_error"] = float(row["abs_error"])
            rows.append(row)
    available = {row["mode"] for row in rows}
    metric_rows = []
    required_rows = []
    report = {}
    for comparison, baseline in COMPARISONS:
        if comparison not in available or baseline not in available:
            continue
        for metric in ("mae", "acc"):
            clusters = per_cluster_metric_delta(
                rows, comparison, baseline, metric=metric
            )
            stats = cluster_mde(list(clusters.values()))
            target = TARGETS[metric]
            needed = required_clusters(stats["sd"], target)
            entry = {
                "comparison": comparison,
                "baseline": baseline,
                "metric": metric,
                "target_effect": target,
                "required_clusters": needed,
                "current_clusters": len(clusters),
                "cluster_multiplier": needed / max(1, len(clusters)),
                **stats,
            }
            entry["detects_target"] = bool(stats["mde"] <= target)
            metric_rows.append(entry)
            required_rows.append(
                {
                    "comparison": comparison,
                    "baseline": baseline,
                    "metric": metric,
                    "target_effect": target,
                    "current_clusters": len(clusters),
                    "required_clusters": needed,
                }
            )
            report[f"{comparison}_vs_{baseline}_{metric}"] = entry
    with (output / "mde_by_metric.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)
    with (output / "required_clusters.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(required_rows[0].keys()))
        writer.writeheader()
        writer.writerows(required_rows)
    payload = {
        "targets": TARGETS,
        "alpha": 0.05,
        "power": 0.80,
        "comparisons": report,
        "note": "Normal approximation on cluster-level deltas; approximate for 10 clusters",
    }
    (output / "mde_analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    for entry in metric_rows:
        print(
            f"{entry['comparison']:>12} vs {entry['baseline']:<13} {entry['metric']} "
            f"sd={entry['sd']:.4f} MDE={entry['mde']:.4f} target={entry['target_effect']:.3f} "
            f"detects={entry['detects_target']} need_clusters={entry['required_clusters']:.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
