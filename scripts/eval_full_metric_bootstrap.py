from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.knowledge.cluster_stats import (
    metric_value,
    paired_cluster_bootstrap,
    per_cluster_metric_delta,
    wild_cluster_bootstrap,
)
from ComP.knowledge.statistics import holm_adjust

COMPARISONS = (
    ("card", "empty"),
    ("confidence", "empty"),
    ("random_action", "empty"),
    ("confidence", "random_action"),
    ("card", "random_card"),
    ("card", "shuffled_card"),
    ("card", "source_only"),
    ("listwise_filter", "confidence"),
    ("listwise_filter", "random_action"),
    ("pairwise_filter", "confidence"),
    ("pairwise_filter", "random_action"),
    ("oracle_card", "empty"),
)
METRICS = ("mae", "acc", "f1", "corr")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True,
                        help="mode=path/to/per_sample.csv (repeatable)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=66)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in args.input:
        mode, path = item.split("=", 1)
        for row in csv.DictReader(Path(path).open(newline="")):
            row["mode"] = mode
            row["label"] = float(row["label"])
            row["prediction"] = float(row["prediction"])
            row["abs_error"] = float(row["abs_error"])
            rows.append(row)
    with (output_dir / "metrics_by_sample.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    by_cluster = defaultdict(list)
    for row in rows:
        by_cluster[(row["mode"], row["dialogue_id"])].append(row)
    with (output_dir / "metrics_by_dialogue.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = ("mode", "dialogue_id", "n", "acc", "f1", "mae", "corr")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (mode, cluster), values in sorted(by_cluster.items()):
            labels = [float(v["label"]) for v in values]
            predictions = [float(v["prediction"]) for v in values]
            writer.writerow(
                {
                    "mode": mode,
                    "dialogue_id": cluster,
                    "n": len(values),
                    **{
                        metric: metric_value(labels, predictions, metric)
                        for metric in ("acc", "f1", "mae", "corr")
                    },
                }
            )
    available = {row["mode"] for row in rows}
    results = []
    for comparison, baseline in COMPARISONS:
        if comparison not in available or baseline not in available:
            continue
        for metric in METRICS:
            stats = paired_cluster_bootstrap(
                rows,
                comparison,
                baseline,
                metric=metric,
                n_boot=args.n_boot,
                seed=args.seed,
            )
            results.append(stats)
    for metric in METRICS:
        family = [row for row in results if row["metric"] == metric]
        p_values = []
        for row in family:
            if row["ci_high"] < 0 or row["ci_low"] > 0:
                p_values.append(0.0001)
            else:
                p_values.append(1.0)
        for row, p_value in zip(family, holm_adjust(p_values)):
            row["p_holm"] = p_value
    with (output_dir / "full_metric_bootstrap.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    wild_rows = []
    for comparison, baseline in COMPARISONS:
        if comparison not in available or baseline not in available:
            continue
        clusters = per_cluster_metric_delta(
            rows, comparison, baseline, metric="mae"
        )
        wild = wild_cluster_bootstrap(clusters, n_boot=args.n_boot, seed=args.seed)
        wild_rows.append(
            {
                "comparison": comparison,
                "baseline": baseline,
                **wild,
            }
        )
    with (output_dir / "wild_cluster_bootstrap.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(wild_rows[0].keys()))
        writer.writeheader()
        writer.writerows(wild_rows)
    loo_rows = [
        {
            "comparison": row["comparison"],
            "baseline": row["baseline"],
            "metric": row["metric"],
            "loo_min": row["loo_min"],
            "loo_max": row["loo_max"],
            "delta": row["delta"],
        }
        for row in results
    ]
    with (output_dir / "leave_one_dialogue_out.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(loo_rows[0].keys()))
        writer.writeheader()
        writer.writerows(loo_rows)
    summary = {
        "n_boot": args.n_boot,
        "cluster_key": "dialogue_id (=sample_id)",
        "rows": len(rows),
        "modes": sorted(available),
        "results": results,
    }
    (output_dir / "full_metric_bootstrap.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    for row in results:
        print(
            f"{row['comparison']:>16} vs {row['baseline']:<13} {row['metric']:>4} "
            f"d={row['delta']:+.4f} CI[{row['ci_low']:+.4f},{row['ci_high']:+.4f}] "
            f"LOO[{row['loo_min']:+.4f},{row['loo_max']:+.4f}] "
            f"signs={int(row['negative_clusters'])}-/{int(row['positive_clusters'])} "
            f"p_holm={row['p_holm']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
