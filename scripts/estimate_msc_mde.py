from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.knowledge.msc.power import msc_power_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    payload = torch.load(args.subset_cache, map_location="cpu", weights_only=False)
    rows = payload["query_rows"]
    by_query = defaultdict(list)
    for row in rows:
        by_query[row["query_id"]].append(row)
    per_dialogue_gain = defaultdict(list)
    for qid, subset_rows in by_query.items():
        sample_id = subset_rows[0]["sample_id"]
        by_value = {int(row["subset_mask"]): float(row["utility"]) for row in subset_rows}
        singletons = {
            value: utility
            for value, utility in by_value.items()
            if value and (value & (value - 1)) == 0
        }
        if not singletons:
            continue
        exact = max(by_value.values())
        top1 = max(singletons.values())
        per_dialogue_gain[sample_id].append(exact - top1)
    deltas = {
        "exact_set_minus_top1": [
            sum(values) / len(values) for values in per_dialogue_gain.values()
        ]
    }
    report = msc_power_report(deltas, {"exact_set_minus_top1": 0.02})
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "msc_mde.json").write_text(
        json.dumps(
            {
                "dialogue_clusters": len(per_dialogue_gain),
                "report": report,
                "note": "Normal approximation on dialogue-level paired deltas",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with (output / "msc_mde.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "metric", "clusters", "mean_delta", "sd", "mde",
                "target_effect", "detects_target", "required_clusters",
            ),
        )
        writer.writeheader()
        for metric, stats in report.items():
            writer.writerow({"metric": metric, **stats})
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
