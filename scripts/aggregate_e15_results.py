from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def add_metric_rows(rows, variant, rate, mode, metrics):
    if metrics:
        rows.append({
            "variant": variant, "rate": rate, "mode": mode,
            "group": "overall", **metrics,
        })


def metric_delta(left, right):
    return {key: float(left[key]) - float(right[key]) for key in ("acc", "f1", "mae", "corr")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    cli = parser.parse_args()
    root = cli.run_root
    rows = []
    comparisons = {}
    completed = 0
    for result_path in sorted(root.glob("*/mr*/*/result.json")):
        variant, rate_dir, mode = result_path.parts[-4], result_path.parts[-3], result_path.parts[-2]
        rate = rate_dir[2:]
        data = json.loads(result_path.read_text(encoding="utf-8"))
        add_metric_rows(rows, variant, rate, mode, data.get("metrics"))
        grouped = data.get("metrics_by_missing_pattern_and_target", {})
        for group, metrics in grouped.items():
            rows.append({
                "variant": variant, "rate": rate, "mode": mode,
                "group": group, **metrics,
            })
        completed += 1
    for summary_path in sorted(root.glob("*/mr*/oracle_v2/oracle_v2_summary.json")):
        variant, rate_dir = summary_path.parts[-4], summary_path.parts[-3]
        rate = rate_dir[2:]
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        add_metric_rows(rows, variant, rate, "oracle_teacher_v2", data.get("oracle_teacher_metrics"))
        add_metric_rows(rows, variant, rate, "oracle_task_v2", data.get("oracle_task_metrics"))
        for group_planner, grouped in data.get("metrics_by_missing_pattern_and_target", {}).items():
            for group, metrics in grouped.items():
                rows.append({
                    "variant": variant, "rate": rate, "mode": group_planner,
                    "group": group, **metrics,
                })
        completed += 1
    fields = sorted({key for row in rows for key in row})
    with (root / "e15_evidence_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    for variant in sorted({row["variant"] for row in rows}):
        for rate in ("0.5", "0.7"):
            overall = {
                row["mode"]: row for row in rows
                if row["variant"] == variant and row["rate"] == rate and row["group"] == "overall"
            }
            if not {"empty", "feature", "random", "source_only"} <= set(overall):
                continue
            comparisons[f"{variant}|{rate}"] = {
                "real_minus_random": metric_delta(overall["feature"], overall["random"]),
                "real_minus_source_only": metric_delta(overall["feature"], overall["source_only"]),
                "task_oracle_minus_empty": (
                    metric_delta(overall["oracle_task_v2"], overall["empty"])
                    if "oracle_task_v2" in overall else None
                ),
                "teacher_oracle_minus_empty": (
                    metric_delta(overall["oracle_teacher_v2"], overall["empty"])
                    if "oracle_teacher_v2" in overall else None
                ),
            }
    summary = {
        "completed_json_files": completed,
        "row_count": len(rows),
        "comparisons": comparisons,
        "split": "validation",
        "test_access": "forbidden",
    }
    (root / "e15_aggregate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
