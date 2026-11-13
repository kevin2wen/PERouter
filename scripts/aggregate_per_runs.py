from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

METRIC_KEYS = (
    "acc", "f1", "mae", "corr", "loss",
    "route_mean", "route_std", "route_low_fraction", "route_high_fraction",
    "route_target_mean", "route_target_std", "route_target_correlation",
    "teacher_utility_gap_mean", "teacher_utility_gap_std",
    "route_audio_mean", "route_text_mean", "route_video_mean",
)


def latest_summary(run_dir: Path):
    paths = sorted(
        run_dir.glob("CMUMOSI/*/summary.json"),
        key=lambda path: path.stat().st_mtime,
    )
    return paths[-1] if paths else None


def row_from_summary(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    fold = data["folds"][0]
    validation = fold.get("validation", {})
    test = fold.get("test", {})
    row: Dict[str, object] = {
        "run_dir": str(path.parents[2]),
        "summary": str(path),
        "dataset": data.get("dataset"),
        "per_mode": data.get("per_mode"),
        "coordinator_mode": data.get("coordinator_mode"),
        "mask_rate": data.get("mask_rate"),
        "seed": data.get("seed"),
        "missing_protocol": data.get("missing_protocol"),
        "fixed_source_weight": data.get("fixed_source_weight"),
        "teacher_checkpoint": data.get("teacher_checkpoint_template"),
        "best_epoch": fold.get("best_epoch"),
        "checkpoint": fold.get("checkpoint"),
        "val_mask_manifest": fold.get("val_mask_manifest"),
        "test_mask_manifest": fold.get("test_mask_manifest"),
    }
    for key in METRIC_KEYS:
        row[f"val_{key}"] = validation.get(key)
        row[f"test_{key}"] = test.get(key)
    return row


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        summary = latest_summary(run_dir)
        if summary is not None:
            rows.append(row_from_summary(summary))
    rows.sort(key=lambda row: str(row["run_dir"]))
    write_rows(output / "rows.csv", rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(
            str(row.get("dataset")),
            str(row.get("per_mode")),
            str(row.get("coordinator_mode")),
            str(row.get("mask_rate")),
        )].append(row)

    stats_rows = []
    for key, group in sorted(grouped.items()):
        result = {
            "dataset": key[0],
            "per_mode": key[1],
            "coordinator_mode": key[2],
            "mask_rate": key[3],
            "seeds": ",".join(str(row.get("seed")) for row in group),
            "runs": len(group),
        }
        for split in ("val", "test"):
            for metric in (
                "acc", "f1", "mae", "corr", "loss",
                "route_mean", "route_std", "route_target_correlation",
                "teacher_utility_gap_mean",
            ):
                values = [
                    float(row[f"{split}_{metric}"])
                    for row in group
                    if row.get(f"{split}_{metric}") is not None
                    and math.isfinite(float(row[f"{split}_{metric}"]))
                ]
                result[f"{split}_{metric}_mean"] = (
                    statistics.mean(values) if values else ""
                )
                result[f"{split}_{metric}_std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
        stats_rows.append(result)
    write_rows(output / "stats.csv", stats_rows)

    print(f"COMPLETE_RUNS={len(rows)}")
    print(f"ROWS={output / 'rows.csv'}")
    print(f"STATS={output / 'stats.csv'}")
    for row in rows:
        print(
            "RESULT",
            Path(str(row["run_dir"])).name,
            "mode=", row["per_mode"],
            "coordinator=", row["coordinator_mode"],
            "rate=", row["mask_rate"],
            "seed=", row["seed"],
            "val=", {key: row[f"val_{key}"] for key in ("acc", "f1", "mae", "corr")},
            "test=", {key: row[f"test_{key}"] for key in ("acc", "f1", "mae", "corr")},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
