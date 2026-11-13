from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


def metric(row, split, name):
    value = row.get(f"{split}_{name}", "")
    return float(value) if value not in ("", None) else float("nan")


def run_key(row):
    return (row["per_mode"], row["coordinator_mode"], float(row["mask_rate"]))


def fmt(value):
    return f"{value:+.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--b0", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(Path(args.rows).open(encoding="utf-8")))
    b0 = pd.read_csv(args.b0)
    by_key = {run_key(row): row for row in rows}

    report = {"runs": len(rows), "comparisons": {}, "teacher_training": {}}
    print("=== CORE RESULTS ===")
    for row in rows:
        print(
            Path(row["run_dir"]).name,
            "VAL",
            {key: round(metric(row, "val", key), 6)
             for key in ("acc", "f1", "mae", "corr")},
            "TEST",
            {key: round(metric(row, "test", key), 6)
             for key in ("acc", "f1", "mae", "corr")},
        )

    print("=== B0 COMPARISONS ===")
    for rate in (0.5, 0.7):
        b0_row = b0[b0["mask_rate"].round(2) == rate].iloc[0]
        b0_metrics = {
            split: {key: float(b0_row[f"{split}_{key}"])
                    for key in ("acc", "f1", "mae", "corr")}
            for split in ("val", "test")
        }
        print(f"B0 rate={rate}", b0_metrics)
        for mode in ("fixed", "task_router", "teacher_router"):
            row = by_key.get((mode, "on", rate))
            if row is None:
                continue
            delta = {}
            for split in ("val", "test"):
                delta[split] = {
                    key: metric(row, split, key) - b0_metrics[split][key]
                    for key in ("acc", "f1", "mae", "corr")
                }
            print(f"{mode} vs B0 rate={rate}", delta)
            report["comparisons"][f"{mode}_vs_B0_mr{rate}"] = delta

    print("=== GATE COMPARISONS ===")
    for rate in (0.5, 0.7):
        task = by_key[("task_router", "on", rate)]
        teacher = by_key[("teacher_router", "on", rate)]
        fixed = by_key[("fixed", "on", rate)]
        off = by_key[("teacher_router", "off", rate)]
        for label, left, right in (
            ("teacher_minus_task", teacher, task),
            ("task_minus_fixed", task, fixed),
            ("coord_off_minus_on", off, teacher),
        ):
            delta = {
                split: {
                    key: metric(left, split, key) - metric(right, split, key)
                    for key in ("acc", "f1", "mae", "corr")
                }
                for split in ("val", "test")
            }
            print(label, "rate=", rate, delta)
            report["comparisons"][f"{label}_mr{rate}"] = delta

    print("=== TEACHER TRAINING SIGNAL ===")
    for row in rows:
        if row["per_mode"] != "teacher_router":
            continue
        result_path = Path(row["checkpoint"]).parent / "result.json"
        if not result_path.exists():
            print("MISSING_RESULT", result_path)
            continue
        data = json.loads(result_path.read_text(encoding="utf-8"))
        stage2 = [
            item for item in data.get("history", [])
            if item.get("stage") == "stage2"
        ]
        last = stage2[-1].get("train", {}) if stage2 else {}
        keys = (
            "route_mean", "route_std", "route_low_fraction", "route_high_fraction",
            "route_target_mean", "route_target_std",
            "route_target_correlation", "teacher_utility_gap_mean",
            "teacher_utility_gap_std",
        )
        print(Path(row["run_dir"]).name, {key: last.get(key) for key in keys})
        report["teacher_training"][Path(row["run_dir"]).name] = {
            "stage2_epochs": len(stage2),
            "last_train_diagnostics": last,
        }

    gate1 = True
    gate3 = True
    for rate in (0.5, 0.7):
        task = by_key[("task_router", "on", rate)]
        teacher = by_key[("teacher_router", "on", rate)]
        off = by_key[("teacher_router", "off", rate)]
        teacher_val = (
            metric(teacher, "val", "acc") > metric(task, "val", "acc")
            and metric(teacher, "val", "mae") < metric(task, "val", "mae")
        )
        gate1 = gate1 and teacher_val
        off_val = (
            metric(off, "val", "acc") >= metric(teacher, "val", "acc")
            or metric(off, "val", "mae") <= metric(teacher, "val", "mae")
        )
        gate3 = gate3 and off_val

    report["gates"] = {
        "gate0_teacher_available": "PASS_AUDIT",
        "gate1_teacher_beats_task": "PASS" if gate1 else "FAIL",
        "gate2_router_beats_fixed": "MIXED_VALIDATION",
        "gate3_gain_without_coordinator": "MIXED" if gate3 else "FAIL",
        "gate4_real_vs_random_shuffled": "NOT_EVALUATED",
        "gate5_multiseed": "NOT_STARTED",
    }
    print("=== GATES ===")
    for key, value in report["gates"].items():
        print(key, value)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("REPORT", output)


if __name__ == "__main__":
    main()
