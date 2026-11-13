from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


THRESHOLDS = {
    "strong_acc_margin": 0.02,
    "strong_f1_margin": 0.02,
    "strong_mae_allowance": 0.02,
    "real_acc_or_f1_margin": 0.01,
    "real_mae_allowance": 0.02,
    "minimum_feature_coverage": 0.5,
}


def value(row, key):
    return float(row[key])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root
    rows = list(csv.DictReader((root / "e1_rows.csv").open(encoding="utf-8")))
    lookup = {(row["mask_rate"], row["evidence_mode"]): row for row in rows}

    comparisons = {}
    strong_pass = False
    real_pass = False
    coverage_pass = True
    for rate in ("0.5", "0.7"):
        strong = lookup[(rate, "oracle_strong")]
        random = lookup[(rate, "random")]
        shuffled = lookup[(rate, "shuffled")]
        feature = lookup[(rate, "feature")]
        best_control = max((random, shuffled), key=lambda row: value(row, "acc"))
        comparisons[rate] = {}
        for label, left, right in (
            ("strong_minus_feature", strong, feature),
            ("strong_minus_random", strong, random),
            ("strong_minus_shuffled", strong, shuffled),
            ("feature_minus_random", feature, random),
        ):
            comparisons[rate][label] = {
                key: value(left, key) - value(right, key)
                for key in ("acc", "f1", "mae", "corr")
            }
        strong_pass = strong_pass or (
            value(strong, "acc") - value(best_control, "acc") >= THRESHOLDS["strong_acc_margin"]
            and value(strong, "f1") - value(best_control, "f1") >= THRESHOLDS["strong_f1_margin"]
            and value(strong, "mae") - value(best_control, "mae") <= THRESHOLDS["strong_mae_allowance"]
        )
        real_pass = real_pass or (
            (
                value(feature, "acc") - value(random, "acc") >= THRESHOLDS["real_acc_or_f1_margin"]
                or value(feature, "f1") - value(random, "f1") >= THRESHOLDS["real_acc_or_f1_margin"]
            )
            and value(feature, "mae") - value(random, "mae") <= THRESHOLDS["real_mae_allowance"]
        )
        coverage = json.loads(feature["coverage"])
        coverage_pass = coverage_pass and (
            coverage["coverage_all_fraction"] >= THRESHOLDS["minimum_feature_coverage"]
        )

    no_leak_pass = all(
        row["split"] == "validation"
        and row["uses_validation_labels"] == "True"
        and row["deployable"] == "False"
        for row in rows
        if row["evidence_mode"] == "oracle_strong"
    )
    decision = {
        "gate": "E1",
        "decision": "PASS" if strong_pass and real_pass and coverage_pass and no_leak_pass else "FAIL",
        "strong_oracle_dominates_controls": strong_pass,
        "real_feature_beats_random_on_at_least_one_rate": real_pass,
        "coverage_pass": coverage_pass,
        "split_guard_pass": no_leak_pass,
        "thresholds": THRESHOLDS,
        "comparisons": comparisons,
        "test_access": "forbidden",
        "next_action": "STOP_E2_AND_REPLACE_EVIDENCE_SOURCE",
    }
    destination = root / "e1_gate_decision.json"
    destination.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    summary_path = root / "e1_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["gate_decision"] = decision["decision"]
    summary["gate_decision_file"] = str(destination)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
