from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nested-root", required=True)
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--acc-margin", type=float, default=0.01)
    args = parser.parse_args()
    root = Path(args.nested_root)
    rows = []
    for path in sorted(root.glob("adapter_gamma*/fold*/fold_result.json")):
        payload = json.loads(path.read_text())
        rows.append(payload["row"])
    if not rows:
        raise SystemExit("no fold results")
    with (root / "cv_fold_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    table = []
    for gamma in sorted({row["gamma"] for row in rows}):
        subset = [row for row in rows if row["gamma"] == gamma]
        folds = len(subset)
        non_inf = sum(
            1
            for row in subset
            if row["card_mae"] <= row["empty_mae"] + args.tolerance
            and row["card_acc"] >= row["empty_acc"] - args.acc_margin
        )
        oracle_positive = sum(
            1 for row in subset if row["oracle_minus_empty_mae"] < 0
        )
        card_positive = sum(
            1 for row in subset if row["card_minus_empty_mae"] < 0
        )
        table.append(
            {
                "gamma": gamma,
                "folds": folds,
                "non_inferiority_folds": non_inf,
                "oracle_positive_folds": oracle_positive,
                "card_positive_folds": card_positive,
                "mean_card_minus_empty_mae": sum(
                    row["card_minus_empty_mae"] for row in subset
                ) / folds,
                "mean_card_minus_empty_acc": sum(
                    row["card_minus_empty_acc"] for row in subset
                ) / folds,
                "mean_oracle_minus_empty_mae": sum(
                    row["oracle_minus_empty_mae"] for row in subset
                ) / folds,
                "mean_card_mae": sum(row["card_mae"] for row in subset) / folds,
                "mean_empty_mae": sum(row["empty_mae"] for row in subset) / folds,
            }
        )
    with (root / "gamma_selection_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    eligible = [
        row for row in table
        if row["non_inferiority_folds"] >= max(1, row["folds"] - 1)
        and row["oracle_positive_folds"] >= max(1, row["folds"] - 1)
    ]
    pool = eligible or table
    selected = sorted(
        pool,
        key=lambda row: (
            -row["oracle_positive_folds"],
            row["mean_card_minus_empty_mae"],
            row["gamma"],
        ),
    )[0]
    payload = {
        "selected_gamma": selected["gamma"],
        "selection_table": table,
        "eligible": bool(eligible),
        "tolerance": args.tolerance,
        "acc_margin": args.acc_margin,
        "rule": "non-inferiority, then oracle stability, then mean card MAE, then smaller gamma",
    }
    (root / "selected_gamma.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
