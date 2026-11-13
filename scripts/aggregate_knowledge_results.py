from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.knowledge.gate import evaluate_ek3, evaluate_ek4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for path in sorted(root.rglob("result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "split": payload.get("split"),
                "mode": payload.get("mode"),
                "gamma": payload.get("gamma"),
                "acc": metrics.get("acc"),
                "f1": metrics.get("f1"),
                "mae": metrics.get("mae"),
                "corr": metrics.get("corr"),
                "count": metrics.get("count"),
                "loss": metrics.get("loss"),
                "card_adapter_sha256": payload.get("card_adapter_sha256"),
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "path",
        "split",
        "mode",
        "gamma",
        "acc",
        "f1",
        "mae",
        "corr",
        "count",
        "loss",
        "card_adapter_sha256",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metric_by_mode = {
        (
            "learned_filter"
            if row["mode"] in {"listwise_filter", "pairwise_filter"}
            else row["mode"]
        ): {
            "acc": row["acc"],
            "f1": row["f1"],
            "mae": row["mae"],
            "corr": row["corr"],
        }
        for row in rows
        if row["mode"] is not None
    }
    if {"empty", "card", "random_card", "shuffled_card", "source_only"}.issubset(
        metric_by_mode
    ):
        (output.parent / "ek3_gate.json").write_text(
            json.dumps(evaluate_ek3(metric_by_mode), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    if {"learned_filter", "confidence", "random_action", "card"}.issubset(
        metric_by_mode
    ):
        (output.parent / "ek4_gate.json").write_text(
            json.dumps(
                evaluate_ek4(metric_by_mode, lower_is_better=False),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"rows": len(rows), "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
