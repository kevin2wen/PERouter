from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir or run_root / "stat_recheck")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted((run_root / "validation").glob("*/per_sample.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise SystemExit("no per_sample.csv found; rerun evals with current code")
    fieldnames = list(rows[0].keys())
    with (output_dir / "paired_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    modes = sorted({row["mode"] for row in rows})
    usage = {"modes": modes, "rows": len(rows)}
    summary_path = run_root / "adapter" / "training_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        usage["best_epoch"] = summary.get("best_epoch")
        usage["epochs_run"] = len(summary.get("history", []))
        usage["validation_used_for_adapter_checkpoint"] = True
        usage["validation_tuned"] = True
        usage["selection_metric"] = "validation mae (score = -mae)"
    usage["protocol"] = {
        "train_fit": "train split",
        "train_adapter_validation": "not yet separated",
        "official_validation_final": "current validation; tuned",
    }
    (output_dir / "validation_usage.json").write_text(
        json.dumps(usage, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(usage, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
