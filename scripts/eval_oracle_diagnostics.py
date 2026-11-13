from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_eval(args, *, mode, objective, target):
    command = [
        sys.executable,
        "-u",
        "ComP/eval_knowledge.py",
        f"--student-checkpoint={args.student_checkpoint}",
        f"--card-adapter-checkpoint={args.card_adapter_checkpoint}",
        f"--knowledge-index={args.knowledge_index}",
        f"--mask-manifest={args.mask_manifest}",
        f"--mode={mode}",
        f"--oracle-objective={objective}",
        "--split=validation",
        f"--gamma={args.gamma}",
        f"--gpu={args.gpu}",
        f"--output-dir={target}",
    ]
    if args.max_batches:
        command.append(f"--max-batches={args.max_batches}")
    print("RUN", mode, objective, flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise SystemExit(f"eval failed: {mode}/{objective}")
    return json.loads((target / "result.json").read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--card-adapter-checkpoint", required=True)
    parser.add_argument("--knowledge-index", required=True)
    parser.add_argument("--mask-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    empty = run_eval(
        args, mode="empty", objective="task", target=output_dir / "empty"
    )
    bounds = {}
    rows = []
    for objective in ("task", "mae", "acc", "composite"):
        result = run_eval(
            args,
            mode="oracle_card",
            objective=objective,
            target=output_dir / f"oracle_{objective}",
        )
        metrics = result["metrics"]
        rows.append(
            {
                "objective": objective,
                "acc": metrics["acc"],
                "f1": metrics["f1"],
                "mae": metrics["mae"],
                "corr": metrics.get("corr", 0.0),
                "count": metrics["count"],
            }
        )
        bounds[f"oracle_{objective}"] = {
            "delta_mae_vs_empty": metrics["mae"] - empty["metrics"]["mae"],
            "delta_acc_vs_empty": metrics["acc"] - empty["metrics"]["acc"],
            "has_mae_headroom": metrics["mae"] < empty["metrics"]["mae"],
            "has_acc_headroom": metrics["acc"] > empty["metrics"]["acc"],
            "label_conditional_oracle": True,
            "generalization_estimate": False,
            "diagnostic_only": True,
        }
    base_row = {
        "objective": "empty",
        "acc": empty["metrics"]["acc"],
        "f1": empty["metrics"]["f1"],
        "mae": empty["metrics"]["mae"],
        "corr": empty["metrics"].get("corr", 0.0),
        "count": empty["metrics"]["count"],
    }
    with (output_dir / "oracle_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(base_row.keys()))
        writer.writeheader()
        writer.writerow(base_row)
        writer.writerows(rows)
    payload = {
        "gamma": args.gamma,
        "empty": base_row,
        "oracles": rows,
        "upper_bounds": bounds,
        "test_accessed": False,
    }
    (output_dir / "oracle_upper_bounds.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
