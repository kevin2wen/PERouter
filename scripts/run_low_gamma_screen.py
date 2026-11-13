from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--card-adapter-checkpoint", required=True)
    parser.add_argument("--knowledge-index", required=True)
    parser.add_argument("--mask-manifest", required=True)
    parser.add_argument("--reranker-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gammas", default="0,0.1,0.25,0.5")
    parser.add_argument(
        "--modes",
        default="empty,card,random_card,shuffled_card,source_only,oracle_card",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gammas = [float(value) for value in args.gammas.split(",") if value]
    modes = [value for value in args.modes.split(",") if value]
    rows = []
    for gamma in gammas:
        for mode in modes:
            target = output_dir / f"gamma_{gamma:g}" / mode
            command = [
                sys.executable,
                "-u",
                "ComP/eval_knowledge.py",
                f"--student-checkpoint={args.student_checkpoint}",
                f"--card-adapter-checkpoint={args.card_adapter_checkpoint}",
                f"--knowledge-index={args.knowledge_index}",
                f"--mask-manifest={args.mask_manifest}",
                f"--mode={mode}",
                "--split=validation",
                f"--gamma={gamma}",
                f"--gpu={args.gpu}",
                f"--output-dir={target}",
            ]
            if args.max_batches:
                command.append(f"--max-batches={args.max_batches}")
            if mode in {"listwise_filter", "pairwise_filter"}:
                if not args.reranker_dir:
                    raise SystemExit("filter modes require --reranker-dir")
                command.append(f"--reranker-dir={args.reranker_dir}")
            print("RUN", gamma, mode, flush=True)
            completed = subprocess.run(command, cwd=REPO_ROOT)
            if completed.returncode != 0:
                raise SystemExit(f"eval failed: gamma={gamma} mode={mode}")
            result = json.loads((target / "result.json").read_text())
            metrics = result["metrics"]
            rows.append(
                {
                    "gamma": gamma,
                    "mode": mode,
                    "acc": metrics["acc"],
                    "f1": metrics["f1"],
                    "mae": metrics["mae"],
                    "corr": metrics.get("corr", 0.0),
                    "count": metrics["count"],
                }
            )
    with (output_dir / "gamma_screen_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("gamma", "mode", "acc", "f1", "mae", "corr", "count"),
        )
        writer.writeheader()
        writer.writerows(rows)
    payload = {"gammas": gammas, "modes": modes, "rows": rows}
    (output_dir / "gamma_screen.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
