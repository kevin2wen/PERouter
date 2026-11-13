from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument(
        "--objective", choices=("listwise", "pairwise", "utility"), required=True
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    command = [
        sys.executable,
        "-u",
        "scripts/train_evidence_reranker.py",
        f"--candidate-cache={args.candidate_cache}",
        "--source=knowledge",
        f"--objective={args.objective}",
        f"--folds={args.folds}",
        f"--seed={args.seed}",
        f"--epochs={args.epochs}",
        f"--patience={args.patience}",
        f"--progress-every={args.progress_every}",
        f"--batch-size={args.batch_size}",
        f"--learning-rate={args.learning_rate}",
        f"--weight-decay={args.weight_decay}",
        f"--hidden-dim={args.hidden_dim}",
        f"--output-dir={args.output_dir}",
    ]
    return subprocess.call(command, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
