from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--card-adapter-checkpoint", required=True)
    parser.add_argument("--knowledge-index")
    parser.add_argument("--mask-manifest", required=True)
    parser.add_argument("--reranker-dir", required=True)
    parser.add_argument("--objective", choices=("listwise", "pairwise"), required=True)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    command = [
        sys.executable,
        "-u",
        "ComP/eval_knowledge.py",
        f"--student-checkpoint={args.student_checkpoint}",
        f"--card-adapter-checkpoint={args.card_adapter_checkpoint}",
        f"--mask-manifest={args.mask_manifest}",
        f"--mode={args.objective}_filter",
        f"--reranker-dir={args.reranker_dir}",
        "--split=validation",
        f"--gamma={args.gamma}",
        f"--max-batches={args.max_batches}",
        f"--gpu={args.gpu}",
        f"--output-dir={args.output_dir}",
    ]
    if args.knowledge_index:
        command.append(f"--knowledge-index={args.knowledge_index}")
    return subprocess.call(command, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
