from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-cache", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--card-adapter-checkpoint", required=True)
    parser.add_argument("--knowledge-index", required=True)
    parser.add_argument("--mask-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variants", default="s0,s1,s2")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.output_root)
    for variant in [value for value in args.variants.split(",") if value]:
        train_dir = root / f"single_{variant}"
        subprocess.run(
            [
                sys.executable, "-u", "scripts/train_msc_single_scorer.py",
                f"--single-cache={args.single_cache}",
                f"--output-dir={train_dir}",
                f"--variant={variant}",
                f"--epochs={args.epochs}",
                f"--gpu={args.gpu}",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        calibration = root / f"calibration_{variant}.json"
        subprocess.run(
            [
                sys.executable, "-u", "scripts/calibrate_msc_skip.py",
                f"--single-cache={args.single_cache}",
                f"--checkpoint={train_dir / 'best.pt'}",
                f"--output={calibration}",
                f"--gpu={args.gpu}",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable, "-u", "scripts/eval_msc_single.py",
                f"--student-checkpoint={args.student_checkpoint}",
                f"--card-adapter-checkpoint={args.card_adapter_checkpoint}",
                f"--knowledge-index={args.knowledge_index}",
                f"--mask-manifest={args.mask_manifest}",
                f"--scorer-checkpoint={train_dir / 'best.pt'}",
                f"--calibration={calibration}",
                "--split=validation",
                "--gamma=0.25",
                f"--gpu={args.gpu}",
                f"--output-dir={root / f'eval_{variant}'}",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
