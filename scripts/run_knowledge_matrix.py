from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--card-adapter-checkpoint")
    parser.add_argument("--mask-manifest", required=True)
    parser.add_argument(
        "--modes",
        default="empty,raw_feature,card,random_card,shuffled_card,source_only",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--knowledge-index")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    modes = [value for value in args.modes.split(",") if value]
    commands = []
    for mode in modes:
        command = [
            sys.executable,
            "-u",
            "ComP/eval_knowledge.py",
            f"--student-checkpoint={args.student_checkpoint}",
            f"--mask-manifest={args.mask_manifest}",
            f"--mode={mode}",
            f"--split={args.split}",
            f"--gamma={args.gamma}",
            f"--seed={args.seed}",
            f"--gpu={args.gpu}",
            f"--output-dir={output_root / mode}",
        ]
        if args.card_adapter_checkpoint:
            command.append(
                f"--card-adapter-checkpoint={args.card_adapter_checkpoint}"
            )
        if args.allow_test:
            command.append("--allow-test")
        if args.max_batches:
            command.append(f"--max-batches={args.max_batches}")
        if args.knowledge_index:
            command.append(f"--knowledge-index={args.knowledge_index}")
        commands.append((mode, command))
    with (output_root / "commands.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        for mode, command in commands:
            writer.writerow([mode, " ".join(command)])
    log_path = output_root / "matrix.log"
    for mode, command in commands:
        print(("RUN" if args.execute else "DRY_RUN"), mode, flush=True)
        print(" ", " ".join(command), flush=True)
        if not args.execute:
            continue
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"RUN_START mode={mode}\n")
            log.flush()
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            if result.returncode != 0:
                log.write(f"RUN_FAILED mode={mode} rc={result.returncode}\n")
                raise SystemExit(result.returncode)
            log.write(f"RUN_DONE mode={mode}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
