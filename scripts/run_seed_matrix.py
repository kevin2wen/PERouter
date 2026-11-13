from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_command(method: str, rate: float, seed: int, output_root: Path, teacher: str | None):
    command = [
        sys.executable, "-u", "ComP/train_per.py",
        "--protocol=fair",
        "--dataset=CMUMOSI",
        "--audio-feature=wav2vec-large-c-UTT",
        "--text-feature=deberta-large-4-UTT",
        "--video-feature=manet_UTT",
        f"--seed={seed}",
        "--batch-size=32",
        "--epochs=300",
        "--stage_epoch=150",
        "--lr=0.0001",
        "--hidden=256",
        "--depth=4",
        "--num_heads=2",
        "--drop_rate=0.5",
        "--attn_drop_rate=0.0",
        "--gpu=0",
        f"--mask_rate={rate}",
        "--lbd=0.3",
        "--missing-protocol=categorical",
        "--top-k=5",
        "--route-weight=0.5",
        "--route-temperature=1.0",
    ]
    label = f"{method}_mr{rate}_seed{seed}"
    command.append(f"--output-dir={output_root / label}")
    if method == "B0":
        command.extend(["--per-mode=off", "--evidence-mode=empty", "--coordinator-mode=on"])
    elif method == "fixed":
        command.extend([
            "--per-mode=fixed", "--fixed-source-weight=0.5",
            "--evidence-mode=feature", "--coordinator-mode=on",
        ])
    elif method == "task":
        command.extend([
            "--per-mode=task_router", "--evidence-mode=feature",
            "--coordinator-mode=on",
        ])
    elif method == "teacher":
        teacher_value = teacher or "<TEACHER_CHECKPOINT>"
        command.extend([
            "--per-mode=teacher_router", "--evidence-mode=feature",
            "--coordinator-mode=on", f"--teacher-checkpoint={teacher_value}",
        ])
    else:
        raise ValueError(f"unknown method: {method}")
    return label, command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", default="11,22,33,44,66")
    parser.add_argument("--rates", default="0.5,0.7")
    parser.add_argument("--methods", default="B0,fixed,task,teacher")
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    methods = [value for value in args.methods.split(",") if value]
    if args.execute and "teacher" in methods and not args.teacher_checkpoint:
        raise ValueError("--teacher-checkpoint is required with --execute")
    seeds = [int(value) for value in args.seeds.split(",") if value]
    rates = [float(value) for value in args.rates.split(",") if value]
    commands = []
    for rate in rates:
        for seed in seeds:
            for method in methods:
                label, command = build_command(
                    method, rate, seed, output_root, args.teacher_checkpoint
                )
                commands.append({"label": label, "command": command})
    manifest = output_root / "commands.tsv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for item in commands:
            writer.writerow([item["label"], " ".join(item["command"])])
    log_path = output_root / "matrix.log"
    for item in commands:
        run_dir = output_root / item["label"]
        if (run_dir / "summary.json").exists() or list(run_dir.rglob("summary.json")):
            print("SKIP", item["label"])
            continue
        print("DRY_RUN" if not args.execute else "RUN", item["label"], flush=True)
        if not args.execute:
            print(" ", " ".join(item["command"]))
            continue
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"RUN_START label={item['label']}\n")
            log.flush()
            result = subprocess.run(item["command"], cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                log.write(f"RUN_FAILED label={item['label']} rc={result.returncode}\n")
                raise SystemExit(result.returncode)
            log.write(f"RUN_DONE label={item['label']}\n")
    if args.execute:
        log_path.open("a", encoding="utf-8").write("MATRIX_DONE\n")
    print("COMMANDS", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
