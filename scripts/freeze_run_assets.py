from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout.strip()


def artifact_hashes(run_dir: Path) -> list[dict[str, object]]:
    patterns = (
        "summary.json",
        "result.json",
        "*mask*manifest*.csv",
        "best_val.pt",
        "teacher.pt",
        "*.log",
    )
    records = []
    seen = set()
    for pattern in patterns:
        for path in sorted(run_dir.rglob(pattern)):
            resolved = path.resolve()
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            records.append(
                {
                    "path": str(resolved),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze code and experiment provenance.")
    parser.add_argument("--repo", default="/home/xuanqin/PERouter")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--command-text", default="")
    parser.add_argument("--dataset-version", default="")
    parser.add_argument("--teacher-checkpoint", action="append", default=[])
    parser.add_argument("--strict-clean", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    tracked_status = git_output(repo, "status", "--porcelain", "--untracked-files=no")
    if args.strict_clean and tracked_status:
        raise RuntimeError("tracked working tree is dirty")

    commit = git_output(repo, "rev-parse", "HEAD")
    branch = git_output(repo, "branch", "--show-current")
    describe = git_output(repo, "describe", "--tags", "--always", "--dirty")
    status = git_output(repo, "status", "--short")
    diff = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=repo,
        check=False,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout

    (output / "git_diff.patch").write_text(diff, encoding="utf-8")
    (output / "git_status.txt").write_text(status + "\n", encoding="utf-8")
    (output / "formal_v1_tag.txt").write_text(describe + "\n", encoding="utf-8")

    environment = [
        f"timestamp={datetime.datetime.now().astimezone().isoformat()}",
        f"hostname={platform.node()}",
        f"platform={platform.platform()}",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
    ]
    try:
        import numpy
        import pandas
        import sklearn
        import torch

        environment.extend(
            [
                f"torch={torch.__version__}",
                f"torch_cuda={torch.version.cuda}",
                f"torch_cudnn={torch.backends.cudnn.version()}",
                f"cuda_available={torch.cuda.is_available()}",
                f"torch_num_threads={torch.get_num_threads()}",
                f"numpy={numpy.__version__}",
                f"pandas={pandas.__version__}",
                f"sklearn={sklearn.__version__}",
            ]
        )
        if torch.cuda.is_available():
            environment.append(f"gpu_name={torch.cuda.get_device_name(0)}")
    except Exception as error:
        environment.append(f"python_environment_error={error!r}")
    if shutil.which("nvidia-smi"):
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout.strip()
        environment.append(f"nvidia_smi={gpu}")
    (output / "environment.txt").write_text("\n".join(environment) + "\n", encoding="utf-8")

    teacher_records = []
    for raw in args.teacher_checkpoint:
        path = Path(raw).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        teacher_records.append(
            {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}
        )

    manifest = {
        "schema_version": 1,
        "git": {
            "commit": commit,
            "branch": branch,
            "describe": describe,
            "tracked_worktree_clean": not bool(tracked_status),
        },
        "launch": {
            "command": args.command_text,
            "run_dir": str(run_dir),
            "dataset_version": args.dataset_version,
        },
        "environment_file": str(output / "environment.txt"),
        "git_diff_file": str(output / "git_diff.patch"),
        "teacher_checkpoints": teacher_records,
        "artifacts": artifact_hashes(run_dir),
    }
    manifest_path = output / "code_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
