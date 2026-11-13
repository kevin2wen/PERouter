from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENV_PY = Path("/home/xuanqin/.venvs/pytorch/bin/python")
TEACHER = REPO / "runs/per/teachers/CMUMOSI/seed66_fold0/teacher.pt"
ADAPTER_ROOT = REPO / "runs/evidence_gates/E15_adapters_20261112-193154"
STUDENTS = {
    "0.5": REPO / "runs/per/core_v1_20261108-114108/teacher_router_mr0.5_coord_on_seed66/CMUMOSI/20261108-120811_seed66_mr0.5_teacher_router/fold0/best_val.pt",
    "0.7": REPO / "runs/per/core_v1_20261108-114108/teacher_router_mr0.7_coord_on_seed66/CMUMOSI/20261108-121547_seed66_mr0.7_teacher_router/fold0/best_val.pt",
}
MASKS = {
    "0.5": REPO / "runs/per/core_v1_20261108-114108/teacher_router_mr0.5_coord_on_seed66/CMUMOSI/20261108-120811_seed66_mr0.5_teacher_router/fold0/val_mask_manifest.csv",
    "0.7": REPO / "runs/per/core_v1_20261108-114108/teacher_router_mr0.7_coord_on_seed66/CMUMOSI/20261108-121547_seed66_mr0.7_teacher_router/fold0/val_mask_manifest.csv",
}
MODES = ("empty", "feature", "random", "shuffled", "source_only", "oracle_weak", "oracle_strong")
VARIANTS = {
    "original_feature": ("feature", None),
    "original_text": ("text", None),
    "adapter_feature": ("feature", ADAPTER_ROOT / "feature/evidence_adapter_feature.pt"),
    "adapter_text": ("text", ADAPTER_ROOT / "text/evidence_adapter_text.pt"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    cli = parser.parse_args()
    root = cli.run_root
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "status.tsv"
    with status_path.open("w", encoding="utf-8") as handle:
        handle.write("variant\trate\tmode\tcommand\texit_code\n")
    failures = 0
    for variant, (source, adapter) in VARIANTS.items():
        for rate in ("0.5", "0.7"):
            for mode in MODES + ("oracle_v2",):
                output = root / variant / f"mr{rate}" / mode
                output.mkdir(parents=True, exist_ok=True)
                expected = (
                    output / "oracle_v2_summary.json"
                    if mode == "oracle_v2"
                    else output / "result.json"
                )
                if cli.skip_existing and expected.is_file():
                    code = 0
                    print(f"SKIP_EXISTING {expected}", flush=True)
                else:
                    if mode == "oracle_v2":
                        command = [
                            str(VENV_PY), str(REPO / "scripts/eval_oracle_v2.py"),
                            "--student-checkpoint", str(STUDENTS[rate]),
                            "--teacher-checkpoint", str(TEACHER),
                            "--split", "validation",
                            "--mask-manifest", str(MASKS[rate]),
                            "--evidence-source", source,
                            "--output-dir", str(output), "--gpu", "0",
                        ]
                        if adapter is not None:
                            command.extend(["--adapter-checkpoint", str(adapter)])
                    else:
                        command = [
                            str(VENV_PY), str(REPO / "scripts/eval_evidence_oracle.py"),
                            "--student-checkpoint", str(STUDENTS[rate]),
                            "--teacher-checkpoint", str(TEACHER),
                            "--split", "validation",
                            "--mask-manifest", str(MASKS[rate]),
                            "--evidence-source", source,
                            "--evidence-mode", mode,
                            "--fixed-source-weight", "0.5",
                            "--output-dir", str(output), "--gpu", "0",
                        ]
                        if adapter is not None:
                            command.extend(["--adapter-checkpoint", str(adapter)])
                    with (output / "command.txt").open("w", encoding="utf-8") as handle:
                        handle.write(" ".join(command) + "\n")
                    with (output / "run.log").open("w", encoding="utf-8") as handle:
                        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
                    code = result.returncode
                with status_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"{variant}\t{rate}\t{mode}\t{output}\t{code}\n")
                print(f"{'DONE' if code == 0 else 'FAILED'} variant={variant} rate={rate} mode={mode} exit={code}", flush=True)
                if code != 0:
                    failures += 1
                    log = output / "run.log"
                    if log.is_file():
                        print(log.read_text(encoding="utf-8")[-12000:], flush=True)
    metadata = {
        "expected_runs": len(VARIANTS) * 2 * (len(MODES) + 1),
        "failures": failures,
        "split": "validation",
        "test_access": "forbidden",
        "variants": {key: value[0] for key, value in VARIANTS.items()},
        "run_root": str(root),
    }
    (root / "matrix_summary.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
