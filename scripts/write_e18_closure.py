from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/home/xuanqin/PERouter"))
    parser.add_argument("--closure-dir", type=Path, required=True)
    parser.add_argument("--lineage-root", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--original-validation-root", type=Path, required=True)
    args = parser.parse_args()
    lineage = load(args.lineage_root / "evidence_lineage_summary.json")
    rates = {}
    for rate in ("0.5", "0.7"):
        clean_base = args.clean_root / "validation" / f"mr{rate}"
        original_base = args.original_validation_root / f"mr{rate}"
        rates[rate] = {
            "empty": load(original_base / "raw_empty" / "result.json")["metrics"],
            "original_listwise": load(original_base / "listwise_feature" / "result.json")["metrics"],
            "clean_listwise_real": load(clean_base / "feature" / "result.json")["metrics"],
            "clean_listwise_random": load(clean_base / "random" / "result.json")["metrics"],
            "clean_listwise_source_only": load(clean_base / "source_only" / "result.json")["metrics"],
            "clean_oof": load(args.clean_root / "reranker" / f"mr{rate}" / "listwise_clean" / "reranker_training_summary.json"),
        }
    decision = {
        "schema_version": 1,
        "date": "2026-11-12",
        "gate": "E1.8",
        "decision": "FAIL",
        "category": "CLEAN_BANK_PARTIALLY_EXPLAINS_LEAKAGE_BUT_RERANKER_STILL_FAILS",
        "test_access": "forbidden",
        "lineage": lineage,
        "rates": rates,
        "findings": [
            "Train candidates frequently came from the same video, while validation candidates never did.",
            "Removing same-video/same-dialogue candidates reduced OOF optimism but retained positive OOF policy utility.",
            "Clean-listwise validation improved over the original learned reranker.",
            "Clean-listwise real evidence still did not beat empty or source-only.",
        ],
        "unexecuted": [
            "strict_student_crossfit",
            "joint_oracle_task_metrics",
            "use_rate_matched_random_action",
            "action_calibration",
            "text_clean_bank",
            "K_10_20",
            "test_evaluation",
        ],
    }
    json_path = args.closure_dir / "E18_CLOSURE_20261112.json"
    json_path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# ComP-PER E1.8 Clean Bank / Generalization Closure",
        "",
        "日期：2026-11-12  ",
        "数据集：CMU-MOSI  ",
        "评价 split：validation  ",
        "test：未访问  ",
        "最终判定：**E1.8 FAIL**  ",
        f"归因：**{decision['category']}**",
        "",
        "## Lineage",
        "",
        f"- Train candidate 中存在同视频候选，审计标记 `high_risk={lineage['high_risk']}`。",
        f"- Validation 的最大同视频率为 `{lineage['validation_same_video_max']:.4f}`。",
        f"- Same-speaker 元数据不可用：`{lineage['speaker_metadata_available']}`。",
        "",
        "## Clean-listwise",
        "",
        "| Rate | Original learned Acc | Clean learned Acc | Empty Acc | Clean random Acc | Clean source-only Acc |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for rate, values in rates.items():
        lines.append(
            f"| {rate} | {values['original_listwise']['acc']:.4f} | "
            f"{values['clean_listwise_real']['acc']:.4f} | {values['empty']['acc']:.4f} | "
            f"{values['clean_listwise_random']['acc']:.4f} | "
            f"{values['clean_listwise_source_only']['acc']:.4f} |"
        )
    baseline = load(Path("/home/xuanqin/PERouter/runs/evidence_gates/E175_reranker_train_20261112-200353/reranker_retry_20261112-200453/mr0.5/listwise/reranker_training_summary.json"))
    lines += [
        "",
        "## 结论",
        "",
        "同视频 evidence 是 E1.75 OOF 乐观的重要来源之一，清理后 validation 指标改善，但仍低于 empty 和 source-only。",
        "",
        "- 不能继续把完整 leakage correction 之外的 reranker 训练视为可通过。",
        "- 未执行严格 student crossfit，是因为 clean bank 下的较宽松 simplified crossfit 已经未通过验证门禁；不能据此声称严格版必然失败，但当前证据不支持继续进入 E2/E3/E4。",
        "- 当前只能说明 feature-neighbor source 和当前 MLP policy 不可部署。",
        "",
        "## 未执行",
        "",
        "- Strict student crossfit",
        "- Joint-Oracle task metrics",
        "- Use-rate matched random action",
        "- Action calibration",
        "- Text clean bank",
        "- K=10/20",
        "- test 最终评价",
    ]
    md_path = args.closure_dir / "E18_CLOSURE_20261112.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    identity = {
        "branch": git(args.repo, "branch", "--show-current"),
        "head": git(args.repo, "rev-parse", "HEAD"),
        "closure_json_sha256": sha256(json_path),
        "closure_md_sha256": sha256(md_path),
    }
    (args.closure_dir / "E18_identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"decision": decision["decision"], "category": decision["category"], "identity": identity}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
