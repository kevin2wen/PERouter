from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


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
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    args = parser.parse_args()
    closure = args.closure_dir
    closure.mkdir(parents=True, exist_ok=True)
    gate = json.loads((closure / "e175_gate_decision.json").read_text(encoding="utf-8"))
    oof = {}
    for path in sorted(args.train_root.glob("mr*/*/reranker_training_summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        oof[f"{path.parts[-3][2:]}|{data['objective']}"] = data
    payload = {
        "schema_version": 1,
        "date": "2026-11-12",
        "gate": "E1.75",
        "decision": gate["decision"],
        "category": gate["category"],
        "passing_cells": gate["passing_cells"],
        "test_access": "forbidden",
        "validation_root": str(args.validation_root),
        "train_root": str(args.train_root),
        "oof_summaries": oof,
        "gate_details": gate["strategies"],
        "unexecuted": [
            "E2_fixed_gamma_gate", "E3_fusion_factorial_gate",
            "E4_student_bottleneck", "text_reranker", "K_10_20",
            "multi_seed", "cross_dataset", "test_evaluation",
        ],
    }
    json_path = closure / "E175_CLOSURE_20261112.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# ComP-PER E1.75 Candidate Reranker Closure",
        "",
        "日期：2026-11-12  ",
        "数据集：CMU-MOSI  ",
        "评价 split：validation  ",
        "test 访问：forbidden  ",
        "最终判定：**E1.75 FAIL**  ",
        f"归因类别：**{gate['category']}**",
        "",
        "## Train OOF 结果",
        "",
        "| Rate | Objective | Action agreement | Policy utility gain | Oracle gain | Use evidence |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for key, data in sorted(oof.items()):
        rate, objective = key.split("|")
        metric = data["global_oof_metrics"]
        lines.append(
            f"| {rate} | {objective} | {data['mean_action_agreement']:.4f} | "
            f"{metric['policy_delta_utility_mean']:.4f} | "
            f"{metric['oracle_gain_mean']:.4f} | "
            f"{data['mean_use_evidence_rate']:.4f} |"
        )
    lines += [
        "",
        "## Validation 门禁",
        "",
        "| Strategy | Rate | Learned Acc | Empty Acc | Confidence Acc | Random Action Acc | Gain capture Acc | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy, rates in sorted(gate["strategies"].items()):
        for rate, data in sorted(rates.items()):
            lines.append(
                f"| {strategy} | {rate} | {data['learned']['acc']:.4f} | "
                f"{data['empty']['acc']:.4f} | {data['confidence']['acc']:.4f} | "
                f"{data['random_action']['acc']:.4f} | "
                f"{data['gain_capture_acc']:.4f} | {data['pass']} |"
            )
    lines += [
        "",
        "## 结论",
        "",
        "Learned listwise/pairwise reranker 在 train dialogue OOF 上取得正 policy utility，但 validation 上没有迁移为超过 empty、confidence 或 random-action 的收益。",
        "",
        "- Learned real 在缺失率 0.5 上高于 learned random，但仍在 0.7 上失去优势，方向不稳定。",
        "- Learned real 未超过 source-only 的独立内容收益。",
        "- gain capture 为负，未达到 25% 门禁。",
        "- 因此 E1.75 FAIL，禁止进入 E2/E3/E4。",
        "",
        "当前不能说明 ComP student、fusion 或外部语义知识无效；只能说明当前 train-only utility cache、部署输入和小型 reranker 没有在 validation 上泛化。",
        "",
        "## 证据边界",
        "",
        "- Simplified cross-fitting：student 已见过全部 train，只对 reranker 做 dialogue OOF。",
        "- 首轮仅 feature source、K=5、固定 gamma=0.5。",
        "- 未训练 text reranker，未扩展 K=10/20。",
        "- 未访问 test，未执行多 seed 或跨数据集。",
    ]
    md_path = closure / "E175_CLOSURE_20261112.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    identity = {
        "branch": git(args.repo, "branch", "--show-current"),
        "head": git(args.repo, "rev-parse", "HEAD"),
        "closure_json_sha256": sha256(json_path),
        "closure_md_sha256": sha256(md_path),
    }
    (closure / "E175_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"decision": payload["decision"], "category": payload["category"], "identity": identity}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
