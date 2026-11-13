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


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/home/xuanqin/PERouter"))
    parser.add_argument("--closure-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo
    root = repo / "runs/evidence_gates"
    sensitivity_original = root / "E15_sensitivity_20261112-192550"
    candidate_root = root / "E15_candidate_quality_20261112-192837"
    oracle_root = root / "E15_oracle_v2_20261112-193026"
    adapter_root = root / "E15_adapters_20261112-193154"
    sensitivity_adapters = root / "E15_sensitivity_adapters_20261112-193610"
    matrix_root = root / "E15_evidence_matrix_grouped_20261112-194056"

    candidate = {}
    for path in sorted(candidate_root.glob("*/candidate_quality_summary.json")):
        data = load(path)
        candidate[f"{data['evidence_source']}@{path.parent.name.split('_')[-1]}"] = data
    aggregate = load(matrix_root / "e15_aggregate.json")
    adapter_summaries = {}
    for source in ("feature", "text"):
        adapter_summaries[source] = load(adapter_root / source / "adapter_summary.json")

    sensitivity = {}
    for label, run_root in (("original", sensitivity_original), ("adapter", sensitivity_adapters)):
        sensitivity[label] = {}
        for path in sorted(run_root.glob("*/evidence_sensitivity_summary.json")):
            data = load(path)
            sensitivity[label][f"{data['evidence_source']}@{path.parent.name.split('_')[-1]}"] = data["comparisons"]

    task_gains = []
    teacher_metrics = []
    for key, value in aggregate["comparisons"].items():
        if value.get("task_oracle_minus_empty"):
            task_gains.append({**value["task_oracle_minus_empty"], "variant_rate": key})
        if value.get("teacher_oracle_minus_empty"):
            teacher_metrics.append({**value["teacher_oracle_minus_empty"], "variant_rate": key})

    helpful_rates = [value["helpful_candidate_rate"] for value in candidate.values()]
    confidence_rhos = [
        value["confidence_utility_spearman"]["rho"] for value in candidate.values()
    ]
    adapter_text = {
        "0.5": aggregate["comparisons"]["adapter_text|0.5"],
        "0.7": aggregate["comparisons"]["adapter_text|0.7"],
    }
    gate = {
        "schema_version": 1,
        "date": "2026-11-12",
        "gate": "E1.5",
        "decision": "FAIL",
        "category": "RETRIEVER_OR_RERANKER_WEAK",
        "secondary_categories": ["EVIDENCE_SIGNAL_ONLY_IN_METADATA"],
        "reason": (
            "Oracle-Task establishes a positive candidate-pool upper bound, but real evidence "
            "does not exceed random by the required margin; text adapter gains are also matched "
            "by source_only, and confidence/utility ranking is weak."
        ),
        "test_access": "forbidden",
        "validation_only": True,
        "candidate_helpful_rate_min": min(helpful_rates),
        "candidate_helpful_rate_max": max(helpful_rates),
        "confidence_utility_rho_min": min(confidence_rhos),
        "confidence_utility_rho_max": max(confidence_rhos),
        "oracle_task_gains_vs_empty": task_gains,
        "oracle_teacher_gains_vs_empty": teacher_metrics,
        "adapter_text_comparisons": adapter_text,
        "adapter_summaries": adapter_summaries,
        "unexecuted": [
            "E2_fixed_gamma_gate",
            "E3_fusion_factorial_gate",
            "E4_student_bottleneck",
            "test_evaluation",
            "multi_seed",
            "cross_dataset",
        ],
    }
    json_path = args.closure_dir / "E15_CLOSURE_20261112.json"
    json_path.write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# ComP-PER E1.5 Closure",
        "",
        "日期：2026-11-12  ",
        "数据集：CMU-MOSI  ",
        "split：validation  ",
        "test：未访问  ",
        "最终判定：**E1.5 FAIL**  ",
        "归因类别：**RETRIEVER_OR_RERANKER_WEAK**",
        "",
        "## A. 敏感性",
        "",
        "- Evidence 对 logits 的实际影响非零，梯度进入 evidence encoder。",
        "- Feature evidence 的敏感性高于 text evidence。",
        "- Text adapter 后 real-vs-random 的 logits 差异没有实质提升。",
        "",
        "## B. 候选质量",
        "",
        f"- Helpful candidate rate：{min(helpful_rates):.4f} – {max(helpful_rates):.4f}。",
        f"- Confidence 与 utility 的 Spearman ρ：{min(confidence_rhos):.4f} 至 {max(confidence_rhos):.4f}。",
        "- 候选池中存在 helpful candidates，但 confidence 不能对 utility 做正向排序。",
        "",
        "## C. Oracle v2",
        "",
        "- Oracle-Task 相对 empty 为正，说明候选池存在任务层面的可选择性。",
        "- Oracle-Teacher 在多数配置下负增益，说明当前 teacher utility 不能承担 reranker。",
        "- 每个目标模态允许独立选择 rank/gamma 后，仍无法由 real evidence 复现 task Oracle 收益。",
        "",
        "## D. Source-specific adapter",
        "",
        f"- Feature adapter SHA256：`{adapter_summaries['feature']['adapter_sha256']}`。",
        f"- Text adapter SHA256：`{adapter_summaries['text']['adapter_sha256']}`。",
        "- 每个 adapter 参数量：2,164,224。",
        "- Feature adapter 未恢复有效收益。",
        "- Text adapter 改善连续误差，但 classification 相对 source_only 没有增量。",
        "",
        "## E. Evidence 对照",
        "",
        "### Adapter text, ratio 0.5",
        f"- real-random：`{json.dumps(adapter_text['0.5']['real_minus_random'], sort_keys=True)}`",
        f"- real-source_only：`{json.dumps(adapter_text['0.5']['real_minus_source_only'], sort_keys=True)}`",
        "",
        "### Adapter text, ratio 0.7",
        f"- real-random：`{json.dumps(adapter_text['0.7']['real_minus_random'], sort_keys=True)}`",
        f"- real-source_only：`{json.dumps(adapter_text['0.7']['real_minus_source_only'], sort_keys=True)}`",
        "",
        "## 结论",
        "",
        "候选池不是完全无信息，但现有 TF-IDF/feature retrieval 和 confidence ranking 无法选择出对任务有利的 evidence。Text adapter 的部分收益可由 source-only 解释。因此 E1.5 不通过，禁止进入 E2/E3/E4。",
        "",
        "## 证据边界",
        "",
        "- 不能据此否定 Semantic KB、LMM verifier 或其他 reranker。",
        "- 不能据此宣称所有 fusion 或 student 无效。",
        "- 当前结果只覆盖 CMU-MOSI validation、seed 66、缺失率 0.5/0.7。",
        "",
        "## 工件",
        "",
        f"- Sensitivity original：`{sensitivity_original}`",
        f"- Candidate quality：`{candidate_root}`",
        f"- Oracle v2：`{oracle_root}`",
        f"- Adapters：`{adapter_root}`",
        f"- Adapter sensitivity：`{sensitivity_adapters}`",
        f"- Grouped evidence matrix：`{matrix_root}`",
    ]
    md_path = args.closure_dir / "E15_CLOSURE_20261112.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    identity = {
        "branch": git_output(repo, "branch", "--show-current"),
        "head": git_output(repo, "rev-parse", "HEAD"),
        "closure_json_sha256": sha256(json_path),
        "closure_md_sha256": sha256(md_path),
    }
    (args.closure_dir / "E15_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"gate": gate, "identity": identity}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
