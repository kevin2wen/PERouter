from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ComP.per.e175_gate import evaluate_strategy


def load_result(root: Path, rate: str, directory: str):
    return json.loads((root / f"mr{rate}" / directory / "result.json").read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--e15-root", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--closure-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.validation_root
    args.closure_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = {
        "schema_version": 1,
        "split": "validation",
        "test_access": "forbidden",
        "strategies": {},
    }
    strategies = {}
    for strategy in ("listwise", "pairwise"):
        strategies[strategy] = {}
        for rate in ("0.5", "0.7"):
            empty = load_result(root, rate, "raw_empty")["metrics"]
            real = load_result(root, rate, "raw_feature")["metrics"]
            random_evidence_raw = load_result(root, rate, "raw_random")["metrics"]
            shuffled = load_result(root, rate, "raw_shuffled")["metrics"]
            source_only_raw = load_result(root, rate, "raw_source_only")["metrics"]
            confidence = load_result(root, rate, "confidence_feature")["metrics"]
            random_action = load_result(root, rate, "random_action_feature")["metrics"]
            learned = load_result(root, rate, f"{strategy}_feature")["metrics"]
            learned_random = load_result(root, rate, f"{strategy}_random")["metrics"]
            learned_shuffled = load_result(root, rate, f"{strategy}_shuffled")["metrics"]
            learned_source = load_result(root, rate, f"{strategy}_source_only")["metrics"]
            oracle_payload = json.loads(
                (args.e15_root / "original_feature" / f"mr{rate}" / "oracle_v2" / "oracle_v2_summary.json").read_text(encoding="utf-8")
            )
            oracle_task = oracle_payload["oracle_task_metrics"]
            result = evaluate_strategy(
                learned, empty, learned_random, learned_source, confidence,
                random_action, oracle_task,
            )
            result["rate"] = rate
            result["learned"] = learned
            result["empty"] = empty
            result["real_raw"] = real
            result["random_evidence_raw"] = random_evidence_raw
            result["shuffled_raw"] = shuffled
            result["source_only_raw"] = source_only_raw
            result["confidence"] = confidence
            result["random_action"] = random_action
            result["learned_random"] = learned_random
            result["learned_shuffled"] = learned_shuffled
            result["learned_source_only"] = learned_source
            result["oracle_task"] = oracle_task
            strategies[strategy][rate] = result
            rows.append({"strategy": strategy, "rate": rate, **learned})
    fields = sorted({key for row in rows for key in row})
    with (args.closure_dir / "e175_validation_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    passing = [
        f"{strategy}@{rate}"
        for strategy, by_rate in strategies.items()
        for rate, value in by_rate.items()
        if value["pass"]
    ]
    decision = {
        "decision": "PASS" if passing else "FAIL",
        "category": "LEARNED_RERANKER_GATE" if passing else "LEARNED_RERANKER_DID_NOT_GENERALIZE",
        "passing_cells": passing,
        "strategies": strategies,
        "test_access": "forbidden",
        "summary": summary,
    }
    (args.closure_dir / "e175_gate_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({
        "decision": decision["decision"],
        "category": decision["category"],
        "passing_cells": passing,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
