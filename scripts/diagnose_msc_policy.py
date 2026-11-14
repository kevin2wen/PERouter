from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.knowledge.msc.cache import load_state_cache
from ComP.knowledge.msc.metrics import interaction_gap, regret_vs_baseline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-cache", required=True)
    parser.add_argument("--subset-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    state = load_state_cache(args.state_cache)
    subset_payload = torch.load(args.subset_cache, map_location="cpu", weights_only=False)
    rows = subset_payload["query_rows"]
    by_query = defaultdict(list)
    for row in rows:
        by_query[row["query_id"]].append(row)
    index_by_query = {qid: index for index, qid in enumerate(state["query_ids"])}
    oracle = []
    always_use = []
    always_skip = []
    pointwise_positive = []
    pointwise_top1 = []
    random_same_budget = []
    gaps = []
    for qid, subset_rows in by_query.items():
        index = index_by_query.get(qid)
        if index is None:
            continue
        mask = state["candidate_mask"][index]
        single = state["utility"][index]
        valid_positions = mask.nonzero(as_tuple=False).flatten().tolist()
        if not valid_positions:
            continue
        by_value = {int(row["subset_mask"]): float(row["utility"]) for row in subset_rows}
        full_value = 0
        for rank in valid_positions:
            full_value |= 1 << rank
        oracle_utility = max(by_value.values())
        always_use_utility = by_value.get(full_value, 0.0)
        positive_ranks = [
            rank for rank in valid_positions if float(single[rank]) > 0
        ]
        pos_value = 0
        for rank in positive_ranks:
            pos_value |= 1 << rank
        pos_utility = by_value.get(pos_value, 0.0)
        top1_rank = max(valid_positions, key=lambda rank: float(single[rank]))
        top1_utility = by_value.get(1 << top1_rank, 0.0)
        budget = len(positive_ranks) if positive_ranks else 1
        random_utils = [
            by_value[value]
            for value in by_value
            if bin(value).count("1") == budget
        ]
        rand_utility = sum(random_utils) / len(random_utils) if random_utils else 0.0
        oracle.append(oracle_utility)
        always_use.append(always_use_utility)
        always_skip.append(0.0)
        pointwise_positive.append(pos_utility)
        pointwise_top1.append(top1_utility)
        random_same_budget.append(rand_utility)
        gaps.append(
            {
                "query_id": qid,
                "oracle_utility": oracle_utility,
                "always_use_utility": always_use_utility,
                "pointwise_positive_utility": pos_utility,
                "oracle_subset_size": bin(
                    max(by_value, key=by_value.get)
                ).count("1"),
                "pointwise_budget": budget,
            }
        )
    oracle_t = torch.tensor(oracle)
    report = {
        "queries": len(oracle),
        "mean_oracle_utility": float(oracle_t.mean()),
        "always_use_regret": regret_vs_baseline(oracle_t, torch.tensor(always_use)),
        "always_skip_regret": regret_vs_baseline(oracle_t, torch.tensor(always_skip)),
        "pointwise_positive_regret": regret_vs_baseline(
            oracle_t, torch.tensor(pointwise_positive)
        ),
        "pointwise_top1_regret": regret_vs_baseline(
            oracle_t, torch.tensor(pointwise_top1)
        ),
        "random_same_budget_regret": regret_vs_baseline(
            oracle_t, torch.tensor(random_same_budget)
        ),
        "oracle_vs_pointwise_gap": float(
            oracle_t.mean() - torch.tensor(pointwise_positive).mean()
        ),
        "oracle_vs_always_use_gap": float(
            oracle_t.mean() - torch.tensor(always_use).mean()
        ),
        "query_rows": gaps,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "msc_diagnostics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    summary = {key: value for key, value in report.items() if key != "query_rows"}
    for key in (
        "always_use_regret",
        "always_skip_regret",
        "pointwise_positive_regret",
        "pointwise_top1_regret",
        "random_same_budget_regret",
    ):
        summary[key] = summary[key]["mean_regret"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
