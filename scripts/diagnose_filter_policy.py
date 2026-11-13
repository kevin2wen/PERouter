from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def roc_auc(scores, labels) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if labels.size == 0 or labels.all() or (~labels).all():
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    positive = ranks[labels].sum()
    n_pos = labels.sum()
    n_neg = labels.size - n_pos
    return float(
        (positive - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--reranker-dir", required=True)
    parser.add_argument("--objective", choices=("listwise", "pairwise"),
                        default="listwise")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    cache = torch.load(args.candidate_cache, map_location="cpu", weights_only=False)
    oof = torch.load(
        Path(args.reranker_dir) / "oof_predictions.pt",
        map_location="cpu",
        weights_only=False,
    )
    logits = oof["logits"]
    mask = oof["candidate_mask"].bool()
    utility = oof["delta_utility"]
    valid = mask.any(dim=-1)
    logits = logits[valid]
    mask = mask[valid]
    utility = utility[valid]
    helpful = (utility[:, 1:] > 0) & mask
    actions = logits.argmax(dim=-1)
    ranks = actions - 1
    selected_utility = torch.zeros(logits.size(0))
    use = actions > 0
    rows = torch.arange(logits.size(0))
    selected_utility[use] = utility[rows[use], actions[use]]
    best_utility = utility[:, 1:].masked_fill(~mask, float("-inf")).max(dim=-1).values
    false_use = use & (selected_utility < 0)
    false_skip = (~use) & (best_utility > 0)
    candidate_scores = []
    candidate_labels = []
    for index in range(logits.size(0)):
        for rank in range(mask.size(1)):
            if bool(mask[index, rank]):
                candidate_scores.append(float(logits[index, rank + 1]))
                candidate_labels.append(bool(helpful[index, rank]))
    skip_prob = torch.softmax(logits, dim=-1)[:, 0]
    thresholds = np.round(np.arange(-2.0, 2.01, 0.1), 2)
    sweep = []
    best = (float("-inf"), None, None)
    for threshold in thresholds:
        margin = logits[:, 0] - logits[:, 1:].masked_fill(~mask, float("-inf")).max(dim=-1).values
        choose_skip = margin > float(threshold)
        chosen_utility = torch.where(
            choose_skip,
            torch.zeros_like(selected_utility),
            selected_utility,
        )
        use_rate = float((~choose_skip).float().mean())
        mean_utility = float(chosen_utility.mean())
        sweep.append(
            {"threshold": float(threshold), "use_rate": use_rate,
             "mean_utility": mean_utility}
        )
        if mean_utility > best[0]:
            best = (mean_utility, float(threshold), use_rate)
    with (Path(args.output_dir) / "threshold_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep[0].keys()))
        writer.writeheader()
        writer.writerows(sweep)
    selected_helpful = use & (selected_utility > 0)
    precision = float(selected_helpful.sum() / max(1, int(use.sum())))
    recall = float(selected_helpful.sum() / max(1, int(helpful.sum())))
    report = {
        "objective": args.objective,
        "records": int(logits.size(0)),
        "candidates": int(mask.sum()),
        "helpful_candidates": int(helpful.sum()),
        "helpfulness_auc": roc_auc(candidate_scores, candidate_labels),
        "use_rate": float(use.float().mean()),
        "skip_rate": float((~use).float().mean()),
        "selected_precision": precision,
        "selected_recall": recall,
        "false_use_rate": float(false_use.float().mean()),
        "false_skip_rate": float(false_skip.float().mean()),
        "mean_selected_utility": float(selected_utility.mean()),
        "mean_best_utility": float(best_utility.mean()),
        "skip_prob_mean": float(skip_prob.mean()),
        "skip_prob_p10": float(np.quantile(skip_prob.numpy(), 0.1)),
        "skip_prob_p50": float(np.quantile(skip_prob.numpy(), 0.5)),
        "skip_prob_p90": float(np.quantile(skip_prob.numpy(), 0.9)),
        "oracle_action_utility": float(best_utility.clamp_min(0).mean()),
        "best_threshold": {"threshold": best[1], "mean_utility": best[0],
                           "use_rate": best[2]},
    }
    (Path(args.output_dir) / "filter_policy_diagnostics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
