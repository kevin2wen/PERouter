from __future__ import annotations

from collections import Counter

import torch


def audit_single_labels(cache: dict) -> dict:
    utility = cache["single_utility"]
    mask = cache["candidate_mask"].bool()
    helpful = (utility > 0) & mask
    valid_utility = utility.masked_fill(~mask, float("-inf"))
    best_utility, best_index = valid_utility.max(dim=-1)
    any_positive = helpful.any(dim=-1)
    expected_skip = ~any_positive
    expected_action = torch.where(
        expected_skip, torch.zeros_like(best_index), best_index + 1
    )
    skip_label = cache["skip_label"].bool()
    skip_mismatch = int((skip_label != expected_skip).sum())
    action_mismatch = None
    if "target_action" in cache:
        action_mismatch = int(
            (expected_action != cache["target_action"].long()).sum()
        )
    target = cache["target_modality"].argmax(dim=-1)
    return {
        "query_count": int(mask.size(0)),
        "candidate_count": int(mask.sum()),
        "candidate_helpful_rate": float(helpful.sum() / mask.sum()),
        "query_any_positive_rate": float(any_positive.float().mean()),
        "query_skip_rate": float(expected_skip.float().mean()),
        "stored_skip_rate": float(skip_label.float().mean()),
        "skip_label_mismatch": skip_mismatch,
        "target_action_mismatch": action_mismatch,
        "utility_mean": float(valid_utility[torch.isfinite(valid_utility)].mean()),
        "utility_std": float(valid_utility[torch.isfinite(valid_utility)].std()),
        "best_utility_mean": float(best_utility.mean()),
        "best_utility_positive_rate": float((best_utility > 0).float().mean()),
        "by_target_modality": {
            name: {
                "queries": int((target == index).sum()),
                "skip_rate": (
                    float(expected_skip[target == index].float().mean())
                    if (target == index).any() else None
                ),
            }
            for index, name in enumerate(("audio", "text", "video"))
        },
        "by_missing_pattern": {
            str(pattern): int(count)
            for pattern, count in Counter(cache["missing_pattern"].tolist()).items()
        },
    }
