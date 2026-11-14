from __future__ import annotations

import torch


def subset_utility_regret(
    subset_utility: torch.Tensor,   # [N,S] true utilities, -inf for invalid
) -> dict:
    oracle, index = subset_utility.max(dim=-1)
    valid = torch.isfinite(oracle)
    return {
        "oracle_utility_mean": float(oracle[valid].mean()),
        "oracle_utility": oracle,
        "oracle_index": index,
        "valid_queries": int(valid.sum()),
    }


def regret_vs_baseline(
    oracle: torch.Tensor,
    achieved: torch.Tensor,
) -> dict:
    valid = torch.isfinite(oracle)
    regret = (oracle - achieved)[valid]
    return {
        "mean_regret": float(regret.mean()),
        "median_regret": float(regret.median()),
        "regret": [float(value) for value in regret],
    }


def action_use_skip(selected_mask: torch.Tensor) -> dict:
    use = selected_mask.any(dim=-1)
    return {
        "use_rate": float(use.float().mean()),
        "skip_rate": float((~use).float().mean()),
    }


def interaction_gap(
    subset_utility: torch.Tensor,
    subset_masks: torch.Tensor,
    single_utility: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> dict:
    """Mean of U(A) - sum_i in A u_i over non-empty valid subsets."""
    valid = candidate_mask.bool()
    gaps = []
    for index in range(subset_utility.size(0)):
        utilities = subset_utility[index]
        finite = torch.isfinite(utilities)
        for slot in finite.nonzero(as_tuple=False).flatten():
            mask = subset_masks[index, slot]
            additive = (single_utility[index] * mask.to(single_utility.dtype)).sum()
            gaps.append(float(utilities[slot] - additive))
    if not gaps:
        return {"interaction_mean": 0.0, "interaction_count": 0}
    values = torch.tensor(gaps, dtype=torch.float32)
    return {
        "interaction_mean": float(values.mean()),
        "interaction_std": float(values.std(unbiased=True)),
        "interaction_count": int(values.numel()),
    }
