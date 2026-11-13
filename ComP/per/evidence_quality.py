from __future__ import annotations

from typing import Dict, Tuple

import torch


def candidate_statistics(
    utility: torch.Tensor,
    prompt_utility: torch.Tensor,
    active: torch.BoolTensor,
) -> Dict[str, torch.Tensor]:
    """Compute best-candidate diagnostics for active missing positions.

    utility:        [B,T,3,K] teacher evidence utility
    prompt_utility: [B,T,3] teacher prompt utility
    active:         [B,T,3] valid missing position with evidence available
    """
    if utility.ndim != 4:
        raise ValueError("utility must be [B,T,3,K]")
    if utility.size(2) != 3:
        raise ValueError("utility must contain three target modalities")
    if prompt_utility.shape != utility.shape[:3]:
        raise ValueError("prompt_utility must be [B,T,3]")
    if active.shape != utility.shape[:3]:
        raise ValueError("active must be [B,T,3]")
    if active.dtype != torch.bool:
        raise TypeError("active must be bool")
    if not torch.isfinite(utility).all() or not torch.isfinite(prompt_utility).all():
        raise FloatingPointError("utility contains non-finite values")

    masked_utility = utility.masked_fill(
        ~active.unsqueeze(-1), torch.finfo(utility.dtype).min
    )
    best_utility, best_rank = masked_utility.max(dim=-1)
    valid = active.clone()
    best_utility = torch.where(valid, best_utility, torch.zeros_like(best_utility))
    gain = torch.where(
        valid,
        best_utility - prompt_utility,
        torch.zeros_like(prompt_utility),
    )
    helpful = valid & (gain > 0)
    return {
        "active": active,
        "best_utility": best_utility,
        "best_rank": best_rank.long(),
        "oracle_gain": gain,
        "has_helpful_candidate": helpful,
    }


def recall_at_k(
    utility: torch.Tensor,
    prompt_utility: torch.Tensor,
    active: torch.BoolTensor,
) -> Dict[int, float]:
    """Fraction of active positions whose top-k prefix contains a helpful candidate."""
    if utility.ndim != 4 or utility.size(2) != 3:
        raise ValueError("utility must be [B,T,3,K]")
    if prompt_utility.shape != utility.shape[:3] or active.shape != utility.shape[:3]:
        raise ValueError("prompt_utility and active must be [B,T,3]")
    top_k = utility.size(-1)
    results: Dict[int, float] = {}
    count = int(active.sum().item())
    for k in range(1, top_k + 1):
        prefix_best = utility[..., :k].masked_fill(
            ~active.unsqueeze(-1), torch.finfo(utility.dtype).min
        ).max(dim=-1).values
        helpful = active & (prefix_best > prompt_utility)
        results[k] = float(helpful.sum().item() / count) if count else 0.0
    return results


def rank_histogram(best_rank: torch.LongTensor, active: torch.BoolTensor, top_k: int) -> Dict[int, int]:
    if best_rank.shape != active.shape:
        raise ValueError("best_rank and active must be [B,T,3]")
    counts = {rank: 0 for rank in range(top_k)}
    for rank in best_rank[active].detach().cpu().tolist():
        rank = int(rank)
        counts[rank] = counts.get(rank, 0) + 1
    return counts


def binary_label_consistency(
    candidate_labels: torch.Tensor,
    target_labels: torch.Tensor,
    active: torch.BoolTensor,
) -> float:
    if candidate_labels.shape != target_labels.shape or target_labels.shape != active.shape:
        raise ValueError("candidate labels, target labels, and active must match")
    if not active.any():
        return 0.0
    candidate_binary = candidate_labels > 0
    target_binary = target_labels > 0
    return float((candidate_binary[active] == target_binary[active]).float().mean().item())
