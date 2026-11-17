from __future__ import annotations

import torch
import torch.nn.functional as F


def state_features(
    availability: torch.Tensor,      # [N,3]
    target_onehot: torch.Tensor,     # [N,3]
    missing_pattern: torch.Tensor,   # [N] long (0..7)
    missing_rate: torch.Tensor,      # [N]
    empty_logits: torch.Tensor,      # [N,C] classification or [N,1] regression
    task: str,
) -> torch.Tensor:
    """State vector: mask/target/pattern/rate + empty base-state statistics.

    Contains no target-modality raw content and no label.
    """
    n = availability.size(0)
    observed = availability.float()
    target = target_onehot.float()
    pattern = F.one_hot(missing_pattern.long().clamp(0, 7), num_classes=8).float()
    rate = missing_rate.float().reshape(n, 1)
    if task == "classification":
        probs = torch.softmax(empty_logits.float(), dim=-1)
        top2 = probs.topk(min(2, probs.size(-1)), dim=-1).values
        margin = (
            (top2[:, 0] - top2[:, 1]).unsqueeze(-1)
            if top2.size(1) > 1
            else top2[:, :1]
        )
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1, keepdim=True)
        base = torch.cat([probs, margin, entropy], dim=-1)
    else:
        prediction = empty_logits.float().reshape(n, 1)
        base = torch.cat([prediction, prediction.abs()], dim=-1)
    interaction = (observed.unsqueeze(-1) * target.unsqueeze(-2)).flatten(1)
    return torch.cat([observed, target, pattern, rate, interaction, base], dim=-1)
