from __future__ import annotations

import torch


class FilterV2ActionSelector:
    """Choose argmax utility above a calibrated threshold, else skip."""

    def __init__(self, tau: float = 0.0) -> None:
        self.tau = float(tau)

    def select(self, utility: torch.Tensor, mask: torch.Tensor) -> dict:
        mask = mask.bool()
        masked = utility.masked_fill(~mask, float("-inf"))
        best, index = masked.max(dim=-1)
        has_valid = mask.any(dim=-1)
        use = has_valid & (best > self.tau)
        selected = torch.where(use, index, torch.full_like(index, -1))
        return {
            "use": use,
            "skip": ~use,
            "index": selected,
            "max_utility": best,
        }
