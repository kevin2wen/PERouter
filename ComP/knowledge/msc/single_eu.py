from __future__ import annotations

import torch
import torch.nn as nn

from .single_scorer import MSCSingleScorer
from .state import MSCStateEncoder


class MSCSingleEU(nn.Module):
    """Missing-state-conditioned single-evidence utility model.

    Utility scorer + independent skip head. No subset scorer is used in this
    deployment path (the set scorer remains available for diagnostics only).
    """

    def __init__(
        self,
        candidate_dim: int,
        prompt_dim: int,
        state_dim: int = 128,
        hidden_dim: int = 256,
        *,
        use_semantic: bool = True,
        use_cross_attention: bool = False,
        use_skip_head: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.state_encoder = MSCStateEncoder(hidden_dim=state_dim)
        self.scorer = MSCSingleScorer(
            candidate_dim,
            state_dim,
            prompt_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_semantic=use_semantic,
            use_cross_attention=use_cross_attention,
        )
        self.use_skip_head = use_skip_head
        self.skip_head = None
        if use_skip_head:
            self.skip_head = nn.Sequential(
                nn.Linear(candidate_dim + prompt_dim + state_dim + 6, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(
        self,
        availability: torch.Tensor,
        target_modality: torch.Tensor,
        missing_pattern: torch.Tensor,
        missing_rate: torch.Tensor,
        query_prompt: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        source_confidence: torch.Tensor,
        conflict_score: torch.Tensor | None = None,
    ) -> dict:
        state = self.state_encoder(
            availability, target_modality, missing_pattern, missing_rate
        )
        utility = self.scorer(
            state,
            query_prompt,
            candidate_features,
            candidate_mask,
            source_confidence,
            conflict_score,
        )
        skip_logit = torch.zeros(
            utility.size(0), device=utility.device, dtype=utility.dtype
        )
        if self.skip_head is not None:
            valid = candidate_mask.bool()
            pooled = (candidate_features * valid.unsqueeze(-1)).sum(1) / valid.sum(
                1, keepdim=True
            ).clamp_min(1)
            conflict = (
                conflict_score
                if conflict_score is not None
                else torch.zeros_like(source_confidence)
            )
            top2 = torch.topk(
                utility.masked_fill(~valid, -1e9), k=min(2, utility.size(1)), dim=-1
            ).values
            if top2.size(1) == 1:
                top2 = torch.cat([top2, top2], dim=-1)
            stats = torch.stack(
                [
                    source_confidence.mean(1),
                    source_confidence.max(1).values,
                    conflict.mean(1),
                    conflict.max(1).values,
                ],
                dim=-1,
            )
            skip_input = torch.cat(
                [pooled, query_prompt, state, top2, stats], dim=-1
            )
            skip_logit = self.skip_head(skip_input).squeeze(-1)
        return {"utility": utility, "skip_logit": skip_logit, "state": state}


class SingleEUSelector:
    """Top-1 card or skip; skip threshold is calibrated on inner validation."""

    def __init__(self, tau_skip: float = 0.5, utility_threshold: float = 0.0):
        self.tau_skip = float(tau_skip)
        self.utility_threshold = float(utility_threshold)

    def select(
        self,
        utility: torch.Tensor,        # [N,K]
        candidate_mask: torch.Tensor, # [N,K]
        skip_logit: torch.Tensor,     # [N]
    ) -> dict:
        valid = candidate_mask.bool()
        masked = utility.masked_fill(~valid, -1e9)
        best_utility, best_index = masked.max(dim=-1)
        has_valid = valid.any(dim=-1)
        skip_probability = torch.sigmoid(skip_logit)
        skip = (
            (~has_valid)
            | (best_utility <= self.utility_threshold)
            | (skip_probability >= self.tau_skip)
        )
        selected = torch.where(skip, torch.full_like(best_index, -1), best_index)
        return {
            "skip": skip,
            "skip_probability": skip_probability,
            "utility": best_utility,
            "selected_index": selected,
        }
