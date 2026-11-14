from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .set_scorer import MSCSetScorer


class MSCPolicy(nn.Module):
    """Setwise evidence policy: set scorer plus an independent skip head.

    The skip head consumes pooled candidate content, conflict/confidence
    statistics and the query state; it does not read the subset scores, so
    skip is a separate risk decision rather than a pseudo-candidate.
    """

    def __init__(
        self,
        candidate_dim: int,
        state_dim: int,
        top_k: int,
        *,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_set_scorer: bool = True,
        use_skip_head: bool = True,
    ) -> None:
        super().__init__()
        self.top_k = int(top_k)
        self.set_scorer = (
            MSCSetScorer(
                candidate_dim,
                state_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            if use_set_scorer
            else None
        )
        self.use_set_scorer = use_set_scorer
        self.use_skip_head = use_skip_head
        self.skip_head = None
        if use_skip_head:
            self.skip_head = nn.Sequential(
                nn.Linear(candidate_dim + state_dim + 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(
        self,
        candidate_vectors: torch.Tensor,   # [N,K,D]
        candidate_mask: torch.Tensor,      # [N,K]
        state_vector: torch.Tensor,        # [N,Ds]
        subset_masks: torch.Tensor,        # [N,S,K]
        source_confidence: Optional[torch.Tensor] = None,  # [N,K]
        conflict_score: Optional[torch.Tensor] = None,     # [N,K]
    ) -> dict:
        if self.set_scorer is None:
            raise RuntimeError("set scorer is disabled")
        subset_scores = self.set_scorer(
            candidate_vectors, candidate_mask, state_vector, subset_masks
        )
        skip_logit = torch.zeros(
            candidate_vectors.size(0), device=candidate_vectors.device
        )
        if self.skip_head is not None:
            valid = candidate_mask.bool()
            pooled = (candidate_vectors * valid.unsqueeze(-1)).sum(1) / valid.sum(
                1, keepdim=True
            ).clamp_min(1)
            conf = source_confidence if source_confidence is not None else torch.zeros(
                candidate_vectors.size(0), self.top_k, device=candidate_vectors.device
            )
            conflict = conflict_score if conflict_score is not None else torch.zeros_like(
                conf
            )
            stats = torch.stack(
                [conf.mean(1), conf.max(1).values, conflict.max(1).values], dim=-1
            )
            skip_logit = self.skip_head(
                torch.cat([pooled, state_vector, stats], dim=-1)
            ).squeeze(-1)
        return {"subset_scores": subset_scores, "skip_logit": skip_logit}


class MSCActionSelector:
    """Select a subset action or skip from policy outputs."""

    def __init__(self, tau_skip: float = 0.5) -> None:
        if not 0.0 <= tau_skip <= 1.0:
            raise ValueError("tau_skip must be in [0,1]")
        self.tau_skip = float(tau_skip)

    def select(
        self,
        subset_scores: torch.Tensor,  # [N,S]
        subset_masks: torch.Tensor,   # [N,S,K]
        skip_logit: torch.Tensor,     # [N]
    ) -> dict:
        if subset_scores.ndim != 2 or subset_masks.ndim != 3:
            raise ValueError("expected [N,S] scores and [N,S,K] masks")
        skip_probability = torch.sigmoid(skip_logit)
        best_score, best_index = subset_scores.max(dim=-1)
        has_valid = torch.isfinite(best_score)
        no_positive = best_score <= 0.0
        skip = (skip_probability > self.tau_skip) | ~has_valid | no_positive
        chosen_mask = torch.zeros(
            subset_masks.size(0), subset_masks.size(2),
            dtype=torch.bool, device=subset_masks.device,
        )
        use_rows = (~skip).nonzero(as_tuple=False).flatten()
        if use_rows.numel():
            chosen_mask[use_rows] = subset_masks[
                use_rows, best_index[use_rows]
            ]
        return {
            "skip": skip,
            "skip_probability": skip_probability,
            "subset_mask": chosen_mask,
            "subset_index": torch.where(skip, torch.full_like(best_index, -1), best_index),
            "score": best_score,
            "reason": torch.where(
                skip,
                torch.where(
                    has_valid,
                    torch.where(
                        skip_probability > self.tau_skip,
                        torch.full_like(best_index, 1),  # threshold
                        torch.full_like(best_index, 2),  # no positive utility
                    ),
                    torch.full_like(best_index, 3),      # no valid subset
                ),
                torch.full_like(best_index, 0),          # use best subset
            ),
        }
