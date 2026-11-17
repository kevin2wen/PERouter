from __future__ import annotations

import torch
import torch.nn as nn


class FilterV2Model(nn.Module):
    """Joint query-card encoder with one cross-candidate attention layer.

    Candidate features are concatenated with query state and metadata, encoded
    jointly (cross-encoder), then mixed across the K candidates. No card ID or
    sample ID embedding is used.
    """

    def __init__(
        self,
        candidate_dim: int,
        state_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        aux_mae: bool = True,
    ) -> None:
        super().__init__()
        self.joint = nn.Sequential(
            nn.Linear(candidate_dim + state_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.utility_head = nn.Sequential(
            nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.corr_head = nn.Sequential(
            nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.mae_head = (
            nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, 1))
            if aux_mae
            else None
        )
        self.skip_head = nn.Sequential(
            nn.Linear(hidden_dim + state_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        candidate_vectors: torch.Tensor,   # [N,K,D]
        candidate_mask: torch.Tensor,      # [N,K]
        state_vector: torch.Tensor,        # [N,Ds]
        source_confidence: torch.Tensor,   # [N,K]
        retrieval_score: torch.Tensor | None = None,  # [N,K]
    ) -> dict:
        n, k, _ = candidate_vectors.shape
        mask = candidate_mask.bool()
        state = state_vector.unsqueeze(1).expand(-1, k, -1)
        retrieval = (
            retrieval_score
            if retrieval_score is not None
            else torch.zeros_like(source_confidence)
        )
        metadata = torch.stack(
            [source_confidence.float(), retrieval.float()], dim=-1
        )
        tokens = self.joint(
            torch.cat([candidate_vectors, state, metadata], dim=-1)
        )
        padded = tokens.masked_fill(~mask.unsqueeze(-1), 0.0)
        attended, _ = self.attention(
            padded, padded, padded, key_padding_mask=~mask
        )
        hidden = self.norm(tokens + attended)
        utility = self.utility_head(hidden).squeeze(-1)
        corr = self.corr_head(hidden).squeeze(-1)
        mae = self.mae_head(hidden).squeeze(-1) if self.mae_head is not None else None
        if mae is not None:
            mae = mae.masked_fill(~mask, 0.0)
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(
            1, keepdim=True
        ).clamp_min(1)
        stats = torch.stack(
            [source_confidence.float().mean(1), source_confidence.float().max(1).values],
            dim=-1,
        )
        skip_logit = self.skip_head(
            torch.cat([pooled, state_vector, stats], dim=-1)
        ).squeeze(-1)
        return {
            "utility": utility,
            "correctness": corr,
            "mae_utility": mae,
            "skip_logit": skip_logit,
        }
