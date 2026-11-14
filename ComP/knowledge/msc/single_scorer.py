from __future__ import annotations

import torch
import torch.nn as nn


class MSCSingleScorer(nn.Module):
    """Predict one task-utility score per candidate under a missing state.

    S0: state + metadata (candidate content zeroed).
    S1: state + metadata + candidate content.
    S2: S1 + one cross-candidate self-attention layer before scoring.
    """

    def __init__(
        self,
        candidate_dim: int,
        state_dim: int,
        prompt_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_semantic: bool = True,
        use_cross_attention: bool = False,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if candidate_dim <= 0 or state_dim <= 0 or prompt_dim <= 0:
            raise ValueError("candidate/state/prompt dims must be positive")
        self.candidate_dim = int(candidate_dim)
        self.state_dim = int(state_dim)
        self.prompt_dim = int(prompt_dim)
        self.use_semantic = bool(use_semantic)
        self.use_cross_attention = bool(use_cross_attention)
        token_dim = candidate_dim + prompt_dim + state_dim + 2
        self.token_projection = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention = (
            nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
            if use_cross_attention
            else None
        )
        self.scorer = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        query_state: torch.Tensor,       # [N,Ds]
        query_prompt: torch.Tensor,      # [N,Dp]
        candidate_features: torch.Tensor,  # [N,K,Dc]
        candidate_mask: torch.Tensor,    # [N,K]
        source_confidence: torch.Tensor,  # [N,K]
        conflict_score: torch.Tensor | None = None,  # [N,K]
    ) -> torch.Tensor:                    # [N,K]
        n, k, dim = candidate_features.shape
        if dim != self.candidate_dim:
            raise ValueError("candidate feature dim mismatch")
        content = (
            candidate_features
            if self.use_semantic
            else torch.zeros_like(candidate_features)
        )
        prompt = query_prompt.unsqueeze(1).expand(-1, k, -1)
        state = query_state.unsqueeze(1).expand(-1, k, -1)
        conflict = (
            conflict_score
            if conflict_score is not None
            else torch.zeros_like(source_confidence)
        )
        token = torch.cat(
            [content, prompt, state, source_confidence.unsqueeze(-1),
             conflict.unsqueeze(-1)],
            dim=-1,
        )
        hidden = self.token_projection(token)
        if self.attention is not None:
            padded = hidden.masked_fill(~candidate_mask.bool().unsqueeze(-1), 0.0)
            attended, _ = self.attention(
                padded, padded, padded,
                key_padding_mask=~candidate_mask.bool(),
            )
            hidden = hidden + attended
        scores = self.scorer(hidden).squeeze(-1)
        return scores.masked_fill(~candidate_mask.bool(), float("-inf"))
