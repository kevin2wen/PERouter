from __future__ import annotations

import torch
import torch.nn as nn


class MSCSetScorer(nn.Module):
    """Score non-empty candidate subsets conditioned on query state.

    Permutation-invariant: each subset is represented by the mean of its
    selected candidate vectors concatenated with the state vector. Empty sets
    are handled by the skip head, never by this scorer.
    """

    def __init__(
        self,
        candidate_dim: int,
        state_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if candidate_dim <= 0 or state_dim <= 0:
            raise ValueError("candidate_dim and state_dim must be positive")
        self.candidate_dim = int(candidate_dim)
        self.state_dim = int(state_dim)
        self.mlp = nn.Sequential(
            nn.Linear(candidate_dim + state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        candidate_vectors: torch.Tensor,  # [N,K,D]
        candidate_mask: torch.Tensor,     # [N,K]
        state_vector: torch.Tensor,       # [N,Ds]
        subset_masks: torch.Tensor,       # [N,S,K]
    ) -> torch.Tensor:                    # [N,S]
        if candidate_vectors.ndim != 3 or subset_masks.ndim != 3:
            raise ValueError("expected [N,K,D] candidates and [N,S,K] subsets")
        if subset_masks.size(0) != candidate_vectors.size(0):
            raise ValueError("subset and candidate batch sizes differ")
        if subset_masks.size(2) != candidate_vectors.size(1):
            raise ValueError("subset mask width must equal K")
        selected = subset_masks.to(candidate_vectors.dtype) * candidate_mask.unsqueeze(
            1
        ).to(candidate_vectors.dtype)
        counts = selected.sum(dim=-1, keepdim=True)  # [N,S,1]
        valid = counts.squeeze(-1) > 0
        summed = torch.bmm(selected, candidate_vectors)  # [N,S,D]
        mean = summed / counts.clamp_min(1.0)
        state = state_vector.unsqueeze(1).expand(-1, subset_masks.size(1), -1)
        features = torch.cat([mean, state], dim=-1)
        scores = self.mlp(features).squeeze(-1)
        return scores.masked_fill(~valid, float("-inf"))
