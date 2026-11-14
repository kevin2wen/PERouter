from __future__ import annotations

import torch
import torch.nn as nn


class MSCStateEncoder(nn.Module):
    """Encode the missing-state context of an evidence decision.

    Inputs are discrete query-state facts, not target-modality content:
    observed mask, target one-hot, missing-pattern id, and missing rate.
    The observed × target interaction is kept explicit so a card's utility
    can depend on *which* modality is missing and *what* is observed.
    """

    def __init__(
        self,
        modality_count: int = 3,
        hidden_dim: int = 128,
        pattern_vocab_size: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if modality_count != 3:
            raise ValueError("PERouter uses exactly three modalities")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.modality_count = int(modality_count)
        self.hidden_dim = int(hidden_dim)
        self.pattern_embedding = nn.Embedding(
            int(pattern_vocab_size), int(hidden_dim)
        )
        interaction_dim = self.modality_count * self.modality_count
        input_dim = 2 * self.modality_count + interaction_dim + 1
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        availability: torch.Tensor,      # [N,3] observed mask (float or bool)
        target_modality: torch.Tensor,   # [N,3] one-hot
        missing_pattern: torch.Tensor,   # [N] long id
        missing_rate: torch.Tensor,      # [N] float in [0,1]
    ) -> torch.Tensor:
        if availability.shape[-1] != 3:
            raise ValueError("availability must be [...,3]")
        if target_modality.shape != availability.shape:
            raise ValueError("target_modality must match availability")
        observed = availability.to(availability.dtype)
        if observed.dtype == torch.bool:
            observed = observed.to(torch.float32)
        target = target_modality.to(observed.dtype)
        interaction = (
            observed.unsqueeze(-1) * target.unsqueeze(-2)
        ).flatten(start_dim=1)
        rate = missing_rate.to(observed.dtype).reshape(-1, 1)
        features = torch.cat([observed, target, interaction, rate], dim=-1)
        body = self.mlp(features)
        pattern = self.pattern_embedding(
            missing_pattern.long().clamp(
                min=0, max=self.pattern_embedding.num_embeddings - 1
            )
        )
        return body + pattern
