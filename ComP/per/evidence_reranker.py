from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceReranker(nn.Module):
    """Small listwise scorer over K evidence candidates plus skip."""

    def __init__(
        self,
        candidate_dim: int,
        prompt_dim: int,
        top_k: int,
        hidden_dim: int = 128,
        rank_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        if candidate_dim <= 0 or prompt_dim <= 0 or top_k <= 0:
            raise ValueError("dimensions and top_k must be positive")
        self.top_k = int(top_k)
        self.evidence_projection = nn.Linear(candidate_dim, hidden_dim)
        self.prompt_projection = nn.Linear(prompt_dim, hidden_dim)
        self.rank_embedding = nn.Embedding(top_k, rank_dim)
        self.skip_embedding = nn.Parameter(torch.zeros(rank_dim))
        meta_dim = 1 + 1 + 3 + 3 + rank_dim
        token_dim = hidden_dim * 2 + meta_dim
        self.scorer = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _one_hot(values: torch.LongTensor, classes: int) -> torch.Tensor:
        return F.one_hot(values.long(), num_classes=classes).to(torch.float32)

    def forward(
        self,
        candidate_vectors: torch.Tensor,
        prompt_vectors: torch.Tensor,
        source_confidence: torch.Tensor,
        target_modalities: torch.LongTensor,
        missing_patterns: torch.Tensor,
        candidate_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Return action logits [N,K+1]. Inputs contain no labels or utilities."""
        if candidate_vectors.ndim != 3:
            raise ValueError("candidate_vectors must be [N,K,D]")
        n, k, _ = candidate_vectors.shape
        if k != self.top_k:
            raise ValueError(f"expected K={self.top_k}, got {k}")
        if prompt_vectors.shape != (n, self.prompt_projection.in_features):
            raise ValueError("prompt_vectors has invalid shape")
        expected = (n, k)
        if source_confidence.shape != expected or candidate_mask.shape != expected:
            raise ValueError("source_confidence and candidate_mask must be [N,K]")
        if target_modalities.shape != (n,) or missing_patterns.shape != (n, 3):
            raise ValueError("target_modalities must be [N] and missing_patterns [N,3]")
        if candidate_mask.dtype != torch.bool:
            raise TypeError("candidate_mask must be bool")

        evidence = self.evidence_projection(candidate_vectors)
        prompt = self.prompt_projection(prompt_vectors).unsqueeze(1).expand(-1, k, -1)
        cosine = F.cosine_similarity(evidence, prompt, dim=-1)
        confidence = source_confidence.to(candidate_vectors.dtype).unsqueeze(-1)
        target = self._one_hot(target_modalities, 3).unsqueeze(1).expand(-1, k, -1)
        pattern = missing_patterns.to(candidate_vectors.dtype).unsqueeze(1).expand(-1, k, -1)
        ranks = self.rank_embedding.weight.unsqueeze(0).expand(n, -1, -1)
        tokens = torch.cat(
            [evidence, prompt, cosine.unsqueeze(-1), confidence, target, pattern, ranks],
            dim=-1,
        )
        rank_logits = self.scorer(tokens).squeeze(-1)
        rank_logits = rank_logits.masked_fill(
            ~candidate_mask, torch.finfo(rank_logits.dtype).min
        )

        zero_evidence = torch.zeros_like(evidence[:, 0])
        skip_cos = torch.zeros(n, 1, dtype=tokens.dtype, device=tokens.device)
        skip_conf = torch.zeros(n, 1, dtype=tokens.dtype, device=tokens.device)
        target_skip = self._one_hot(target_modalities, 3)
        pattern_skip = missing_patterns.to(tokens.dtype)
        skip_rank = self.skip_embedding.unsqueeze(0).expand(n, -1)
        skip_tokens = torch.cat(
            [
                zero_evidence,
                self.prompt_projection(prompt_vectors),
                skip_cos,
                skip_conf,
                target_skip,
                pattern_skip,
                skip_rank,
            ],
            dim=-1,
        )
        skip_logit = self.scorer(skip_tokens.unsqueeze(1)).squeeze(-1)
        return torch.cat([skip_logit, rank_logits], dim=-1)
