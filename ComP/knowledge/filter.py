from __future__ import annotations

import torch
import torch.nn as nn

from ComP.per.evidence_reranker import EvidenceReranker

from .schema import EvidencePolicyOutput


class KnowledgeEvidenceFilter(nn.Module):
    """Thin policy wrapper over the existing label-free listwise reranker."""

    def __init__(
        self,
        candidate_dim: int,
        prompt_dim: int,
        top_k: int,
        hidden_dim: int = 128,
        rank_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.top_k = int(top_k)
        self.reranker = EvidenceReranker(
            candidate_dim=candidate_dim,
            prompt_dim=prompt_dim,
            top_k=top_k,
            hidden_dim=hidden_dim,
            rank_dim=rank_dim,
            dropout=dropout,
        )

    def forward(
        self,
        candidate_vectors: torch.Tensor,
        prompt_vectors: torch.Tensor,
        source_confidence: torch.Tensor,
        target_modalities: torch.LongTensor,
        missing_patterns: torch.Tensor,
        candidate_mask: torch.BoolTensor,
    ) -> EvidencePolicyOutput:
        logits = self.reranker(
            candidate_vectors,
            prompt_vectors,
            source_confidence,
            target_modalities,
            missing_patterns,
            candidate_mask,
        )
        probabilities = torch.softmax(logits, dim=-1)
        rank_scores = probabilities[:, 1:]
        masked_scores = rank_scores.masked_fill(
            ~candidate_mask, torch.finfo(rank_scores.dtype).min
        )
        best_rank = masked_scores.argmax(dim=-1)
        best_score = masked_scores.gather(1, best_rank[:, None]).squeeze(1)
        skip = probabilities[:, 0] >= best_score
        selected_rank = torch.where(
            skip,
            torch.full_like(best_rank, -1),
            best_rank,
        )
        return EvidencePolicyOutput(
            action_logits=logits,
            use_probability=1.0 - probabilities[:, 0],
            candidate_scores=rank_scores,
            selected_rank=selected_rank,
            skip=skip,
        )
