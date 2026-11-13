from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence

import torch

from ComP.per.types import MODALITIES

from .schema import EvidenceCandidate, KnowledgeCard, RetrievedContext


def _target(value: str | int) -> str:
    if isinstance(value, int):
        if value not in range(3):
            raise ValueError("target modality index must be 0, 1, or 2")
        return MODALITIES[value]
    if value not in MODALITIES:
        raise ValueError(f"unknown target modality: {value}")
    return value


class KnowledgeCandidateBuilder:
    """Attach cards to retrieved provenance and produce ranked candidates."""

    def build(
        self,
        query_id: str,
        target_modality: str | int,
        contexts: Sequence[RetrievedContext],
        cards: Iterable[KnowledgeCard],
        encoded_features: Mapping[str, torch.Tensor],
        *,
        top_k: int,
    ) -> List[EvidenceCandidate]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("query_id must be a non-empty string")
        target = _target(target_modality)
        context_by_id = {context.source_id: context for context in contexts}
        candidates: List[EvidenceCandidate] = []
        for card in cards:
            if not isinstance(card, KnowledgeCard):
                raise TypeError("cards must contain KnowledgeCard objects")
            if target not in card.applicability:
                continue
            if card.knowledge_id not in encoded_features:
                raise KeyError(f"missing encoded feature for card {card.knowledge_id}")
            supporting = [
                context_by_id[source_id]
                for source_id in card.supporting_context_ids
                if source_id in context_by_id
            ]
            if not supporting:
                continue
            feature = encoded_features[card.knowledge_id]
            if not isinstance(feature, torch.Tensor):
                raise TypeError("encoded feature must be a torch.Tensor")
            if not torch.isfinite(feature).all():
                raise FloatingPointError("encoded feature contains non-finite values")
            retrieval_score = max(context.retrieval_score for context in supporting)
            source_confidence = min(
                card.source_confidence,
                max(context.source_confidence for context in supporting),
            )
            candidates.append(
                EvidenceCandidate(
                    query_id=query_id,
                    target_modality=target,
                    rank=0,
                    card=card,
                    applicability_score=float(target in card.applicability),
                    retrieval_score=float(retrieval_score),
                    source_confidence=float(source_confidence),
                    conflict_score=0.0,
                    encoded_feature=feature.detach().clone(),
                )
            )
        ranked = sorted(
            candidates,
            key=lambda item: (
                -item.source_confidence,
                -item.retrieval_score,
                item.card.knowledge_id,
            ),
        )
        return [
            EvidenceCandidate(
                query_id=candidate.query_id,
                target_modality=candidate.target_modality,
                rank=rank,
                card=candidate.card,
                applicability_score=candidate.applicability_score,
                retrieval_score=candidate.retrieval_score,
                source_confidence=candidate.source_confidence,
                conflict_score=candidate.conflict_score,
                encoded_feature=candidate.encoded_feature,
            )
            for rank, candidate in enumerate(ranked[:top_k])
        ]
