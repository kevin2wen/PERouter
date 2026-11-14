from __future__ import annotations

import torch

from ComP.per.types import MODALITIES, EvidenceBatch
from ComP.knowledge.pipeline import KnowledgeBatchResult


def apply_subset(
    result: KnowledgeBatchResult,
    target: str,
    batch_index: int,
    time_index: int,
    subset_mask: torch.Tensor,
) -> KnowledgeBatchResult:
    """Return a result where only one query's chosen subset is active.

    All other positions and targets are zeroed so a full-batch forward
    measures the isolated effect of this subset for this query.
    """
    if target not in MODALITIES:
        raise ValueError(f"unknown target modality: {target}")
    evidence_by_target = {}
    for name in MODALITIES:
        evidence = result.evidence_by_target[name]
        mask = torch.zeros_like(evidence.mask)
        confidence = torch.zeros_like(evidence.source_confidence)
        conflict = (
            torch.zeros_like(evidence.conflict_score)
            if evidence.conflict_score is not None
            else None
        )
        if name == target:
            keep = subset_mask.to(device=mask.device, dtype=torch.bool)
            mask[batch_index, time_index] = keep
            confidence[batch_index, time_index] = (
                evidence.source_confidence[batch_index, time_index]
                * keep.to(evidence.source_confidence.dtype)
            )
            if conflict is not None:
                conflict[batch_index, time_index] = (
                    evidence.conflict_score[batch_index, time_index]
                    * keep.to(evidence.conflict_score.dtype)
                )
        evidence_by_target[name] = EvidenceBatch(
            ids=evidence.ids,
            features=evidence.features.clone(),
            mask=mask,
            source_confidence=confidence,
            conflict_score=conflict,
        )
    return KnowledgeBatchResult(
        result.queries,
        result.contexts,
        result.cards,
        result.candidates,
        evidence_by_target,
    )
