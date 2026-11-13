from __future__ import annotations

from typing import Tuple

import torch

from .types import EvidenceBatch

ORACLE_MODES = frozenset({"oracle_weak", "oracle_strong"})


def validate_evidence_oracle_split(mode: str, split: str) -> None:
    """Reject label-dependent evidence selection outside validation."""
    if mode not in ORACLE_MODES:
        return
    if split not in {"validation", "test"}:
        raise ValueError(f"unsupported split: {split}")
    if mode == "oracle_strong" and split != "validation":
        raise PermissionError(
            "oracle_strong is diagnostic-only and must never run on test"
        )


def isolate_rank_evidence(reference: EvidenceBatch, rank: int) -> EvidenceBatch:
    """Keep exactly one retrieved rank available at each evidence position."""
    reference.validate()
    k = reference.mask.size(-1)
    if rank < 0 or rank >= k:
        raise ValueError(f"rank must be in [0,{k}), got {rank}")
    mask = torch.zeros_like(reference.mask)
    mask[:, :, rank] = reference.mask[:, :, rank]
    confidence = reference.source_confidence * mask.to(reference.source_confidence.dtype)
    conflict = None
    if reference.conflict_score is not None:
        conflict = reference.conflict_score * mask.to(reference.conflict_score.dtype)
    ids = []
    for batch_row in reference.ids:
        sample_ids = []
        for rank_ids in batch_row:
            selected = rank_ids[rank] if mask.numel() else ""
            sample_ids.append([selected if index == rank else "" for index in range(k)])
        ids.append(sample_ids)
    return EvidenceBatch(
        ids=ids,
        features=reference.features.clone(),
        mask=mask,
        source_confidence=confidence,
        conflict_score=conflict,
    )


def select_confidence_ranks(reference: EvidenceBatch) -> torch.LongTensor:
    """Select the highest-confidence retrieved rank without reading labels."""
    reference.validate()
    scores = reference.source_confidence.masked_fill(
        ~reference.mask, torch.finfo(reference.source_confidence.dtype).min
    )
    ranks = scores.argmax(dim=-1).long()
    valid_rows = reference.mask.any(dim=-1)
    return torch.where(valid_rows, ranks, torch.zeros_like(ranks))


def select_strong_ranks(
    utilities: torch.Tensor,
    availability: torch.BoolTensor,
    sequence_mask: torch.BoolTensor,
    evidence_mask: torch.BoolTensor,
) -> Tuple[torch.LongTensor, torch.BoolTensor]:
    """Choose one shared evidence rank per utterance using rank utilities.

    utilities must be [B,T,3,K]. Utilities are averaged across active missing
    target modalities. Current feature evidence is shared across targets, so a
    single rank is selected per utterance rather than pretending it is
    target-modality-specific.
    """
    if utilities.ndim != 4:
        raise ValueError("utilities must be [B,T,3,K]")
    batch_size, seq_len, modalities, ranks = utilities.shape
    if modalities != 3:
        raise ValueError("utilities must have three target modalities")
    if availability.shape != (batch_size, seq_len, 3):
        raise ValueError("availability must match utilities [B,T,3]")
    if sequence_mask.shape != (batch_size, seq_len):
        raise ValueError("sequence_mask must be [B,T]")
    if evidence_mask.shape != (batch_size, seq_len, ranks):
        raise ValueError("evidence_mask must be [B,T,K]")
    if not torch.isfinite(utilities).all():
        raise FloatingPointError("oracle utilities contain non-finite values")

    active = (
        (~availability)
        & sequence_mask.unsqueeze(-1)
        & evidence_mask.any(dim=-1, keepdim=True)
    )
    score_sum = torch.where(active.unsqueeze(-1), utilities, torch.zeros_like(utilities)).sum(dim=2)
    target_count = active.sum(dim=2)
    valid_rows = target_count > 0
    mean_utility = score_sum / target_count.clamp_min(1).unsqueeze(-1).to(score_sum.dtype)
    filled = mean_utility.masked_fill(
        ~valid_rows.unsqueeze(-1), torch.finfo(mean_utility.dtype).min
    )
    ranks_out = filled.argmax(dim=-1).long()
    ranks_out = torch.where(valid_rows, ranks_out, torch.zeros_like(ranks_out))
    return ranks_out, valid_rows


def apply_selected_ranks(
    reference: EvidenceBatch,
    ranks: torch.LongTensor,
) -> EvidenceBatch:
    """Repeat a selected rank across K so coverage matches feature/random."""
    reference.validate()
    batch_size, seq_len, top_k = reference.mask.shape
    if ranks.shape != (batch_size, seq_len):
        raise ValueError("ranks must be [B,T]")
    ranks = ranks.to(device=reference.features.device, dtype=torch.long)
    valid_rows = reference.mask.any(dim=-1)
    safe_ranks = torch.where(valid_rows, ranks, torch.zeros_like(ranks))
    feature_index = safe_ranks[:, :, None, None].expand(
        batch_size, seq_len, 1, reference.features.size(-1)
    )
    selected_features = reference.features.gather(2, feature_index)
    selected_features = selected_features.expand(-1, -1, top_k, -1).clone()

    confidence_index = safe_ranks[:, :, None]
    selected_confidence = reference.source_confidence.gather(2, confidence_index)
    selected_confidence = selected_confidence.expand(-1, -1, top_k).clone()
    selected_mask = valid_rows.unsqueeze(-1).expand(-1, -1, top_k).clone()
    selected_confidence = selected_confidence * selected_mask.to(selected_confidence.dtype)

    selected_conflict = None
    if reference.conflict_score is not None:
        selected_conflict = reference.conflict_score.gather(2, confidence_index)
        selected_conflict = selected_conflict.expand(-1, -1, top_k).clone()
        selected_conflict = selected_conflict * selected_mask.to(selected_conflict.dtype)

    ids = []
    for batch_index in range(batch_size):
        sample_ids = []
        for time_index in range(seq_len):
            selected_id = (
                reference.ids[batch_index][time_index][int(safe_ranks[batch_index, time_index])]
                if bool(valid_rows[batch_index, time_index])
                else ""
            )
            sample_ids.append([selected_id for _ in range(top_k)])
        ids.append(sample_ids)

    return EvidenceBatch(
        ids=ids,
        features=selected_features,
        mask=selected_mask,
        source_confidence=selected_confidence,
        conflict_score=selected_conflict,
    )
