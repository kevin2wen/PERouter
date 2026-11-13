from __future__ import annotations

import hashlib
from typing import Dict, Iterable, Mapping, Sequence

import torch

from ComP.per.types import MODALITIES

AFFECTS = ("positive", "negative", "neutral", "mixed")
HASH_BUCKETS = 8
FEATURE_DIM = (
    4  # affect prior
    + 3  # applicability
    + 3  # target modality
    + 3  # missing pattern
    + HASH_BUCKETS  # trigger type
    + HASH_BUCKETS  # relation type
    + HASH_BUCKETS  # source type
    + 1  # source confidence
    + 1  # conflict score
    + 1  # rank / top_k
    + 1  # evidence text length / 32
    + 1  # trigger count / 3
    + 1  # supporting context count / top_k
)


def _bucket(value: object) -> int:
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % HASH_BUCKETS


def _one_hot_bucket(value: object) -> list[float]:
    vector = [0.0] * HASH_BUCKETS
    vector[_bucket(value)] = 1.0
    return vector


def card_feature_vector(
    card,
    *,
    target: str,
    missing_pattern: Sequence[bool],
    rank: int,
    top_k: int,
    conflict_score: float = 0.0,
) -> list[float]:
    affect_name = getattr(card, "affect_prior", "neutral")
    affect_vector = [
        1.0 if affect_name == name else 0.0 for name in AFFECTS
    ]
    applicability = [
        1.0 if name in set(card.applicability) else 0.0
        for name in MODALITIES
    ]
    target_vector = [
        1.0 if name == target else 0.0 for name in MODALITIES
    ]
    pattern_vector = [
        1.0 if bool(value) else 0.0 for value in missing_pattern
    ]
    trigger_types = list(getattr(card, "trigger_modalities", ()) or ())
    vector = (
        affect_vector
        + applicability
        + target_vector
        + pattern_vector
        + _one_hot_bucket(getattr(card, "trigger_type", "none"))
        + _one_hot_bucket(getattr(card, "relation_type", "none"))
        + _one_hot_bucket(getattr(card, "source_type", "none"))
        + [float(getattr(card, "source_confidence", 0.0) or 0.0)]
        + [float(conflict_score)]
        + [float(rank) / max(1, top_k)]
        + [min(1.0, len(getattr(card, "evidence_text", "") or "") / 32.0)]
        + [min(1.0, len(trigger_types) / 3.0)]
        + [
            min(
                1.0,
                len(getattr(card, "supporting_context_ids", ()) or ())
                / max(1, top_k),
            )
        ]
    )
    return vector


def card_index(cards: Iterable[object]) -> Dict[str, object]:
    return {
        str(getattr(card, "knowledge_id", "")): card for card in cards
    }


def structured_candidate_vectors(
    cards: Mapping[str, object],
    ids: Sequence[Sequence[Sequence[str]]],
    *,
    target: str,
    missing_pattern: Sequence[bool],
    top_k: int,
    confidence: torch.Tensor,
    conflict: torch.Tensor | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build [B,T,K,FEATURE_DIM] structured features from card metadata."""
    batch_size = len(ids)
    seq_len = len(ids[0]) if batch_size else 0
    features = torch.zeros(
        batch_size, seq_len, top_k, FEATURE_DIM, dtype=dtype, device=device
    )
    pattern = [bool(value) for value in missing_pattern]
    for batch_index, sample in enumerate(ids):
        for time_index, rank_ids in enumerate(sample):
            for rank, knowledge_id in enumerate(rank_ids[:top_k]):
                card = cards.get(str(knowledge_id))
                if card is None:
                    continue
                vector = card_feature_vector(
                    card,
                    target=target,
                    missing_pattern=pattern,
                    rank=rank,
                    top_k=top_k,
                    conflict_score=(
                        float(conflict[batch_index, time_index, rank])
                        if conflict is not None
                        else 0.0
                    ),
                )
                features[batch_index, time_index, rank] = torch.tensor(
                    vector, dtype=dtype, device=device
                )
    return features
