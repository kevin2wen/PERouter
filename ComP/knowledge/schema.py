from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

import torch

from ComP.per.types import MODALITIES


AFFECT_PRIORS = ("positive", "negative", "neutral", "mixed")
CARD_SCOPES = ("utterance", "dialogue", "scene")


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _as_three_tuple(values, name: str) -> Tuple[bool, bool, bool]:
    if not isinstance(values, (tuple, list)) or len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    if any(not isinstance(value, bool) for value in values):
        raise TypeError(f"{name} must contain bool values")
    return tuple(values)


def _validate_optional_tensor(value: Optional[torch.Tensor], name: str) -> None:
    if value is None:
        return
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or None")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains non-finite values")


def _normalise_modality(value: str | int) -> str:
    if isinstance(value, int):
        if value not in range(3):
            raise ValueError("target modality index must be 0, 1, or 2")
        return MODALITIES[value]
    if value not in MODALITIES:
        raise ValueError(f"unknown target modality: {value}")
    return value


@dataclass(frozen=True)
class EvidenceQuery:
    """One target-specific query for one utterance.

    This class is intentionally namespaced under ``ComP.knowledge``. The
    existing batch contract ``ComP.per.types.EvidenceQuery`` is unchanged.
    """

    sample_id: str
    dialogue_id: str
    utterance_id: str
    availability: Tuple[bool, bool, bool]
    missing_pattern: Tuple[bool, bool, bool]
    target_modality: str
    text_query: Optional[str]
    audio_query: Optional[torch.Tensor]
    video_query: Optional[torch.Tensor]
    dialogue_context: Optional[torch.Tensor]

    def __post_init__(self) -> None:
        _require_text(self.sample_id, "sample_id")
        _require_text(self.dialogue_id, "dialogue_id")
        _require_text(self.utterance_id, "utterance_id")
        availability = _as_three_tuple(self.availability, "availability")
        missing_pattern = _as_three_tuple(
            self.missing_pattern, "missing_pattern"
        )
        if missing_pattern != tuple(not value for value in availability):
            raise ValueError("missing_pattern must equal the complement of availability")
        if not any(availability):
            raise ValueError("an all-missing query has no observed source modality")
        target = _normalise_modality(self.target_modality)
        target_index = MODALITIES.index(target)
        if availability[target_index]:
            raise ValueError("target modality must be missing in an evidence query")
        if target != "text" and self.text_query is not None:
            if not isinstance(self.text_query, str) or not self.text_query.strip():
                raise ValueError("text_query must be a non-empty string or None")
        elif target == "text" and self.text_query is not None:
            raise ValueError("text_query must be None when text is the target modality")
        if target == "audio" and self.audio_query is not None:
            raise ValueError("audio_query must be None when audio is the target modality")
        if target == "video" and self.video_query is not None:
            raise ValueError("video_query must be None when video is the target modality")
        _validate_optional_tensor(self.audio_query, "audio_query")
        _validate_optional_tensor(self.video_query, "video_query")
        _validate_optional_tensor(self.dialogue_context, "dialogue_context")


@dataclass(frozen=True)
class RetrievedContext:
    query_id: str
    sample_id: str
    dialogue_id: str
    utterance_id: str
    target_modality: str
    rank: int
    text: Optional[str]
    audio_feature: Optional[torch.Tensor]
    video_feature: Optional[torch.Tensor]
    retrieval_score: float
    source_modalities: Tuple[str, ...]
    source_id: str
    source_confidence: float = 0.0

    def __post_init__(self) -> None:
        for value, name in (
            (self.query_id, "query_id"),
            (self.sample_id, "sample_id"),
            (self.dialogue_id, "dialogue_id"),
            (self.utterance_id, "utterance_id"),
            (self.source_id, "source_id"),
        ):
            _require_text(value, name)
        _normalise_modality(self.target_modality)
        if not isinstance(self.rank, int) or self.rank < 0:
            raise ValueError("rank must be a non-negative integer")
        if self.text is not None and not isinstance(self.text, str):
            raise TypeError("text must be a string or None")
        _validate_optional_tensor(self.audio_feature, "audio_feature")
        _validate_optional_tensor(self.video_feature, "video_feature")
        if not isinstance(self.retrieval_score, float):
            raise TypeError("retrieval_score must be a float")
        if not torch.isfinite(torch.tensor(self.retrieval_score)):
            raise FloatingPointError("retrieval_score must be finite")
        if not isinstance(self.source_confidence, float):
            raise TypeError("source_confidence must be a float")
        if not 0.0 <= self.source_confidence <= 1.0:
            raise ValueError("source_confidence must be in [0,1]")
        if not self.source_modalities:
            raise ValueError("source_modalities must not be empty")
        unknown = set(self.source_modalities) - set(MODALITIES)
        if unknown:
            raise ValueError(f"unknown source modalities: {sorted(unknown)}")


@dataclass(frozen=True)
class KnowledgeCard:
    knowledge_id: str
    trigger_type: str
    trigger_modalities: Tuple[str, ...]
    trigger_text: Optional[str]
    trigger_visual_ref: Optional[str]
    trigger_audio_ref: Optional[str]
    scope: Literal["utterance", "dialogue", "scene"]
    relation_type: str
    affect_prior: Literal["positive", "negative", "neutral", "mixed"]
    applicability: Tuple[str, ...]
    evidence_text: str
    supporting_context_ids: Tuple[str, ...]
    conflict_context_ids: Tuple[str, ...]
    source_id: str
    source_type: str
    source_confidence: float
    license: str
    version: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.knowledge_id, "knowledge_id"),
            (self.trigger_type, "trigger_type"),
            (self.relation_type, "relation_type"),
            (self.evidence_text, "evidence_text"),
            (self.source_id, "source_id"),
            (self.source_type, "source_type"),
            (self.license, "license"),
            (self.version, "version"),
        ):
            _require_text(value, name)
        if not self.trigger_modalities:
            raise ValueError("trigger_modalities must not be empty")
        if set(self.trigger_modalities) - set(MODALITIES):
            raise ValueError("trigger_modalities contains an unknown modality")
        if not self.applicability:
            raise ValueError("applicability must not be empty")
        if set(self.applicability) - set(MODALITIES):
            raise ValueError("applicability contains an unknown modality")
        if self.scope not in CARD_SCOPES:
            raise ValueError(f"unsupported card scope: {self.scope}")
        if self.affect_prior not in AFFECT_PRIORS:
            raise ValueError(f"unsupported affect prior: {self.affect_prior}")
        trigger_parts = (
            self.trigger_text,
            self.trigger_visual_ref,
            self.trigger_audio_ref,
        )
        if not any(isinstance(value, str) and value.strip() for value in trigger_parts):
            raise ValueError("a card must have at least one non-empty trigger source")
        if not torch.isfinite(torch.tensor(float(self.source_confidence))):
            raise FloatingPointError("source_confidence must be finite")
        if not 0.0 <= float(self.source_confidence) <= 1.0:
            raise ValueError("source_confidence must be in [0,1]")


@dataclass(frozen=True)
class EvidenceCandidate:
    query_id: str
    target_modality: str
    rank: int
    card: KnowledgeCard
    applicability_score: float
    retrieval_score: float
    source_confidence: float
    conflict_score: float
    encoded_feature: torch.Tensor

    def __post_init__(self) -> None:
        _require_text(self.query_id, "query_id")
        _normalise_modality(self.target_modality)
        if not isinstance(self.rank, int) or self.rank < 0:
            raise ValueError("rank must be a non-negative integer")
        for value, name in (
            (self.applicability_score, "applicability_score"),
            (self.retrieval_score, "retrieval_score"),
            (self.source_confidence, "source_confidence"),
            (self.conflict_score, "conflict_score"),
        ):
            if not torch.isfinite(torch.tensor(float(value))):
                raise FloatingPointError(f"{name} must be finite")
        if not 0.0 <= float(self.source_confidence) <= 1.0:
            raise ValueError("source_confidence must be in [0,1]")
        if not 0.0 <= float(self.conflict_score) <= 1.0:
            raise ValueError("conflict_score must be in [0,1]")
        if not isinstance(self.encoded_feature, torch.Tensor):
            raise TypeError("encoded_feature must be a torch.Tensor")
        if self.encoded_feature.ndim == 0:
            raise ValueError("encoded_feature must have at least one dimension")
        if not torch.isfinite(self.encoded_feature).all():
            raise FloatingPointError("encoded_feature contains non-finite values")


@dataclass(frozen=True)
class EvidencePolicyOutput:
    action_logits: torch.Tensor
    use_probability: torch.Tensor
    candidate_scores: torch.Tensor
    selected_rank: torch.Tensor
    skip: torch.Tensor


@dataclass(frozen=True)
class StructuredComponent:
    component_id: str
    relation_type: str
    affect_prior: Literal["positive", "negative", "neutral", "mixed"]
    trigger_type: str
    trigger_modalities: Tuple[str, ...]
    applicability: Tuple[str, ...]
    evidence_text: str
    supporting_context_ids: Tuple[str, ...]
    conflict_context_ids: Tuple[str, ...]
    source_id: str
    source_type: str
    source_confidence: float = 0.0
    scope: Literal["utterance", "dialogue", "scene"] = "utterance"
    trigger_text: Optional[str] = None
    trigger_visual_ref: Optional[str] = None
    trigger_audio_ref: Optional[str] = None
    license: str = "unknown"
    version: str = "unknown"
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.component_id, "component_id")
        _require_text(self.source_id, "source_id")
        if self.affect_prior not in AFFECT_PRIORS:
            raise ValueError("unsupported affect prior")
        if not torch.isfinite(torch.tensor(float(self.source_confidence))):
            raise FloatingPointError("source_confidence must be finite")
        if not 0.0 <= float(self.source_confidence) <= 1.0:
            raise ValueError("source_confidence must be in [0,1]")


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: Tuple[str, ...] = ()


def validate_card(card: KnowledgeCard) -> ValidationResult:
    """Return a structured validation result without catching programmer errors."""

    errors = []
    try:
        KnowledgeCard(**card.__dict__)
    except (TypeError, ValueError, FloatingPointError) as exc:
        errors.append(str(exc))
    return ValidationResult(valid=not errors, errors=tuple(errors))
