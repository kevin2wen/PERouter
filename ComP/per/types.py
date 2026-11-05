from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch


# This order matches ComP's input columns and every [B, T, 3] mask.
MODALITIES: Tuple[str, str, str] = ("audio", "text", "video")


@dataclass
class MissingSpec:
    availability: torch.BoolTensor  # [B,T,3], True=observed
    sequence_mask: torch.BoolTensor  # [B,T], True=valid utterance
    missing_code: Optional[torch.LongTensor] = None  # legacy compatibility only

    def validate(self) -> None:
        if self.availability.ndim != 3 or self.availability.shape[-1] != 3:
            raise ValueError(
                f"availability must be [B,T,3], got {tuple(self.availability.shape)}"
            )
        if self.sequence_mask.shape != self.availability.shape[:2]:
            raise ValueError(
                "sequence_mask must be [B,T], got "
                f"{tuple(self.sequence_mask.shape)} for availability "
                f"{tuple(self.availability.shape)}"
            )
        if self.availability.dtype != torch.bool:
            raise TypeError("availability must be torch.bool")
        if self.sequence_mask.dtype != torch.bool:
            raise TypeError("sequence_mask must be torch.bool")
        if self.availability.device != self.sequence_mask.device:
            raise ValueError("availability and sequence_mask must share a device")


@dataclass
class RawSampleMeta:
    sample_id: str
    utterance_ids: List[str]
    raw_texts: Optional[List[str]] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class EvidenceQuery:
    sample_ids: List[str]
    utterance_ids: List[List[str]]
    features: torch.Tensor  # [B,T,D_query]
    target_modalities: torch.BoolTensor  # [B,T,3], True=query is needed


@dataclass
class EvidenceBatch:
    ids: List[List[List[str]]]  # [B][T][K]
    features: torch.Tensor  # [B,T,K,D_ev]
    mask: torch.BoolTensor  # [B,T,K], True=valid evidence
    source_confidence: torch.Tensor  # [B,T,K], float32 in [0,1]
    conflict_score: Optional[torch.Tensor] = None  # [B,T,K], float32 in [0,1]

    def validate(self, batch_size: Optional[int] = None, seq_len: Optional[int] = None) -> None:
        if self.features.ndim != 4:
            raise ValueError("evidence features must be [B,T,K,D_ev]")
        b, t, k, _ = self.features.shape
        if batch_size is not None and b != batch_size:
            raise ValueError(f"evidence batch={b}, expected {batch_size}")
        if seq_len is not None and t != seq_len:
            raise ValueError(f"evidence time={t}, expected {seq_len}")
        if self.mask.shape != (b, t, k):
            raise ValueError(
                f"evidence mask must be {(b, t, k)}, got {tuple(self.mask.shape)}"
            )
        if self.source_confidence.shape != (b, t, k):
            raise ValueError(
                "source_confidence must be [B,T,K], got "
                f"{tuple(self.source_confidence.shape)}"
            )
        if self.mask.dtype != torch.bool:
            raise TypeError("evidence mask must be torch.bool")
        if len(self.ids) != b:
            raise ValueError("evidence ids must have shape [B][T][K]")
        for sample_ids in self.ids:
            if len(sample_ids) != t or any(len(row) != k for row in sample_ids):
                raise ValueError("evidence ids do not match [B][T][K]")
        if self.features.device != self.mask.device:
            raise ValueError("evidence features and mask must share a device")
        if self.source_confidence.device != self.mask.device:
            raise ValueError("source_confidence must share the evidence device")
        if self.conflict_score is not None and self.conflict_score.shape != (b, t, k):
            raise ValueError("conflict_score must be [B,T,K]")


@dataclass
class PERBatch:
    features: Dict[str, torch.Tensor]
    availability: torch.BoolTensor  # [B,T,3]
    sequence_mask: torch.BoolTensor  # [B,T]
    labels: torch.Tensor  # [B,T]
    sample_ids: List[str]
    utterance_ids: List[List[str]]
    evidence: Optional[EvidenceBatch]
    raw_texts: Optional[List[List[str]]] = None
    metadata: Optional[List[RawSampleMeta]] = None


@dataclass
class PEROutput:
    logits: torch.Tensor  # [B,T,1] or [B,T,C]
    route_logits: torch.Tensor  # [B,T,3]
    route: torch.Tensor  # [B,T,3]
    prompt_candidates: Dict[str, torch.Tensor]  # [B,T,D_e]
    evidence: Dict[str, torch.Tensor]  # [B,T,D_e]
    fused: Dict[str, torch.Tensor]  # [B,T,D_e]
    coordinator_weights: Optional[torch.Tensor] = None  # [B,T,3]
    coordinator_logits: Optional[torch.Tensor] = None  # [B,T,3]
    coordinator_mode: str = "on"
    modality_logits: Optional[Dict[str, torch.Tensor]] = None
    intermediates: Dict[str, object] = field(default_factory=dict)


@dataclass
class TeacherCandidateScores:
    # Order is fixed as [prompt_candidate, evidence].
    utility: torch.Tensor  # [B,T,3,2], detached
    prompt_logits: torch.Tensor  # [B,T,3,C_or_1], detached
    evidence_logits: torch.Tensor  # [B,T,3,C_or_1], detached
    confidence: torch.Tensor  # [B,T,3], detached, in [0,1]


@dataclass
class TeacherTargets:
    teacher_utility: torch.Tensor  # [B,T,3,2], detached
    route_target: torch.Tensor  # [B,T,3], detached
    teacher_logits: Optional[torch.Tensor] = None
    teacher_confidence: Optional[torch.Tensor] = None


def make_missing_spec(
    availability: torch.BoolTensor,
    sequence_mask: torch.BoolTensor,
) -> MissingSpec:
    """Bind ComP's per-utterance availability mask to a stable type contract."""
    spec = MissingSpec(availability=availability, sequence_mask=sequence_mask)
    spec.validate()
    return spec


def ensure_modality_dict(
    values: Dict[str, torch.Tensor],
    *,
    batch_size: int,
    seq_len: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> None:
    missing = set(MODALITIES) - set(values)
    if missing:
        raise KeyError(f"{name} missing modalities: {sorted(missing)}")
    for modality in MODALITIES:
        value = values[modality]
        expected = (batch_size, seq_len, dim)
        if tuple(value.shape) != expected:
            raise ValueError(
                f"{name}[{modality}] must be {expected}, got {tuple(value.shape)}"
            )
        if value.device != device:
            raise ValueError(f"{name}[{modality}] is on {value.device}, expected {device}")
        if value.dtype != dtype:
            raise TypeError(f"{name}[{modality}] dtype={value.dtype}, expected {dtype}")


def stack_modalities(values: Dict[str, torch.Tensor]) -> torch.Tensor:
    ensure_keys = set(values) >= set(MODALITIES)
    if not ensure_keys:
        raise KeyError("modality dict must contain audio, text, and video")
    return torch.stack([values[modality] for modality in MODALITIES], dim=2)
