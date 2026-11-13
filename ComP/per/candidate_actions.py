from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch

MODALITY_NAMES = ("audio", "text", "video")


@dataclass(frozen=True)
class CandidateAction:
    use_evidence: bool
    rank: int
    target_modality: int

    def __post_init__(self) -> None:
        if self.target_modality not in (0, 1, 2):
            raise ValueError("target_modality must be 0, 1, or 2")
        if self.use_evidence and self.rank < 0:
            raise ValueError("evidence action requires rank >= 0")
        if not self.use_evidence and self.rank != -1:
            raise ValueError("skip action requires rank=-1")

    @property
    def action_id(self) -> str:
        target = MODALITY_NAMES[self.target_modality]
        return f"{target}:rank{self.rank}" if self.use_evidence else f"{target}:skip"


def action_candidates(target_modality: int, top_k: int) -> list[CandidateAction]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    return [CandidateAction(False, -1, target_modality)] + [
        CandidateAction(True, rank, target_modality)
        for rank in range(top_k)
    ]


def select_best_actions(
    candidate_utility: torch.Tensor,
    skip_utility: torch.Tensor,
    active: torch.BoolTensor,
    min_helpful_gain: float,
) -> Dict[str, torch.Tensor]:
    """Select skip or best evidence rank from train-derived utilities.

    candidate_utility: [B,T,3,K]
    skip_utility:       [B,T,3]
    active:             [B,T,3]
    """
    if candidate_utility.ndim != 4 or candidate_utility.size(2) != 3:
        raise ValueError("candidate_utility must be [B,T,3,K]")
    if skip_utility.shape != candidate_utility.shape[:3]:
        raise ValueError("skip_utility must be [B,T,3]")
    if active.shape != skip_utility.shape:
        raise ValueError("active must be [B,T,3]")
    if active.dtype != torch.bool:
        raise TypeError("active must be bool")
    if not torch.isfinite(candidate_utility).all() or not torch.isfinite(skip_utility).all():
        raise FloatingPointError("candidate utilities contain non-finite values")

    delta = candidate_utility - skip_utility.unsqueeze(-1)
    masked = delta.masked_fill(~active.unsqueeze(-1), torch.finfo(delta.dtype).min)
    best_delta, best_rank = masked.max(dim=-1)
    valid = active.clone()
    best_delta = torch.where(valid, best_delta, torch.zeros_like(best_delta))
    best_rank = torch.where(valid, best_rank.long(), torch.zeros_like(best_rank, dtype=torch.long))
    use_evidence = valid & (best_delta >= float(min_helpful_gain))
    selected_rank = torch.where(use_evidence, best_rank, torch.full_like(best_rank, -1))
    return {
        "delta_utility": best_delta,
        "best_rank": best_rank,
        "use_evidence": use_evidence,
        "selected_rank": selected_rank,
        "has_helpful_candidate": use_evidence,
    }


def make_candidate_cache_key(
    *,
    dataset_version: str,
    train_split_hash: str,
    student_checkpoint_hash: str,
    adapter_checkpoint_hash: Optional[str],
    evidence_source: str,
    evidence_source_hash: str,
    evidence_encoder_hash: str,
    mask_manifest_hash: str,
    top_k: int,
    gamma: float,
    evidence_mode: str,
    sampler_version: str,
    min_helpful_gain: float = 0.0,
    exclusion_policy: str = "utterance-only",
) -> str:
    payload = {
        "dataset_version": str(dataset_version),
        "train_split_hash": str(train_split_hash),
        "student_checkpoint_hash": str(student_checkpoint_hash),
        "adapter_checkpoint_hash": str(adapter_checkpoint_hash or ""),
        "evidence_source": str(evidence_source),
        "evidence_source_hash": str(evidence_source_hash),
        "evidence_encoder_hash": str(evidence_encoder_hash),
        "mask_manifest_hash": str(mask_manifest_hash),
        "top_k": int(top_k),
        "gamma": float(gamma),
        "evidence_mode": str(evidence_mode),
        "sampler_version": str(sampler_version),
        "min_helpful_gain": float(min_helpful_gain),
        "exclusion_policy": str(exclusion_policy),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_train_only_split(split: str) -> None:
    if split != "train":
        raise PermissionError("candidate utility cache generation is train-only")
