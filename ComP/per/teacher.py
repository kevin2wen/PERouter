from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model import ComP
from .types import MODALITIES, TeacherCandidateScores, TeacherTargets


class CompleteViewTeacher(nn.Module):
    """Frozen ComP teacher used only during training.

    Candidate scores are produced under no_grad and detached before they leave
    this module. The teacher's parameters are frozen at construction time.
    """

    def __init__(
        self,
        model: ComP,
        task: str,
        dim: int,
        teacher_ckpt_hash: str,
    ):
        super().__init__()
        if task not in {"regression", "classification"}:
            raise ValueError(f"unknown task: {task}")
        self.model = model
        self.task = task
        self.dim = int(dim)
        self.teacher_ckpt_hash = str(teacher_ckpt_hash)
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        # The teacher must remain in evaluation mode even if a parent model is
        # switched to train() by an enclosing training loop.
        return super().train(False)

    def _input_features(
        self,
        features: Dict[str, torch.Tensor],
        availability: torch.BoolTensor,
    ) -> torch.Tensor:
        batch_size, seq_len = availability.shape[:2]
        columns = []
        for modality_index, modality in enumerate(MODALITIES):
            value = features[modality]
            if value.shape[:2] != (batch_size, seq_len):
                raise ValueError(f"{modality} must share [B,T]")
            keep = availability[:, :, modality_index].unsqueeze(-1)
            observed = torch.where(keep, value, torch.zeros_like(value))
            columns.append(observed.permute(1, 0, 2))
        return torch.cat(columns, dim=-1)

    def _element_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.task == "regression":
            return (logits.squeeze(-1) - labels).pow(2)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1).long(),
            reduction="none",
        ).reshape(labels.shape)

    def _run_candidate(
        self,
        features: Dict[str, torch.Tensor],
        availability: torch.BoolTensor,
        labels: torch.Tensor,
        candidate: Dict[str, torch.Tensor],
        target_modality_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputfeats = self._input_features(features, availability)
        batch_size, seq_len = availability.shape[:2]
        input_mask = torch.ones(
            seq_len, batch_size, 3, dtype=torch.bool, device=availability.device
        )
        input_mask &= availability.permute(1, 0, 2)
        target_name = MODALITIES[target_modality_index]

        def fuse(prompts, _mask):
            result = dict(prompts)
            replace = (~availability[:, :, target_modality_index]).unsqueeze(-1)
            result[target_name] = torch.where(
                replace, candidate[target_name], prompts[target_name]
            )
            return result

        outputs = self.model(
            inputfeats,
            labels,
            input_mask,
            availability.any(dim=-1).to(torch.float32),
            False,
            return_intermediates=True,
            coordinator_mode="on",
            fusion_fn=fuse,
        )
        logits = outputs[1]
        return logits, self._element_loss(logits, labels)

    def score_candidates(
        self,
        features: Dict[str, torch.Tensor],
        availability: torch.BoolTensor,
        prompt_candidates: Dict[str, torch.Tensor],
        evidence: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> TeacherCandidateScores:
        if availability.shape != (*labels.shape, 3):
            raise ValueError("availability must be [B,T,3] and labels [B,T]")
        for name, value in {**prompt_candidates, **evidence}.items():
            expected = (labels.size(0), labels.size(1), self.dim)
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} candidate must be {expected}")
        with torch.no_grad():
            prompt_logits = []
            evidence_logits = []
            prompt_utility = []
            evidence_utility = []
            for modality_index, _modality in enumerate(MODALITIES):
                p_logits, p_loss = self._run_candidate(
                    features,
                    availability,
                    labels,
                    prompt_candidates,
                    modality_index,
                )
                e_logits, e_loss = self._run_candidate(
                    features,
                    availability,
                    labels,
                    evidence,
                    modality_index,
                )
                active = (~availability[:, :, modality_index]).to(p_loss.dtype)
                prompt_logits.append(p_logits.detach())
                evidence_logits.append(e_logits.detach())
                prompt_utility.append((-p_loss * active).detach())
                evidence_utility.append((-e_loss * active).detach())

            p_logits_t = torch.stack(prompt_logits, dim=2).detach()
            e_logits_t = torch.stack(evidence_logits, dim=2).detach()
            p_utility_t = torch.stack(prompt_utility, dim=2).detach()
            e_utility_t = torch.stack(evidence_utility, dim=2).detach()
            utility = torch.stack([p_utility_t, e_utility_t], dim=-1).detach()
            agreement = torch.exp(-(p_utility_t - e_utility_t).abs()).clamp(
                min=0.0, max=1.0
            )
            active_modalities = (~availability).to(agreement.dtype)
            confidence = (agreement * active_modalities).detach()
        return TeacherCandidateScores(
            utility=utility,
            prompt_logits=p_logits_t,
            evidence_logits=e_logits_t,
            confidence=confidence,
        )

    def full_view_logits(
        self,
        features: Dict[str, torch.Tensor],
        sequence_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        batch_size, seq_len = sequence_mask.shape
        availability = sequence_mask.unsqueeze(-1).expand(-1, -1, 3)
        inputfeats = self._input_features(features, availability)
        input_mask = torch.ones(
            seq_len, batch_size, 3, dtype=torch.bool, device=sequence_mask.device
        )
        labels = torch.zeros(batch_size, seq_len, device=sequence_mask.device)
        with torch.no_grad():
            outputs = self.model(
                inputfeats,
                labels,
                input_mask,
                sequence_mask.to(torch.float32),
                False,
                return_intermediates=True,
                coordinator_mode="on",
            )
            return outputs[1].detach()


def make_teacher_cache_key(
    *,
    dataset_version: str,
    split: str,
    sample_id: str,
    utterance_id: str,
    availability: Sequence[bool],
    student_candidate_ckpt_hash: str,
    teacher_ckpt_hash: str,
    kb_version: str,
    top_k: int,
    seed: int,
) -> str:
    availability_tuple = tuple(bool(value) for value in availability)
    if len(availability_tuple) != 3 or not any(availability_tuple):
        raise ValueError("availability must be a non-empty three-element mask")
    fields = {
        "dataset_version": str(dataset_version),
        "split": str(split),
        "sample_id": str(sample_id),
        "utterance_id": str(utterance_id),
        "availability": list(availability_tuple),
        "student_candidate_ckpt_hash": str(student_candidate_ckpt_hash),
        "teacher_ckpt_hash": str(teacher_ckpt_hash),
        "kb_version": str(kb_version),
        "top_k": int(top_k),
        "seed": int(seed),
    }
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class TeacherTargetCache:
    """Offline cache whose key includes every required invalidation field."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if len(key) != 64 or any(ch not in "0123456789abcdef" for ch in key):
            raise ValueError("teacher cache key must be a sha256 hex string")
        return self.root / f"{key}.pt"

    def get(self, key: str) -> Optional[TeacherTargets]:
        path = self._path(key)
        if not path.exists():
            return None
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or "teacher_utility" not in payload:
            raise ValueError(f"invalid teacher cache payload: {path}")
        return TeacherTargets(
            teacher_utility=payload["teacher_utility"],
            route_target=payload["route_target"],
            teacher_logits=payload.get("teacher_logits"),
            teacher_confidence=payload.get("teacher_confidence"),
        )

    def put(self, key: str, targets: TeacherTargets) -> None:
        payload = {
            "teacher_utility": targets.teacher_utility.detach().cpu(),
            "route_target": targets.route_target.detach().cpu(),
            "teacher_logits": (
                targets.teacher_logits.detach().cpu()
                if targets.teacher_logits is not None
                else None
            ),
            "teacher_confidence": (
                targets.teacher_confidence.detach().cpu()
                if targets.teacher_confidence is not None
                else None
            ),
        }
        path = self._path(key)
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
