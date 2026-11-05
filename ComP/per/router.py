from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import MODALITIES


def _flatten_modality_summary(summary: torch.Tensor) -> torch.Tensor:
    if summary.ndim == 4:
        if summary.shape[2] != 3:
            raise ValueError("per-modality summary must be [B,T,3,D]")
        return summary.flatten(start_dim=2)
    if summary.ndim == 3:
        return summary
    raise ValueError("summary must be [B,T,3,D] or [B,T,3*D]")


class PERRouter(nn.Module):
    """Predict one evidence-source logit per utterance and modality.

    The router consumes only deployed information: observed projected
    representations, prompt candidates, evidence candidates, availability,
    and evidence confidence. It has no label, teacher, or complete-view input.
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        hidden = int(hidden_dim or dim)
        # observed + prompt + evidence + availability + confidence + target one-hot
        input_dim = 9 * self.dim + 3 + 3 + 9
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )
        nn.init.constant_(self.net[-1].bias, -2.0)

    def forward(
        self,
        observed_summary: torch.Tensor,
        generated_summary: torch.Tensor,
        evidence_summary: torch.Tensor,
        availability: torch.BoolTensor,
        evidence_confidence: torch.Tensor,
    ) -> torch.Tensor:
        observed = _flatten_modality_summary(observed_summary)
        generated = _flatten_modality_summary(generated_summary)
        evidence = _flatten_modality_summary(evidence_summary)
        if observed.shape[:2] != availability.shape[:2]:
            raise ValueError("router summaries and availability must share [B,T]")
        if evidence_confidence.shape != availability.shape:
            raise ValueError("evidence_confidence must be [B,T,3]")

        eye = torch.eye(3, dtype=observed.dtype, device=observed.device)
        target_onehot = eye.view(1, 1, 3, 3).expand(
            observed.shape[0], observed.shape[1], 3, 3
        ).flatten(start_dim=2)
        router_input = torch.cat(
            [
                observed,
                generated,
                evidence,
                availability.to(observed.dtype),
                evidence_confidence.to(observed.dtype),
                target_onehot,
            ],
            dim=-1,
        )
        return self.net(router_input)


def fuse_candidates(
    prompt_candidates: Dict[str, torch.Tensor],
    evidence_candidates: Dict[str, torch.Tensor],
    route: torch.Tensor,
    availability: torch.BoolTensor,
    evidence_available: torch.BoolTensor,
) -> Dict[str, torch.Tensor]:
    """Convexly fuse evidence and prompt only at missing positions."""
    if route.shape != availability.shape or route.shape[-1] != 3:
        raise ValueError("route and availability must both be [B,T,3]")
    missing_mask = ~availability
    if evidence_available.shape != availability.shape:
        raise ValueError("evidence_available must be [B,T,3]")
    active_route = route * missing_mask.to(route.dtype) * evidence_available.to(
        route.dtype
    )
    fused = {}
    for modality_index, modality in enumerate(MODALITIES):
        if modality not in prompt_candidates or modality not in evidence_candidates:
            raise KeyError(f"missing candidate for {modality}")
        prompt = prompt_candidates[modality]
        evidence = evidence_candidates[modality]
        if prompt.shape != evidence.shape:
            raise ValueError(
                f"{modality} candidates differ: {tuple(prompt.shape)} vs "
                f"{tuple(evidence.shape)}"
            )
        weight = active_route[:, :, modality_index].unsqueeze(-1)
        # At observed positions availability wins; at missing positions a
        # zero-evidence fallback keeps the prompt path exact.
        fusion = weight * evidence + (1.0 - weight) * prompt
        keep_observed = availability[:, :, modality_index].unsqueeze(-1)
        fused[modality] = torch.where(keep_observed, prompt, fusion)
    return fused


def gather_modality_summary(candidates: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([candidates[name] for name in MODALITIES], dim=2)


def evidence_target_confidence(
    evidence_mask: torch.BoolTensor,
    source_confidence: torch.Tensor,
) -> torch.Tensor:
    """Summarize top-k evidence confidence into [B,T,3]."""
    if evidence_mask.ndim != 3:
        raise ValueError("evidence mask must be [B,T,K]")
    if source_confidence.shape != evidence_mask.shape:
        raise ValueError("source_confidence must be [B,T,K]")
    confidence = source_confidence * evidence_mask.to(source_confidence.dtype)
    count = evidence_mask.sum(dim=-1).clamp_min(1).to(source_confidence.dtype)
    per_target = confidence.sum(dim=-1) / count
    return per_target.unsqueeze(-1).expand(-1, -1, 3)


def masked_route_probability(
    route_logits: torch.Tensor,
    availability: torch.BoolTensor,
    evidence_available: torch.BoolTensor,
) -> torch.Tensor:
    if route_logits.shape != availability.shape:
        raise ValueError("route_logits and availability must be [B,T,3]")
    active = (~availability) & evidence_available
    return torch.sigmoid(route_logits) * active.to(route_logits.dtype)
