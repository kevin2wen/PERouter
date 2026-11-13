from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EvidenceSensitivityResult:
    sample_id: str
    utterance_id: str
    target_modality: str
    evidence_mode: str
    logits_delta_l1: float
    logits_delta_l2: float
    prediction_delta: float
    sign_flip: bool
    classification_margin_delta: float
    evidence_gradient_norm: float


def _regression_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or logits.size(-1) != 1:
        raise ValueError("regression logits must be [B,T,1]")
    return logits.squeeze(-1)


def _classification_margin(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or logits.size(-1) < 2:
        raise ValueError("classification logits must be [B,T,C], C>=2")
    top2 = logits.topk(2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def paired_logit_deltas(
    reference_logits: torch.Tensor,
    comparison_logits: torch.Tensor,
    labels: torch.Tensor,
    sequence_mask: torch.BoolTensor,
    task: str,
) -> Dict[str, torch.Tensor]:
    """Paired sensitivity metrics under identical inputs and masks."""
    if reference_logits.shape != comparison_logits.shape:
        raise ValueError("reference and comparison logits must match")
    if labels.shape != sequence_mask.shape or reference_logits.shape[:2] != labels.shape:
        raise ValueError("labels/sequence mask must match logits [B,T]")
    if sequence_mask.dtype != torch.bool:
        raise TypeError("sequence_mask must be bool")
    if not torch.isfinite(reference_logits).all() or not torch.isfinite(comparison_logits).all():
        raise FloatingPointError("logits contain non-finite values")

    delta = comparison_logits - reference_logits
    l1 = delta.abs().mean(dim=-1)
    l2 = delta.pow(2).mean(dim=-1).sqrt()
    if task == "regression":
        ref_pred = _regression_logits(reference_logits)
        cmp_pred = _regression_logits(comparison_logits)
        prediction_delta = (cmp_pred - ref_pred).abs()
        sign_flip = (ref_pred > 0) != (cmp_pred > 0)
        margin_delta = (cmp_pred.abs() - ref_pred.abs()).abs()
    elif task == "classification":
        ref_pred = reference_logits.argmax(dim=-1)
        cmp_pred = comparison_logits.argmax(dim=-1)
        prediction_delta = (cmp_pred != ref_pred).to(reference_logits.dtype)
        sign_flip = cmp_pred != ref_pred
        margin_delta = (
            _classification_margin(comparison_logits)
            - _classification_margin(reference_logits)
        ).abs()
    else:
        raise ValueError(f"unknown task: {task}")
    return {
        "valid": sequence_mask,
        "logits_delta_l1": l1,
        "logits_delta_l2": l2,
        "prediction_delta": prediction_delta,
        "sign_flip": sign_flip,
        "classification_margin_delta": margin_delta,
    }


def normalized_gradient(tensor: torch.Tensor | None) -> torch.Tensor:
    if tensor is None:
        return torch.zeros(1, device="cpu")
    return tensor.detach().float().pow(2).sum().sqrt()


def masked_mean(values: torch.Tensor, mask: torch.BoolTensor) -> float:
    if values.shape != mask.shape:
        raise ValueError("values and mask must have the same shape")
    if not mask.any():
        return 0.0
    return float(values[mask].float().mean().item())


def masked_rate(values: torch.BoolTensor, mask: torch.BoolTensor) -> float:
    if values.shape != mask.shape:
        raise ValueError("values and mask must have the same shape")
    if not mask.any():
        return 0.0
    return float(values[mask].float().mean().item())
