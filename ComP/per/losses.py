from __future__ import annotations

from typing import Dict, Literal, Optional

import torch
import torch.nn.functional as F

from ..loss import MaskedCELoss, MaskedMSELoss
from .types import PEROutput


def build_route_target(
    prompt_utility: torch.Tensor,
    evidence_utility: torch.Tensor,
    missing_mask: torch.BoolTensor,
    temperature: torch.Tensor,
    teacher_confidence: torch.Tensor,
) -> torch.Tensor:
    """Return detached [B,T,3] prompt/evidence targets.

    Target=1 means evidence is preferred. Values outside valid missing
    utterances are set to 0.5 and are always excluded by the route loss.
    """
    if prompt_utility.shape != evidence_utility.shape:
        raise ValueError("prompt/evidence utilities must have the same shape")
    if prompt_utility.shape != missing_mask.shape:
        raise ValueError("missing_mask must match utility [B,T,3]")
    if teacher_confidence.shape != missing_mask.shape:
        raise ValueError("teacher_confidence must be [B,T,3]")
    temperature_tensor = torch.as_tensor(
        temperature, dtype=prompt_utility.dtype, device=prompt_utility.device
    )
    if (~torch.isfinite(temperature_tensor)).any() or (temperature_tensor <= 0).any():
        raise ValueError("route temperature must be finite and positive")
    confidence = teacher_confidence.clamp(min=0.0, max=1.0)
    probability = torch.sigmoid(
        (evidence_utility - prompt_utility) / temperature_tensor
    )
    probability = 0.5 + confidence * (probability - 0.5)
    target = torch.where(missing_mask, probability, torch.full_like(probability, 0.5))
    return target.detach()


def route_distillation_loss(
    route_logits: torch.Tensor,
    target_route: torch.Tensor,
    missing_mask: torch.BoolTensor,
    sequence_mask: Optional[torch.BoolTensor] = None,
) -> torch.Tensor:
    """BCE over valid missing utterance-modality positions only."""
    if route_logits.shape != target_route.shape or route_logits.shape != missing_mask.shape:
        raise ValueError("route_logits, target_route, missing_mask must be [B,T,3]")
    valid = missing_mask & torch.isfinite(target_route)
    if sequence_mask is not None:
        if sequence_mask.shape != route_logits.shape[:2]:
            raise ValueError("sequence_mask must be [B,T]")
        valid = valid & sequence_mask.unsqueeze(-1)
    denominator = valid.sum()
    if int(denominator.item()) == 0:
        # Keep the graph while returning exactly zero when no missing position
        # exists in this batch.
        return route_logits.sum() * 0.0
    raw = F.binary_cross_entropy_with_logits(
        route_logits[valid], target_route[valid].detach(), reduction="sum"
    )
    return raw / denominator.to(raw.dtype)


def masked_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    sequence_mask: torch.BoolTensor,
    availability: torch.BoolTensor,
) -> torch.Tensor:
    """MSE over valid, observed modality values only."""
    if reconstruction.shape != target.shape:
        raise ValueError("reconstruction and target shapes differ")
    if sequence_mask.shape != reconstruction.shape[:2]:
        raise ValueError("sequence_mask must be [B,T]")
    if availability.shape != (*sequence_mask.shape, 3):
        raise ValueError("availability must be [B,T,3]")
    # reconstruction is [B,T,3*D]; repeat each modality mask over its feature dim.
    if reconstruction.shape[-1] % 3 != 0:
        raise ValueError("reconstruction last dimension must be divisible by 3")
    dim = reconstruction.shape[-1] // 3
    element_mask = (
        availability & sequence_mask.unsqueeze(-1)
    ).to(reconstruction.dtype).repeat_interleave(dim, dim=-1)
    squared = (reconstruction - target).pow(2) * element_mask
    denominator = element_mask.sum().clamp_min(1.0)
    return squared.sum() / denominator


def response_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    task: Literal["regression", "classification"],
    sequence_mask: torch.BoolTensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    if sequence_mask.shape != student_logits.shape[:2]:
        raise ValueError("sequence_mask must match student logits [B,T]")
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher/student logits must have identical shape")
    if not sequence_mask.any():
        return student_logits.sum() * 0.0
    if task == "classification":
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        student = F.log_softmax(student_logits[sequence_mask] / temperature, dim=-1)
        teacher = F.softmax(
            teacher_logits[sequence_mask].detach() / temperature, dim=-1
        )
        return (
            F.kl_div(student, teacher, reduction="batchmean") * temperature**2
        )
    difference = (
        student_logits[sequence_mask] - teacher_logits[sequence_mask].detach()
    )
    return F.mse_loss(difference, torch.zeros_like(difference))


def masked_task_loss(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    sequence_mask: torch.BoolTensor,
    task: Literal["regression", "classification"],
) -> torch.Tensor:
    if task == "regression":
        return MaskedMSELoss()(prediction, labels, sequence_mask)
    if task == "classification":
        return MaskedCELoss()(prediction, labels, sequence_mask)
    raise ValueError(f"unknown task: {task}")


def per_loss(
    output: PEROutput,
    labels: torch.Tensor,
    task: Literal["regression", "classification"],
    route_target: Optional[torch.Tensor],
    missing_mask: torch.BoolTensor,
    teacher_logits: Optional[torch.Tensor] = None,
    lambda_route: float = 0.5,
    lambda_kd: float = 0.1,
    kd_temperature: float = 2.0,
    sequence_mask: Optional[torch.BoolTensor] = None,
) -> Dict[str, torch.Tensor]:
    sequence_mask = (
        torch.ones(missing_mask.shape[:2], dtype=torch.bool, device=missing_mask.device)
        if sequence_mask is None
        else sequence_mask
    )
    task_loss = masked_task_loss(output.logits, labels, sequence_mask, task)
    zero = output.logits.sum() * 0.0
    route_loss = zero
    if route_target is not None:
        route_loss = route_distillation_loss(
            output.route_logits, route_target, missing_mask, sequence_mask
        )
    kd_loss = zero
    if teacher_logits is not None:
        kd_loss = response_distillation_loss(
            output.logits,
            teacher_logits,
            task,
            sequence_mask,
            temperature=kd_temperature,
        )
    total = task_loss + lambda_route * route_loss + lambda_kd * kd_loss
    return {
        "total": total,
        "task": task_loss,
        "route": route_loss,
        "kd": kd_loss,
    }
