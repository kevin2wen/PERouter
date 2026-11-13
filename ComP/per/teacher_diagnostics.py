from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from .types import MODALITIES


def validate_route_selection(split: str, route_override: str) -> None:
    if split != "validation":
        raise ValueError("teacher diagnostics only support validation split")
    if route_override not in {"oracle", "student", "fixed", "prompt"}:
        raise ValueError(f"unsupported route override: {route_override}")
    if route_override == "oracle" and split != "validation":
        raise PermissionError("oracle route is forbidden outside validation")


def valid_route_mask(
    availability: torch.BoolTensor,
    sequence_mask: torch.BoolTensor,
    evidence_available: torch.BoolTensor,
) -> torch.BoolTensor:
    expected = (*sequence_mask.shape, 3)
    if availability.shape != expected:
        raise ValueError(f"availability must be {expected}")
    if evidence_available.shape != expected:
        raise ValueError(f"evidence_available must be {expected}")
    return (~availability) & sequence_mask.unsqueeze(-1) & evidence_available


def _scalar_stats(values: torch.Tensor) -> Dict[str, float]:
    if values.numel() == 0:
        return {
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
            "p05": 0.0, "p50": 0.0, "p95": 0.0,
        }
    value = values.float().cpu()
    return {
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "min": float(value.min()),
        "max": float(value.max()),
        "p05": float(torch.quantile(value, 0.05)),
        "p50": float(torch.quantile(value, 0.50)),
        "p95": float(torch.quantile(value, 0.95)),
    }


def pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().cpu().reshape(-1)
    right = right.float().cpu().reshape(-1)
    if left.numel() != right.numel():
        raise ValueError("correlation inputs must have equal size")
    if left.numel() < 2 or left.std(unbiased=False) < 1e-12 or right.std(unbiased=False) < 1e-12:
        return 0.0
    value = torch.corrcoef(torch.stack([left, right]))[0, 1]
    return float(value) if torch.isfinite(value) else 0.0


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    value = values.float().cpu()
    order = sorted(range(value.numel()), key=lambda index: (float(value[index]), index))
    ranks = torch.empty(value.numel(), dtype=torch.float64)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and float(value[order[end]]) == float(value[order[position]]):
            end += 1
        average = (position + 1 + end) / 2.0
        for index in order[position:end]:
            ranks[index] = average
        position = end
    return ranks


def spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() != right.numel():
        raise ValueError("correlation inputs must have equal size")
    return pearson(_average_ranks(left), _average_ranks(right))


def compute_route_quality(
    route: torch.Tensor,
    target_route: torch.Tensor,
    utility_gap: torch.Tensor,
    availability: torch.BoolTensor,
    sequence_mask: torch.BoolTensor,
    evidence_available: torch.BoolTensor,
) -> Dict[str, float]:
    valid = valid_route_mask(availability, sequence_mask, evidence_available)
    if route.shape != target_route.shape or route.shape != utility_gap.shape:
        raise ValueError("route, target and utility gap must have identical shape")
    if valid.shape != route.shape:
        raise ValueError("valid mask shape mismatch")
    values = route[valid].float()
    targets = target_route[valid].float()
    gaps = utility_gap[valid].float()
    result: Dict[str, float] = {
        "count": float(values.numel()),
        "positive_target_but_gap_nonpositive": 0.0,
        "negative_target_but_gap_positive": 0.0,
        "route_bce": 0.0,
        "route_target_pearson": 0.0,
        "route_target_spearman": 0.0,
    }
    if values.numel() == 0:
        return result
    if not torch.isfinite(values).all() or not torch.isfinite(targets).all() or not torch.isfinite(gaps).all():
        raise FloatingPointError("non-finite route quality inputs")
    result["route_bce"] = float(
        F.binary_cross_entropy(values.clamp(1e-7, 1.0 - 1e-7), targets, reduction="mean")
    )
    result["route_target_pearson"] = pearson(values, targets)
    result["route_target_spearman"] = spearman(values, targets)
    result["positive_target_but_gap_nonpositive"] = float(
        ((targets > 0.5) & (gaps <= 0)).float().mean()
    )
    result["negative_target_but_gap_positive"] = float(
        ((targets < 0.5) & (gaps > 0)).float().mean()
    )
    result.update({f"target_{key}": value for key, value in _scalar_stats(targets).items()})
    result.update({f"gap_{key}": value for key, value in _scalar_stats(gaps).items()})
    return result


def route_quality_groups(
    route: torch.Tensor,
    target_route: torch.Tensor,
    utility_gap: torch.Tensor,
    availability: torch.BoolTensor,
    sequence_mask: torch.BoolTensor,
    evidence_available: torch.BoolTensor,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for modality_index, modality in enumerate(MODALITIES):
        selected = torch.zeros_like(availability)
        selected[:, :, modality_index] = True
        stats = compute_route_quality(
            route, target_route, utility_gap,
            torch.where(selected, availability, torch.ones_like(availability)),
            sequence_mask,
            evidence_available,
        )
        raw = valid_route_mask(availability, sequence_mask, evidence_available) & selected
        if raw.any():
            rows.append({"group_type": "modality", "group": modality, **stats})

    missing = ~availability
    valid = valid_route_mask(availability, sequence_mask, evidence_available)
    for bits in torch.cartesian_prod(
        torch.tensor([0, 1], device=availability.device),
        torch.tensor([0, 1], device=availability.device),
        torch.tensor([0, 1], device=availability.device),
    ):
        pattern = bits.bool().view(1, 1, 3)
        if not pattern.any():
            continue
        selected = (missing == pattern) & valid
        if not selected.any():
            continue
        local_availability = torch.where(
            selected,
            torch.zeros_like(availability),
            torch.ones_like(availability),
        )
        stats = compute_route_quality(
            route, target_route, utility_gap,
            local_availability,
            sequence_mask,
            torch.ones_like(evidence_available, device=availability.device),
        )
        key = "".join(str(int(value)) for value in bits.tolist())
        rows.append({"group_type": "missing_pattern", "group": key, **stats})
    return rows
