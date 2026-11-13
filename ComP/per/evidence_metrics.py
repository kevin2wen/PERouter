from __future__ import annotations

from typing import Callable, Dict

import torch


def grouped_metric_masks(
    availability: torch.BoolTensor,
    sequence_mask: torch.BoolTensor,
) -> Dict[str, torch.BoolTensor]:
    """Build mutually exclusive missing-pattern masks and target-modality masks."""
    if availability.shape[:2] != sequence_mask.shape or availability.size(-1) != 3:
        raise ValueError("availability must be [B,T,3] and match sequence_mask")
    if sequence_mask.dtype != torch.bool or availability.dtype != torch.bool:
        raise TypeError("availability and sequence_mask must be bool")
    patterns = {}
    pattern_ids = (
        availability[:, :, 0].long() * 4
        + availability[:, :, 1].long() * 2
        + availability[:, :, 2].long()
    )
    for pattern_id in torch.unique(pattern_ids[sequence_mask]).tolist():
        patterns[f"pattern_{int(pattern_id):03b}"] = (
            sequence_mask & (pattern_ids == int(pattern_id))
        )
    targets = {}
    names = ("audio", "text", "video")
    for modality_index, name in enumerate(names):
        targets[f"{name}_missing"] = (
            sequence_mask & (~availability[:, :, modality_index])
        )
    return {**patterns, **targets}


def grouped_metrics(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    sequence_mask: torch.BoolTensor,
    availability: torch.BoolTensor,
    dataset: str,
    metric_fn: Callable,
) -> Dict[str, Dict[str, float]]:
    if labels.shape != predictions.shape or labels.shape != sequence_mask.shape:
        raise ValueError("labels, predictions, and sequence_mask must match [B,T]")
    if availability.shape[:2] != labels.shape:
        raise ValueError("availability must be [B,T,3]")
    output = {}
    for name, mask in grouped_metric_masks(availability, sequence_mask).items():
        if mask.any():
            output[name] = metric_fn(labels, predictions, mask, dataset)
    return output
