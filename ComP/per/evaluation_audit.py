from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score

from .types import MODALITIES


def _metrics(labels, predictions, mask, dataset: str) -> Dict[str, float]:
    labels = labels.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy().astype(bool)
    if dataset in {"CMUMOSI", "CMUMOSEI"}:
        valid = mask & (labels != 0)
        labels = labels[valid]
        predictions = predictions[valid]
        if labels.size == 0:
            return {"acc": 0.0, "f1": 0.0, "mae": 0.0, "corr": 0.0, "count": 0}
        binary_labels = labels > 0
        binary_predictions = predictions > 0
        corr = (
            float(np.corrcoef(labels, predictions)[0, 1])
            if labels.size > 1
            else 0.0
        )
        return {
            "acc": float(accuracy_score(binary_labels, binary_predictions)),
            "f1": float(f1_score(binary_labels, binary_predictions, average="weighted")),
            "mae": float(np.mean(np.abs(labels - predictions))),
            "corr": corr,
            "count": int(labels.size),
        }
    valid = mask
    labels = labels[valid].astype(np.int64)
    predictions = predictions[valid].astype(np.int64)
    if labels.size == 0:
        return {"acc": 0.0, "f1": 0.0, "ua": 0.0, "count": 0}
    return {
        "acc": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, average="weighted")),
        "ua": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "count": int(labels.size),
    }


class EvaluationAuditAccumulator:
    def __init__(self) -> None:
        self.labels: List[torch.Tensor] = []
        self.predictions: List[torch.Tensor] = []
        self.availability: List[torch.Tensor] = []
        self.sequence: List[torch.Tensor] = []

    def add(self, labels, predictions, availability, sequence_mask) -> None:
        self.labels.append(labels.detach().cpu())
        self.predictions.append(predictions.detach().cpu())
        self.availability.append(availability.detach().cpu().bool())
        self.sequence.append(sequence_mask.detach().cpu().bool())

    def summary(self, dataset: str) -> Dict[str, float]:
        if not self.labels:
            return {}
        labels = torch.cat(self.labels, dim=0)
        predictions = torch.cat(self.predictions, dim=0)
        availability = torch.cat(self.availability, dim=0)
        sequence = torch.cat(self.sequence, dim=0)
        missing = ~availability
        group_masks = {
            "full": availability.any(dim=-1) & availability.all(dim=-1),
            "text_missing": missing[:, :, 1],
            "audio_missing": missing[:, :, 0],
            "video_missing": missing[:, :, 2],
            "text_audio_missing": missing[:, :, 1] & missing[:, :, 0],
            "text_video_missing": missing[:, :, 1] & missing[:, :, 2],
            "audio_video_missing": missing[:, :, 0] & missing[:, :, 2],
        }
        for modality_index, modality in enumerate(MODALITIES):
            group_masks[f"target_{modality}"] = missing[:, :, modality_index]
        result: Dict[str, float] = {}
        for name, group in group_masks.items():
            metrics = _metrics(labels, predictions, group & sequence, dataset)
            for key, value in metrics.items():
                result[f"pattern_{name}_{key}"] = float(value)
        return result
