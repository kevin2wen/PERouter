from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def expected_calibration_error(
    probabilities, labels, bins: int = 10
) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probabilities >= lower) & (
            probabilities < upper if index < bins - 1 else probabilities <= upper
        )
        if not mask.any():
            continue
        confidence = probabilities[mask].mean()
        accuracy = labels[mask].mean()
        error += mask.mean() * abs(confidence - accuracy)
    return float(error)


def brier_score(probabilities, labels) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return float(np.mean((probabilities - labels) ** 2))


def fit_temperature(logits, labels, grid=None) -> float:
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.float32)
    grid = grid or [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    best_temperature, best_loss = 1.0, float("inf")
    for temperature in grid:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits / temperature, labels
        ).item()
        if math.isfinite(loss) and loss < best_loss:
            best_loss, best_temperature = loss, float(temperature)
    return best_temperature


def fit_threshold(probabilities, labels, grid=None) -> Dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    grid = grid or [i / 100 for i in range(5, 96, 5)]
    best = {"tau_skip": 0.5, "balanced_accuracy": -1.0, "use_rate": 1.0}
    for threshold in grid:
        predicted = probabilities >= threshold
        positives = labels == 1
        negatives = labels == 0
        recall = (
            float((predicted & positives).sum() / positives.sum())
            if positives.any()
            else 0.0
        )
        specificity = (
            float((~predicted & negatives).sum() / negatives.sum())
            if negatives.any()
            else 0.0
        )
        balanced = 0.5 * (recall + specificity)
        if balanced > best["balanced_accuracy"]:
            best = {
                "tau_skip": float(threshold),
                "balanced_accuracy": float(balanced),
                "use_rate": float((~predicted).mean()),
                "skip_recall": recall,
                "skip_specificity": specificity,
            }
    return best


def fit_calibration(
    skip_logits,
    skip_labels,
    *,
    utility_scorer_hash: str,
    state_encoder_hash: str = "",
    candidate_schema_hash: str = "",
    inner_validation_hash: str = "",
) -> Dict:
    temperature = fit_temperature(skip_logits, skip_labels)
    logits = torch.as_tensor(skip_logits, dtype=torch.float32)
    probabilities = torch.sigmoid(logits / temperature).numpy()
    labels = np.asarray(skip_labels, dtype=np.float64)
    threshold = fit_threshold(probabilities, labels)
    return {
        "calibration_format_version": 1,
        "utility_scorer_hash": utility_scorer_hash,
        "state_encoder_hash": state_encoder_hash,
        "candidate_feature_schema_hash": candidate_schema_hash,
        "inner_validation_manifest_hash": inner_validation_hash,
        "temperature": float(temperature),
        **threshold,
        "brier": brier_score(probabilities, labels),
        "ece": expected_calibration_error(probabilities, labels),
        "n": int(labels.size),
        "positive_rate": float(labels.mean()),
    }


def save_calibration(payload: Dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_calibration(path: str | Path, *, expected_scorer_hash: str) -> Dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("utility_scorer_hash") != expected_scorer_hash:
        raise ValueError("calibration does not match utility scorer hash")
    return payload
