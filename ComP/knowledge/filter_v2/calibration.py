from __future__ import annotations

import numpy as np


def fit_threshold_f1(
    max_utility,
    use_labels,
    *,
    min_tau: float | None = None,
    max_tau: float | None = None,
    steps: int = 200,
) -> dict:
    """Choose a global threshold maximizing F1 of the use decision."""
    scores = np.asarray(max_utility, dtype=np.float64)
    labels = np.asarray(use_labels, dtype=bool)
    if scores.size == 0:
        raise ValueError("empty threshold calibration set")
    lo = float(scores.min()) if min_tau is None else min_tau
    hi = float(scores.max()) if max_tau is None else max_tau
    grid = np.unique(np.linspace(lo, hi, steps))
    best = {"tau": 0.0, "f1": -1.0, "use_rate": 0.0}
    for tau in grid:
        predicted = scores > tau
        tp = float((predicted & labels).sum())
        fp = float((predicted & ~labels).sum())
        fn = float((~predicted & labels).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best["f1"]:
            best = {
                "tau": float(tau),
                "f1": float(f1),
                "use_rate": float(predicted.mean()),
                "precision": precision,
                "recall": recall,
            }
    return best
