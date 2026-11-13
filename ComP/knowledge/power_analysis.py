from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

import numpy as np

Z_ALPHA = 1.959963984540054
Z_POWER = 0.8416212335729143


def _z(alpha: float, power: float):
    try:
        from scipy.stats import norm

        return float(norm.ppf(1 - alpha / 2)), float(norm.ppf(power))
    except Exception:
        if alpha != 0.05 or power != 0.80:
            raise
        return Z_ALPHA, Z_POWER


def cluster_mde(
    deltas: Sequence[float],
    *,
    alpha: float = 0.05,
    power: float = 0.80,
) -> Dict[str, float]:
    """Minimum detectable effect for a cluster-level mean difference.

    Uses the normal approximation with equal cluster weights; the estimate is
    approximate for ten clusters and should be read together with the
    percentile/wild intervals.
    """
    values = np.asarray(deltas, dtype=np.float64)
    if values.size < 2:
        raise ValueError("at least two clusters are required")
    z_alpha, z_power = _z(alpha, power)
    sd = float(values.std(ddof=1))
    n = int(values.size)
    mde = (z_alpha + z_power) * sd / math.sqrt(n)
    return {
        "clusters": float(n),
        "mean_delta": float(values.mean()),
        "sd": sd,
        "mde": float(mde),
        "alpha": float(alpha),
        "power": float(power),
    }


def required_clusters(sd: float, effect: float, *, alpha: float = 0.05, power: float = 0.80) -> float:
    if effect <= 0:
        raise ValueError("effect must be positive")
    z_alpha, z_power = _z(alpha, power)
    return float(math.ceil(((z_alpha + z_power) * sd / effect) ** 2))


def power_report(
    deltas_by_metric: Mapping[str, Sequence[float]],
    targets: Mapping[str, float],
) -> Dict[str, object]:
    report = {}
    for metric, deltas in deltas_by_metric.items():
        stats = cluster_mde(deltas)
        effect = float(targets[metric])
        stats["target_effect"] = effect
        stats["detects_target"] = stats["mde"] <= effect
        stats["required_clusters"] = required_clusters(stats["sd"], effect)
        stats["current_clusters"] = len(deltas)
        stats["cluster_multiplier"] = stats["required_clusters"] / max(1, len(deltas))
        report[metric] = stats
    return report
