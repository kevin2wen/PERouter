from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

from ComP.knowledge.power_analysis import cluster_mde, required_clusters

DEFAULT_TARGETS = {
    "utility_mae": 0.02,
    "top1_regret_reduction": 0.02,
    "utility_spearman": 0.05,
    "helpfulness_auc": 0.05,
    "skip_recall": 0.05,
}


def msc_power_report(
    deltas_by_metric: Mapping[str, Sequence[float]],
    targets: Mapping[str, float] | None = None,
) -> Dict[str, dict]:
    """Approximate cluster-level MDE for SingleEU metrics.

    Inputs are per-dialogue paired deltas (e.g. learned minus oracle/regret).
    Normal approximation matches the existing cluster_mde in power_analysis.
    """
    targets = dict(targets or DEFAULT_TARGETS)
    report = {}
    for metric, deltas in deltas_by_metric.items():
        values = np.asarray(deltas, dtype=np.float64)
        if values.size < 2:
            raise ValueError(f"metric {metric} needs >=2 clusters")
        stats = cluster_mde(values)
        target = float(targets.get(metric, 0.02))
        stats["target_effect"] = target
        stats["detects_target"] = stats["mde"] <= target
        stats["required_clusters"] = required_clusters(stats["sd"], target)
        report[metric] = stats
    return report


def bootstrap_auc_ci(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    n_boot: int = 1000,
    seed: int = 66,
) -> dict:
    """Bootstrap AUC CI (record-level; use dialogue clustering upstream)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if labels.all() or (~labels).all():
        return {"auc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    def auc(s, y):
        order = np.argsort(s)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, s.size + 1)
        n_pos = y.sum()
        n_neg = y.size - n_pos
        return float(
            (ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        )

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        index = rng.integers(0, scores.size, size=scores.size)
        if labels[index].all() or (~labels[index]).all():
            continue
        draws.append(auc(scores[index], labels[index]))
    point = auc(scores, labels)
    lo, hi = np.quantile(draws, [0.025, 0.975]) if draws else (np.nan, np.nan)
    return {"auc": point, "ci_low": float(lo), "ci_high": float(hi)}
