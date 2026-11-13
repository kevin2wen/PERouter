from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Sequence

import numpy as np


def metric_value(labels, predictions, metric: str) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if labels.size == 0:
        return float("nan")
    binary_labels = labels > 0
    binary_predictions = predictions > 0
    if metric == "acc":
        return float(np.mean(binary_labels == binary_predictions))
    if metric == "mae":
        return float(np.mean(np.abs(labels - predictions)))
    if metric == "f1":
        tp = float(np.sum(binary_labels & binary_predictions))
        fp = float(np.sum(~binary_labels & binary_predictions))
        fn = float(np.sum(binary_labels & ~binary_predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    if metric == "corr":
        if labels.size < 2 or labels.std() == 0 or predictions.std() == 0:
            return 0.0
        return float(np.corrcoef(labels, predictions)[0, 1])
    raise ValueError(f"unknown metric: {metric}")


def index_rows(rows: Sequence[Mapping[str, object]]) -> Dict[str, Dict[str, List]]:
    table: Dict[str, Dict[str, List]] = defaultdict(
        lambda: {"labels": [], "predictions": [], "cluster": []}
    )
    for row in rows:
        entry = table[str(row["mode"])]
        entry["labels"].append(float(row["label"]))
        entry["predictions"].append(float(row["prediction"]))
        entry["cluster"].append(
            str(row.get("dialogue_id") or row["sample_id"])
        )
    return table


def paired_cluster_bootstrap(
    rows: Sequence[Mapping[str, object]],
    comparison: str,
    baseline: str,
    *,
    metric: str,
    n_boot: int = 10000,
    seed: int = 66,
    alpha: float = 0.05,
) -> Dict[str, float]:
    table = index_rows(rows)
    if comparison not in table or baseline not in table:
        raise KeyError("missing mode for paired bootstrap")
    keys = [
        (mode, index)
        for mode in (comparison, baseline)
        for index in range(len(table[mode]["labels"]))
    ]
    cluster_map: Dict[str, List[int]] = defaultdict(list)
    for mode in (comparison, baseline):
        for index, cluster in enumerate(table[mode]["cluster"]):
            cluster_map[(mode, cluster)].append(index)
    clusters = sorted(
        {cluster for _, cluster in cluster_map}
    )
    per_mode = {
        mode: {
            cluster: np.asarray(
                [
                    index
                    for index, value in enumerate(table[mode]["cluster"])
                    if value == cluster
                ],
                dtype=np.int64,
            )
            for cluster in clusters
        }
        for mode in (comparison, baseline)
    }
    del keys
    labels = {mode: np.asarray(table[mode]["labels"]) for mode in (comparison, baseline)}
    predictions = {
        mode: np.asarray(table[mode]["predictions"]) for mode in (comparison, baseline)
    }

    def delta_for(selected: Sequence[str]) -> float:
        comp_arrays = [per_mode[comparison][cluster] for cluster in selected]
        base_arrays = [per_mode[baseline][cluster] for cluster in selected]
        comp_idx = np.concatenate(comp_arrays)
        base_idx = np.concatenate(base_arrays)
        comp = metric_value(
            labels[comparison][comp_idx], predictions[comparison][comp_idx], metric
        )
        base = metric_value(
            labels[baseline][base_idx], predictions[baseline][base_idx], metric
        )
        return comp - base

    observed = delta_for(clusters)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for index in range(n_boot):
        picks = rng.integers(0, len(clusters), size=len(clusters))
        draws[index] = delta_for([clusters[pick] for pick in picks])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    loo = [
        delta_for([cluster for cluster in clusters if cluster != left])
        for left in clusters
    ]
    per_cluster = [delta_for([cluster]) for cluster in clusters]
    return {
        "comparison": comparison,
        "baseline": baseline,
        "metric": metric,
        "delta": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "boot_std": float(draws.std(ddof=1)),
        "loo_min": float(min(loo)),
        "loo_max": float(max(loo)),
        "clusters": float(len(clusters)),
        "positive_clusters": float(sum(value > 0 for value in per_cluster)),
        "negative_clusters": float(sum(value < 0 for value in per_cluster)),
        "zero_clusters": float(sum(value == 0 for value in per_cluster)),
    }


def wild_cluster_bootstrap(
    per_cluster: Mapping[str, float],
    *,
    n_boot: int = 10000,
    seed: int = 66,
    alpha: float = 0.05,
) -> Dict[str, float]:
    values = np.asarray(list(per_cluster.values()), dtype=np.float64)
    if values.size == 0:
        raise ValueError("no clusters for wild bootstrap")
    observed = float(values.mean())
    centered = values - observed
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_boot, values.size))
    draws = observed + (signs * centered).mean(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {
        "delta": observed,
        "wild_ci_low": float(lo),
        "wild_ci_high": float(hi),
        "clusters": float(values.size),
    }


def per_cluster_metric_delta(
    rows: Sequence[Mapping[str, object]],
    comparison: str,
    baseline: str,
    *,
    metric: str,
) -> Dict[str, float]:
    table = index_rows(rows)
    if comparison not in table or baseline not in table:
        raise KeyError("missing mode for metric delta")
    clusters = sorted(
        {
            str(row.get("dialogue_id") or row["sample_id"])
            for row in rows
            if row["mode"] in (comparison, baseline)
        }
    )
    results = {}
    for cluster in clusters:
        left = [
            row for row in rows
            if row["mode"] == comparison
            and str(row.get("dialogue_id") or row["sample_id"]) == cluster
        ]
        right = [
            row for row in rows
            if row["mode"] == baseline
            and str(row.get("dialogue_id") or row["sample_id"]) == cluster
        ]
        if not left or not right:
            continue
        comp = metric_value(
            [float(r["label"]) for r in left],
            [float(r["prediction"]) for r in left],
            metric,
        )
        base = metric_value(
            [float(r["label"]) for r in right],
            [float(r["prediction"]) for r in right],
            metric,
        )
        results[cluster] = comp - base
    return results
