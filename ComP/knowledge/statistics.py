from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def cluster_deltas(
    rows: Sequence[Mapping[str, object]],
    baseline: str,
    comparison: str,
    *,
    metric: str = "abs_error",
    cluster_key: str = "dialogue_id",
) -> Dict[str, List[float]]:
    """Return per-cluster deltas of ``comparison - baseline``.

    Rows must be keyed by ``(sample_id, utterance_id)`` and contain ``mode``.
    """
    keyed: Dict[Tuple[str, str], Dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["sample_id"]), str(row["utterance_id"]))
        keyed[key][str(row["mode"])] = row
    grouped: Dict[str, List[float]] = defaultdict(list)
    for _, by_mode in keyed.items():
        if baseline not in by_mode or comparison not in by_mode:
            continue
        left = by_mode[comparison]
        right = by_mode[baseline]
        cluster = str(left.get(cluster_key) or left["sample_id"])
        grouped[cluster].append(float(left[metric]) - float(right[metric]))
    return dict(grouped)


def cluster_bootstrap_mean(
    grouped: Mapping[str, Sequence[float]],
    *,
    n_boot: int = 10000,
    seed: int = 66,
    alpha: float = 0.05,
) -> Dict[str, float]:
    clusters = [np.asarray(values, dtype=np.float64) for values in grouped.values()]
    if not clusters:
        raise ValueError("no paired clusters")
    point = float(np.mean(np.concatenate(clusters)))
    rng = np.random.default_rng(seed)
    n_clusters = len(clusters)
    draws = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        picks = rng.integers(0, n_clusters, size=n_clusters)
        draws[index] = float(np.mean([clusters[pick].mean() for pick in picks]))
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "clusters": float(n_clusters),
        "pairs": float(sum(values.size for values in clusters)),
        "delta": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "boot_mean": float(draws.mean()),
        "boot_std": float(draws.std(ddof=1)),
    }


def cluster_signflip_test(
    grouped: Mapping[str, Sequence[float]],
    *,
    n_perm: int = 10000,
    seed: int = 66,
) -> Dict[str, float]:
    cluster_means = np.asarray(
        [np.mean(values) for values in grouped.values()], dtype=np.float64
    )
    observed = float(
        np.mean(np.concatenate([np.asarray(v) for v in grouped.values()]))
    )
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_perm, cluster_means.size))
    perm = (signs * cluster_means).mean(axis=1)
    p_value = float(np.mean(np.abs(perm) >= abs(observed)))
    return {"observed": observed, "p_value": p_value, "n_perm": float(n_perm)}


def holm_adjust(p_values: Iterable[float]) -> List[float]:
    values = list(p_values)
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [0.0] * len(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = (total - rank) * values[index]
        running = max(running, min(1.0, candidate))
        adjusted[index] = running
    return adjusted


FAMILIES = {
    "A": (
        ("card", "random_card"),
        ("card", "shuffled_card"),
        ("card", "source_only"),
    ),
    "B": (
        ("listwise_filter", "confidence"),
        ("listwise_filter", "random_action"),
        ("pairwise_filter", "confidence"),
        ("pairwise_filter", "random_action"),
    ),
    "C": (("oracle_card", "empty"),),
}


def bootstrap_family(
    rows: Sequence[Mapping[str, object]],
    family: str,
    *,
    n_boot: int = 10000,
    n_perm: int = 10000,
    seed: int = 66,
) -> List[Dict[str, object]]:
    results = []
    for comparison, baseline in FAMILIES[family]:
        grouped = cluster_deltas(rows, baseline, comparison)
        if not grouped:
            continue
        stats = cluster_bootstrap_mean(grouped, n_boot=n_boot, seed=seed)
        test = cluster_signflip_test(grouped, n_perm=n_perm, seed=seed)
        results.append(
            {
                "family": family,
                "comparison": comparison,
                "baseline": baseline,
                "metric": "abs_error",
                **stats,
                "p_perm": test["p_value"],
            }
        )
    adjusted = holm_adjust([item["p_perm"] for item in results])
    for item, p_adj in zip(results, adjusted):
        item["p_holm"] = p_adj
    return results
