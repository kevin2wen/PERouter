from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np


def near_duplicate_report(
    embeddings: np.ndarray,
    thresholds: Sequence[float] = (0.90, 0.95, 0.98),
) -> dict:
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        return {"pairs": 0}
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normed = values / np.clip(norms, 1e-9, None)
    similarity = normed @ normed.T
    upper = similarity[np.triu_indices(values.shape[0], k=1)]
    report = {
        "pairs": int(upper.size),
        "cosine_mean": float(upper.mean()),
        "cosine_std": float(upper.std()),
        "cosine_p10": float(np.quantile(upper, 0.1)),
        "cosine_p50": float(np.quantile(upper, 0.5)),
        "cosine_p90": float(np.quantile(upper, 0.9)),
    }
    for threshold in thresholds:
        report[f"near_duplicate_rate_{threshold:.2f}"] = float(
            (upper >= threshold).mean()
        )
    return report


def relation_separation(
    embeddings: np.ndarray,
    relations: Sequence[str],
) -> dict:
    values = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normed = values / np.clip(norms, 1e-9, None)
    groups = defaultdict(list)
    for index, relation in enumerate(relations):
        groups[str(relation)].append(index)
    intra = {}
    for relation, indices in groups.items():
        if len(indices) < 2:
            continue
        sub = normed[indices]
        sim = sub @ sub.T
        upper = sim[np.triu_indices(sub.shape[0], k=1)]
        intra[relation] = {
            "cards": len(indices),
            "intra_cosine_mean": float(upper.mean()),
        }
    labels = sorted(groups)
    inter = {}
    for i, left in enumerate(labels):
        for right in labels[i + 1:]:
            a = normed[groups[left]]
            b = normed[groups[right]]
            inter[f"{left}|{right}"] = float((a @ b.T).mean())
    return {"intra": intra, "inter": inter}
