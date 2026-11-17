from __future__ import annotations

import numpy as np


def utility_spearman(prediction, target, mask) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    values = []
    for row in range(prediction.shape[0]):
        p = prediction[row][mask[row]]
        t = target[row][mask[row]]
        if p.size < 2:
            continue
        rp = np.argsort(np.argsort(p))
        rt = np.argsort(np.argsort(t))
        if rp.std() == 0 or rt.std() == 0:
            continue
        values.append(float(np.corrcoef(rp, rt)[0, 1]))
    return float(np.mean(values)) if values else float("nan")


def helpfulness_auc(prediction, target, mask) -> float:
    scores = np.asarray(prediction, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    labels = (np.asarray(target, dtype=np.float64)[np.asarray(mask, dtype=bool)] > 0)
    if scores.size == 0 or labels.all() or (~labels).all():
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    n_pos = labels.sum()
    n_neg = scores.size - n_pos
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def top1_regret(prediction, target, mask) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    regrets = []
    for row in range(prediction.shape[0]):
        valid = np.where(mask[row])[0]
        if valid.size == 0:
            continue
        best = target[row, valid].max()
        chosen = valid[int(np.argmax(prediction[row, valid]))]
        regrets.append(best - target[row, chosen])
    return float(np.mean(regrets)) if regrets else float("nan")


def query_macro_auc(prediction, target, mask) -> float:
    """Mean per-query AUC over valid candidates (chance=0.5)."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    values = []
    for row in range(prediction.shape[0]):
        valid = np.where(mask[row])[0]
        if valid.size < 2:
            continue
        scores = prediction[row, valid]
        labels = target[row, valid] > 0
        if labels.all() or (~labels).all():
            continue
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, scores.size + 1)
        n_pos = labels.sum()
        n_neg = scores.size - n_pos
        values.append(
            (ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        )
    return float(np.mean(values)) if values else float("nan")


def action_regret(prediction, target, mask, tau: float = 0.0) -> float:
    """Regret against max(0, u_1..u_K), including the skip action."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    regrets = []
    for row in range(prediction.shape[0]):
        valid = np.where(mask[row])[0]
        if valid.size == 0:
            continue
        best = max(0.0, float(target[row, valid].max()))
        predicted = prediction[row, valid]
        index = int(np.argmax(predicted))
        action_utility = (
            float(target[row, valid[index]]) if predicted[index] > tau else 0.0
        )
        regrets.append(best - action_utility)
    return float(np.mean(regrets)) if regrets else float("nan")
