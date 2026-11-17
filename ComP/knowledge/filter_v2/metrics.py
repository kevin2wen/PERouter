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
