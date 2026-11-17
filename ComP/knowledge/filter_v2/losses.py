from __future__ import annotations

import torch
import torch.nn.functional as F


def huber_masked(prediction, target, mask, delta: float = 1.0):
    loss = F.huber_loss(prediction, target, reduction="none", delta=delta)
    mask = mask.bool()
    return loss[mask].mean() if mask.any() else torch.zeros((), device=prediction.device)


def corr_bce(corr_logits, u_corr, mask):
    mask = mask.bool()
    if not mask.any():
        return torch.zeros((), device=corr_logits.device)
    labels = (u_corr > 0).float()
    return F.binary_cross_entropy_with_logits(corr_logits[mask], labels[mask])


def pairwise_magnitude(prediction, target, mask, eps: float = 1e-6):
    mask = mask.bool()
    total = []
    for row in range(prediction.size(0)):
        valid = mask[row].nonzero(as_tuple=False).flatten()
        if valid.numel() < 2:
            continue
        t = target[row, valid]
        if float(t.max()) <= eps:
            continue
        p = prediction[row, valid]
        diff_t = t.unsqueeze(1) - t.unsqueeze(0)
        diff_p = p.unsqueeze(1) - p.unsqueeze(0)
        upper = torch.triu(
            torch.ones_like(diff_t, dtype=torch.bool), diagonal=1
        )
        if upper.any():
            total.append(F.huber_loss(diff_p[upper], diff_t[upper], reduction="mean"))
    return torch.stack(total).mean() if total else torch.zeros((), device=prediction.device)


def pairwise_weighted(prediction, target, mask, eps: float = 1e-6):
    """Magnitude-weighted pairwise Huber with per-query scale normalization."""
    mask = mask.bool()
    total = []
    for row in range(prediction.size(0)):
        valid = mask[row].nonzero(as_tuple=False).flatten()
        if valid.numel() < 2:
            continue
        t = target[row, valid]
        if float(t.max()) <= eps:
            continue
        p = prediction[row, valid]
        pairs_t = []
        for i in range(valid.numel()):
            for j in range(i + 1, valid.numel()):
                pairs_t.append(abs(float(t[i] - t[j])))
        scale = max(
            float(t.std(unbiased=False)),
            float(torch.tensor(pairs_t).median()) if pairs_t else 0.0,
            eps,
        )
        diff_t = (t.unsqueeze(1) - t.unsqueeze(0)) / scale
        diff_p = (p.unsqueeze(1) - p.unsqueeze(0)) / scale
        weight = diff_t.abs()
        upper = torch.triu(
            torch.ones_like(diff_t, dtype=torch.bool), diagonal=1
        )
        if upper.any():
            total.append(
                (weight[upper] * F.huber_loss(
                    diff_p[upper], diff_t[upper], reduction="none"
                )).mean()
            )
    return torch.stack(total).mean() if total else torch.zeros((), device=prediction.device)


def listwise_positive(prediction, target, mask, temperature: float = 0.5):
    mask = mask.bool()
    terms = []
    for row in range(prediction.size(0)):
        valid = mask[row].nonzero(as_tuple=False).flatten()
        if valid.numel() < 2:
            continue
        t = target[row, valid]
        if float(t.max()) <= 1e-6:
            continue
        q = torch.softmax(t / temperature, dim=-1)
        p = torch.log_softmax(prediction[row, valid] / temperature, dim=-1)
        terms.append(-(q * p).sum())
    return torch.stack(terms).mean() if terms else torch.zeros((), device=prediction.device)


def filter_v2_loss(
    output,
    u_ce,
    u_corr,
    u_mae,
    mask,
    *,
    lambda_corr: float = 0.2,
    lambda_pair: float = 0.5,
    lambda_list: float = 0.5,
    list_temperature: float = 0.5,
    aux_mae: bool = False,
):
    loss_abs = huber_masked(output["utility"], u_ce, mask)
    loss_corr = corr_bce(output["correctness"], u_corr, mask)
    loss_pair = pairwise_magnitude(output["utility"], u_ce, mask)
    loss_pair_weighted = pairwise_weighted(output["utility"], u_ce, mask)
    loss_list = listwise_positive(output["utility"], u_ce, mask, list_temperature)
    loss = loss_abs + lambda_corr * loss_corr + lambda_pair * loss_pair + lambda_list * loss_list
    if aux_mae and output.get("mae_utility") is not None:
        loss = loss + huber_masked(output["mae_utility"], u_mae, mask)
    return {
        "loss": loss,
        "abs": loss_abs,
        "corr": loss_corr,
        "pair": loss_pair,
        "pair_weighted": loss_pair_weighted,
        "list": loss_list,
    }
