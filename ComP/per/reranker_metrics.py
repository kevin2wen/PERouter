from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F


def listwise_loss(logits: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
    if logits.ndim != 2 or targets.shape != (logits.size(0),):
        raise ValueError("logits must be [N,A] and targets [N]")
    return F.cross_entropy(logits, targets.long())


def pairwise_loss(
    logits: torch.Tensor,
    targets: torch.LongTensor,
    candidate_mask: torch.BoolTensor,
    margin: float = 0.1,
) -> torch.Tensor:
    if logits.ndim != 2 or targets.shape != (logits.size(0),):
        raise ValueError("invalid pair loss shapes")
    if candidate_mask.shape != logits[:, 1:].shape:
        raise ValueError("candidate_mask must match rank logits")
    losses = []
    for index in range(logits.size(0)):
        target = int(targets[index])
        positive = logits[index, target]
        negatives = []
        if target == 0:
            negatives = logits[index, 1:][candidate_mask[index]]
        else:
            negatives = torch.cat([
                logits[index, :1],
                logits[index, 1:][
                    candidate_mask[index] & (torch.arange(logits.size(1) - 1, device=logits.device) + 1 != target)
                ],
            ])
        if negatives.numel():
            losses.append(F.softplus(margin - positive + negatives).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def utility_regression_loss(
    logits: torch.Tensor,
    delta_utility: torch.Tensor,
    candidate_mask: torch.BoolTensor,
) -> torch.Tensor:
    if logits.ndim != 2 or delta_utility.shape != logits.shape:
        raise ValueError("logits and delta_utility must be [N,K+1]")
    if candidate_mask.shape != logits[:, 1:].shape:
        raise ValueError("candidate_mask must be [N,K]")
    valid = torch.cat([
        torch.ones_like(candidate_mask[:, :1]),
        candidate_mask,
    ], dim=1)
    return F.smooth_l1_loss(logits[valid], delta_utility[valid])


def dialogue_folds(sample_ids: Sequence[str], folds: int, seed: int) -> List[List[int]]:
    if folds < 2:
        raise ValueError("folds must be >= 2")
    dialogues = sorted(set(str(value) for value in sample_ids))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permuted = [dialogues[index] for index in torch.randperm(len(dialogues), generator=generator).tolist()]
    groups = [[] for _ in range(folds)]
    for index, dialogue in enumerate(permuted):
        groups[index % folds].append(dialogue)
    assignments = []
    for fold, values in enumerate(groups):
        fold_set = set(values)
        assignments.append([index for index, sample in enumerate(sample_ids) if str(sample) in fold_set])
    return assignments


def action_metrics(
    logits: torch.Tensor,
    delta_utility: torch.Tensor,
    candidate_mask: torch.BoolTensor,
    oracle_actions: torch.LongTensor,
) -> Dict[str, float]:
    if logits.ndim != 2 or delta_utility.shape != logits.shape:
        raise ValueError("logits and delta_utility must match [N,A]")
    predicted = logits.argmax(dim=-1)
    oracle_value = delta_utility.gather(1, oracle_actions[:, None]).squeeze(1)
    predicted_value = delta_utility.gather(1, predicted[:, None]).squeeze(1)
    regret = oracle_value - predicted_value
    return {
        "top1_action_agreement": float((predicted == oracle_actions).float().mean().item()),
        "policy_delta_utility_mean": float(predicted_value.mean().item()),
        "oracle_gain_mean": float(oracle_value.mean().item()),
        "regret_mean": float(regret.mean().item()),
        "skip_rate": float((predicted == 0).float().mean().item()),
        "use_evidence_rate": float((predicted != 0).float().mean().item()),
    }
