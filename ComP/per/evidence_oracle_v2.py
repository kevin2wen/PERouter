from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch

GAMMA_VALUES = (0.1, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class OracleAction:
    use_evidence: bool
    candidate_rank: int
    gamma: float

    def __post_init__(self) -> None:
        if not self.use_evidence and (self.candidate_rank != -1 or self.gamma != 0.0):
            raise ValueError("skip evidence requires rank=-1 and gamma=0")
        if self.use_evidence and self.candidate_rank < 0:
            raise ValueError("evidence action requires non-negative rank")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0,1]")


def normalize_action(action: OracleAction) -> OracleAction:
    if not action.use_evidence:
        return OracleAction(False, -1, 0.0)
    if action.gamma == 0.0:
        return OracleAction(False, -1, 0.0)
    return action


def action_grid(top_k: int) -> List[OracleAction]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    return [OracleAction(False, -1, 0.0)] + [
        OracleAction(True, rank, gamma)
        for rank in range(top_k)
        for gamma in GAMMA_VALUES
    ]


def validate_oracle_v2_split(split: str) -> None:
    if split != "validation":
        raise PermissionError("Oracle v2 is validation-only and must never run on test")


def select_actions(
    scores: torch.Tensor,
    active: torch.BoolTensor,
    action_count: int,
) -> torch.LongTensor:
    if scores.ndim != 4 or scores.size(-1) != action_count:
        raise ValueError("scores must be [B,T,3,A]")
    if active.shape != scores.shape[:3]:
        raise ValueError("active must be [B,T,3]")
    if active.dtype != torch.bool:
        raise TypeError("active must be bool")
    if not torch.isfinite(scores).all():
        raise FloatingPointError("action scores contain non-finite values")
    masked = scores.masked_fill(~active.unsqueeze(-1), torch.finfo(scores.dtype).min)
    selected = masked.argmax(dim=-1).long()
    valid = active.any()
    if not valid:
        return torch.zeros_like(selected)
    return selected


def action_histogram(
    selected: torch.LongTensor,
    active: torch.BoolTensor,
    actions: Iterable[OracleAction],
) -> Dict[str, int]:
    actions = list(actions)
    if selected.shape != active.shape:
        raise ValueError("selected and active must be [B,T,3]")
    result = {
        "skip_evidence": 0,
        **{
            f"rank{action.candidate_rank}_gamma{action.gamma:g}": 0
            for action in actions
            if action.use_evidence
        },
    }
    for index in selected[active].detach().cpu().tolist():
        action = normalize_action(actions[int(index)])
        key = (
            "skip_evidence"
            if not action.use_evidence
            else f"rank{action.candidate_rank}_gamma{action.gamma:g}"
        )
        result[key] = result.get(key, 0) + 1
    return result


def action_parameters(
    selected: torch.LongTensor,
    active: torch.BoolTensor,
    actions: Iterable[OracleAction],
) -> Tuple[torch.LongTensor, torch.Tensor, torch.BoolTensor]:
    actions = list(actions)
    if selected.shape != active.shape:
        raise ValueError("selected and active must be [B,T,3]")
    ranks = torch.full_like(selected, -1)
    gammas = torch.zeros_like(selected, dtype=torch.float32)
    use = torch.zeros_like(active)
    for index, raw_action in enumerate(actions):
        action = normalize_action(raw_action)
        mask = selected == index
        use = use | (mask & action.use_evidence)
        ranks = torch.where(mask, torch.full_like(ranks, action.candidate_rank), ranks)
        gammas = torch.where(
            mask, torch.full_like(gammas, action.gamma), gammas
        )
    return ranks, gammas, use
