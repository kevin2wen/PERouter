from __future__ import annotations

from itertools import product
from typing import List, Sequence


def joint_action_space(options_per_target: Sequence[Sequence[int]]) -> List[tuple]:
    """All combinations of per-target options (None=skip, int=rank)."""
    return [tuple(combo) for combo in product(*options_per_target)]


def local_sum(utilities_per_target: Sequence[float]) -> float:
    return float(sum(utilities_per_target))


def joint_interaction_error(realized_joint: float, local: float) -> float:
    return float(realized_joint - local)
