from __future__ import annotations

from typing import List, Tuple

import torch


def nonempty_subset_masks(n_candidates: int) -> List[int]:
    """Bitmask values for all non-empty subsets, ordered 1..2**n-1."""
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")
    if n_candidates > 20:
        raise ValueError("subset enumeration only supports <= 20 candidates")
    return list(range(1, 1 << n_candidates))


def subset_indices(mask: int) -> Tuple[int, ...]:
    """Candidate indices selected by an integer bitmask."""
    return tuple(index for index in range(mask.bit_length()) if (mask >> index) & 1)


def valid_subset_masks(
    candidate_valid: torch.BoolTensor,
    max_k: int,
) -> torch.BoolTensor:
    """Return [S, max_k] bool masks over the valid candidate positions only.

    candidate_valid: [K], True=valid candidate. The returned S equals
    2**count - 1, so with all K valid it is 31 for K=5. Mask rows are ordered
    by increasing integer value of the subset bitmask.
    """
    if candidate_valid.ndim != 1:
        raise ValueError("candidate_valid must be 1-D")
    positions = candidate_valid.nonzero(as_tuple=False).flatten().tolist()
    if not positions:
        return torch.zeros(0, int(max_k), dtype=torch.bool)
    values = nonempty_subset_masks(len(positions))
    masks = torch.zeros(len(values), int(max_k), dtype=torch.bool)
    for row, value in enumerate(values):
        for local in range(len(positions)):
            if (value >> local) & 1:
                masks[row, positions[local]] = True
    return masks
