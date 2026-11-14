from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from .subsets import valid_subset_masks


def query_id(sample_id: str, utterance_id: str, target_modality: str) -> str:
    payload = f"{sample_id}|{utterance_id}|{target_modality}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_state(cache: Dict) -> Dict[str, torch.Tensor]:
    """Derive discrete query-state tensors from an existing candidate cache."""
    missing_patterns: List[str] = cache["missing_patterns"]
    count = len(missing_patterns)
    observed = torch.zeros(count, 3, dtype=torch.float32)
    rate = torch.zeros(count, dtype=torch.float32)
    pattern_ids = torch.zeros(count, dtype=torch.long)
    for index, pattern in enumerate(missing_patterns):
        values = [int(char) for char in str(pattern)]
        if len(values) != 3:
            raise ValueError(f"invalid missing_pattern: {pattern!r}")
        observed[index] = torch.tensor(
            [1.0 - value for value in values], dtype=torch.float32
        )
        rate[index] = sum(values) / 3.0
        pattern_ids[index] = int("".join(str(value) for value in values), 2)
    target = torch.nn.functional.one_hot(
        cache["target_modalities"].long(), num_classes=3
    ).to(torch.float32)
    return {
        "observed_mask": observed,
        "target_modality": target,
        "missing_pattern": pattern_ids,
        "missing_rate": rate,
    }


def build_state_cache(cache: Dict, *, extra: Dict | None = None) -> Dict:
    state = derive_state(cache)
    records = cache["records"]
    query_ids = [
        query_id(
            str(record["sample_id"]),
            str(record["utterance_id"]),
            str(record["target_modality"]),
        )
        for record in records
    ]
    return {
        "msc_format_version": 1,
        "source_cache_sha256": cache.get("cache_sha256"),
        "cache_key": cache.get("cache_key"),
        "split": cache.get("split"),
        "query_ids": query_ids,
        **state,
        "candidate_ids": cache["candidate_ids"],
        "candidate_mask": cache["candidate_mask"],
        "candidate_vectors": cache["candidate_vectors"],
        "prompt_vectors": cache["prompt_vectors"],
        "source_confidence": cache["source_confidence"],
        "conflict_score": cache.get("conflict_score"),
        "utility": cache["utility"],
        "skip_utility": cache["skip_utility"],
        "target_action": cache["target_action"],
        "target_labels": cache["target_labels"],
        "extra": extra or {},
    }


def save_state_cache(payload: Dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def load_state_cache(path: str | Path) -> Dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("msc_format_version") != 1:
        raise ValueError("unsupported msc state cache version")
    return payload


def subset_masks_batch(
    candidate_mask: torch.Tensor, max_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad per-query valid subset masks to [N,Smax,K] plus per-query counts."""
    rows = [
        valid_subset_masks(candidate_mask[index], max_k)
        for index in range(candidate_mask.size(0))
    ]
    counts = torch.tensor([row.size(0) for row in rows], dtype=torch.long)
    if int(counts.max()) == 0:
        raise ValueError("no query has any valid candidate")
    padded = torch.zeros(
        candidate_mask.size(0), int(counts.max()), max_k, dtype=torch.bool
    )
    for index, row in enumerate(rows):
        if row.numel():
            padded[index, : row.size(0)] = row
    return padded, counts
