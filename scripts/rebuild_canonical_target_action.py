from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def canonical_action(utility: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return canonical (target_action, skip_label).

    target_action: 0=skip, 1..K = argmax rank + 1 over valid candidates.
    skip_label: 1 iff max valid utility <= 0.
    """
    masked = utility.masked_fill(~mask.bool(), float("-inf"))
    best_utility, best_index = masked.max(dim=-1)
    skip = best_utility <= 0.0
    action = torch.where(skip, torch.zeros_like(best_index), best_index + 1)
    return action, skip.float()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    cache = torch.load(args.candidate_cache, map_location="cpu", weights_only=False)
    utility = cache["utility"]
    mask = cache["candidate_mask"].bool()
    action, skip = canonical_action(utility, mask)
    legacy = cache.get("target_action")
    legacy_mismatch = None
    legacy_skip_rate = None
    if legacy is not None:
        legacy = legacy.long()
        legacy_mismatch = int((action != legacy).sum())
        legacy_skip_rate = float((legacy == 0).float().mean())
    cache = dict(cache)
    cache["target_action_canonical"] = action
    cache["skip_label"] = skip
    cache["label_schema_version"] = 2
    torch.save(cache, args.output)
    report = {
        "queries": int(action.numel()),
        "canonical_skip_rate": float(skip.mean()),
        "canonical_use_rate": float(1.0 - skip.mean()),
        "legacy_skip_rate": legacy_skip_rate,
        "legacy_action_mismatch": legacy_mismatch,
        "legacy_mismatch_rate": (
            None if legacy_mismatch is None else legacy_mismatch / action.numel()
        ),
        "source_cache_sha256": sha256_file(args.candidate_cache),
        "output_sha256": sha256_file(args.output),
        "label_schema_version": 2,
    }
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
