from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=66)
    args = parser.parse_args()
    cache = torch.load(
        args.candidate_cache, map_location="cpu", weights_only=False
    )
    mask = cache["candidate_mask"].bool()
    utility = cache["utility"]
    valid = mask.any(dim=-1)
    mask = mask[valid]
    utility = utility[valid]
    candidates = int(mask.sum())
    helpful = (utility > 0) & mask
    harmful = (utility < 0) & mask
    neutral = (utility == 0) & mask
    best = utility.masked_fill(~mask, float("-inf")).max(dim=-1).values
    top1 = utility[:, 0].masked_fill(~mask[:, 0], 0.0)
    always_use = float(top1.mean())
    oracle = float(best.clamp_min(0).mean())
    harmful_best = float((best < 0).float().mean())
    payload = {
        "records": int(mask.size(0)),
        "candidates": candidates,
        "helpful_prevalence": float(helpful.sum() / candidates),
        "harmful_prevalence": float(harmful.sum() / candidates),
        "neutral_prevalence": float(neutral.sum() / candidates),
        "beneficial_query_prevalence": float((best > 0).float().mean()),
        "harmful_best_query_prevalence": harmful_best,
        "always_use_utility": always_use,
        "always_skip_utility": 0.0,
        "oracle_utility": oracle,
        "always_use_oracle_gap": oracle - always_use,
        "harmful_rate_sufficient": bool(
            float(harmful.sum() / candidates) >= 0.05
            and oracle - always_use >= 0.05
        ),
    }
    Path(args.output).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
