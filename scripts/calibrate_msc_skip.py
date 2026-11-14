from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.knowledge.msc.calibration import fit_calibration, save_calibration
from ComP.knowledge.msc.power import bootstrap_auc_ci
from ComP.knowledge.msc.single_eu import MSCSingleEU


def spearman(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    cache = torch.load(args.single_cache, map_location="cpu", weights_only=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = MSCSingleEU(
        ckpt["candidate_dim"], ckpt["prompt_dim"],
        state_dim=ckpt["state_dim"], hidden_dim=ckpt["hidden_dim"],
        use_semantic=ckpt["use_semantic"],
        use_cross_attention=ckpt["use_cross_attention"],
        use_skip_head=ckpt["use_skip_head"],
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    sample_ids = [row["sample_id"] for row in cache.get("records", [])]
    dialogues = sorted(set(sample_ids))
    rng = random.Random(args.seed)
    rng.shuffle(dialogues)
    split_at = max(1, int(len(dialogues) * (1 - args.val_frac)))
    val_ids = set(dialogues[split_at:]) or set(dialogues)
    index = torch.tensor(
        [i for i, sid in enumerate(sample_ids) if sid in val_ids], dtype=torch.long
    )
    with torch.no_grad():
        output = model(
            cache["observed_mask"][index].to(device),
            cache["target_modality"][index].to(device),
            cache["missing_pattern"][index].to(device),
            cache["missing_rate"][index].to(device),
            cache["prompt_vectors"][index].to(device),
            cache["candidate_vectors"][index].to(device),
            cache["candidate_mask"][index].to(device),
            cache["source_confidence"][index].to(device),
            (cache["conflict_score"][index].to(device)
             if cache.get("conflict_score") is not None else None),
        )
    utility = output["utility"].cpu()
    mask = cache["candidate_mask"][index].bool()
    target = cache["single_utility"][index]
    valid = mask
    mae = float((utility[valid] - target[valid]).abs().mean())
    rho_values = [
        spearman(utility[row, mask[row]].numpy(), target[row, mask[row]].numpy())
        for row in range(mask.size(0))
        if int(mask[row].sum()) >= 2
    ]
    helpful = ((target > 0) & valid).flatten()
    scores = utility.flatten()
    auc = bootstrap_auc_ci(
        scores[valid.flatten()].numpy(),
        helpful[valid.flatten()].numpy(),
        n_boot=1000,
    )
    predicted_index = utility.masked_fill(~valid, -1e9).argmax(dim=-1)
    predicted_true = target.gather(1, predicted_index[:, None]).squeeze(1)
    true_best = target.masked_fill(~valid, -1e9).max(dim=-1).values
    top1_regret = float((true_best - predicted_true).mean())
    calibration = fit_calibration(
        output["skip_logit"].cpu().numpy(),
        cache["skip_label"][index].numpy(),
        utility_scorer_hash=ckpt["utility_scorer_hash"],
        state_encoder_hash=ckpt["state_encoder_hash"],
        candidate_schema_hash=hashlib.sha256(
            json.dumps(sorted(cache.keys())).encode()
        ).hexdigest(),
    )
    calibration["utility_mae"] = mae
    calibration["utility_spearman"] = float(np.nanmean(rho_values))
    calibration["helpful_count"] = int(helpful.sum())
    calibration["candidates"] = int(valid.sum())
    calibration["helpfulness_auc"] = auc["auc"]
    calibration["helpfulness_auc_ci_low"] = auc["ci_low"]
    calibration["helpfulness_auc_ci_high"] = auc["ci_high"]
    calibration["top1_regret"] = top1_regret
    save_calibration(calibration, args.output)
    print(json.dumps(calibration, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
