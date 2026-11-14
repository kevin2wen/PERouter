from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    parser.add_argument("--semantic-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    cache = torch.load(args.semantic_cache, map_location="cpu", weights_only=False)
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
    sample_ids = [row["sample_id"] for row in cache["records"]]
    dialogues = sorted(set(sample_ids))
    rng = random.Random(args.seed)
    rng.shuffle(dialogues)
    split_at = max(1, int(len(dialogues) * (1 - args.val_frac)))
    val_ids = set(dialogues[split_at:]) or set(dialogues)
    index = torch.tensor(
        [i for i, sid in enumerate(sample_ids) if sid in val_ids], dtype=torch.long
    )
    candidates = cache["candidate_vectors"][index].float()
    mask = cache["candidate_mask"][index].bool()
    utility = cache["single_utility"][index].float()
    generator = torch.Generator().manual_seed(args.seed)
    permuted = torch.randperm(candidates.size(0), generator=generator)
    shuffled = candidates[permuted]
    flat = candidates.reshape(-1, candidates.size(-1))
    random_vectors = flat[
        torch.randint(0, flat.size(0), (candidates.shape[0], candidates.shape[1]),
                      generator=generator)
    ]
    controls = {
        "real": candidates,
        "shuffled": shuffled,
        "random": random_vectors,
        "zero": torch.zeros_like(candidates),
    }
    results = {}
    for name, vectors in controls.items():
        with torch.no_grad():
            output = model(
                cache["observed_mask"][index].float().to(device),
                cache["target_modality"][index].float().to(device),
                cache["missing_pattern"][index].long().to(device),
                cache["missing_rate"][index].float().to(device),
                cache["prompt_vectors"][index].float().to(device),
                vectors.to(device),
                mask.to(device),
                cache["source_confidence"][index].float().to(device),
                (cache["conflict_score"][index].float().to(device)
                 if cache.get("conflict_score") is not None else None),
            )
        predicted = output["utility"].cpu()
        mae = float((predicted[mask] - utility[mask]).abs().mean())
        rhos = [
            spearman(predicted[row, mask[row]], utility[row, mask[row]])
            for row in range(mask.size(0))
            if int(mask[row].sum()) >= 2
        ]
        predicted_index = predicted.masked_fill(~mask, -1e9).argmax(dim=-1)
        true_best = utility.masked_fill(~mask, -1e9).max(dim=-1).values
        predicted_true = utility.gather(1, predicted_index[:, None]).squeeze(1)
        regret = float((true_best - predicted_true).mean())
        helpful = ((utility > 0) & mask).flatten()
        auc = bootstrap_auc_ci(
            predicted.flatten()[mask.flatten()].numpy(),
            helpful[mask.flatten()].numpy(),
            n_boot=1000,
        )
        results[name] = {
            "utility_mae": mae,
            "spearman": float(np.nanmean(rhos)),
            "auc": auc["auc"],
            "auc_ci_low": auc["ci_low"],
            "auc_ci_high": auc["ci_high"],
            "top1_regret": regret,
        }
    payload = {"queries": int(index.numel()), "controls": results}
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
