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
    parser.add_argument("--seeds", type=int, default=20)
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
    base = dict(
        obs=cache["observed_mask"][index].float().to(device),
        target=cache["target_modality"][index].float().to(device),
        pattern=cache["missing_pattern"][index].long().to(device),
        rate=cache["missing_rate"][index].float().to(device),
        prompt=cache["prompt_vectors"][index].float().to(device),
        mask=cache["candidate_mask"][index].bool().to(device),
        confidence=cache["source_confidence"][index].float().to(device),
        conflict=(cache["conflict_score"][index].float().to(device)
                  if cache.get("conflict_score") is not None else None),
    )
    candidates = cache["candidate_vectors"][index].float().to(device)
    utility = cache["single_utility"][index].float().to(device)
    mask = base["mask"]
    helpful = ((utility > 0) & mask)

    def measure(vectors):
        with torch.no_grad():
            output = model(
                base["obs"], base["target"], base["pattern"], base["rate"],
                base["prompt"], vectors, base["mask"], base["confidence"],
                base["conflict"],
            )
        predicted = output["utility"].cpu()
        target = utility.cpu()
        mask_cpu = mask.cpu()
        mae = float((predicted[mask_cpu] - target[mask_cpu]).abs().mean())
        rhos = [
            spearman(predicted[row, mask_cpu[row]], target[row, mask_cpu[row]])
            for row in range(mask_cpu.size(0))
            if int(mask_cpu[row].sum()) >= 2
        ]
        pred_index = predicted.masked_fill(~mask_cpu, -1e9).argmax(dim=-1)
        best = target.masked_fill(~mask_cpu, -1e9).max(dim=-1).values
        actual = target.gather(1, pred_index[:, None]).squeeze(1)
        regret = float((best - actual).mean())
        auc = bootstrap_auc_ci(
            predicted.flatten()[mask_cpu.flatten()].numpy(),
            helpful.cpu().flatten()[mask_cpu.flatten()].numpy(),
            n_boot=500,
        )
        return {
            "auc": auc["auc"],
            "spearman": float(np.nanmean(rhos)),
            "top1_regret": regret,
            "utility_mae": mae,
        }

    real = measure(candidates)
    flat = candidates.reshape(-1, candidates.size(-1))
    random_results = []
    shuffled_results = []
    for seed in range(args.seeds):
        generator = torch.Generator(device=candidates.device).manual_seed(seed)
        random_vectors = flat[
            torch.randint(0, flat.size(0), (candidates.shape[0], candidates.shape[1]),
                          generator=generator, device=candidates.device)
        ]
        perm = torch.randperm(candidates.size(0), generator=generator,
                              device=candidates.device)
        random_results.append(measure(random_vectors))
        shuffled_results.append(measure(candidates[perm]))

    def summarize(values, prefix):
        keys = values[0].keys()
        return {
            key: {
                "median": float(np.median([v[key] for v in values])),
                "p25": float(np.quantile([v[key] for v in values], 0.25)),
                "p75": float(np.quantile([v[key] for v in values], 0.75)),
                "min": float(np.min([v[key] for v in values])),
                "max": float(np.max([v[key] for v in values])),
            }
            for key in keys
        }

    deltas = {
        "real_minus_random": {
            key: [real[key] - v[key] for v in random_results]
            for key in real
        },
        "real_minus_shuffled": {
            key: [real[key] - v[key] for v in shuffled_results]
            for key in real
        },
    }
    report = {
        "real": real,
        "random_summary": summarize(random_results, "random"),
        "shuffled_summary": summarize(shuffled_results, "shuffled"),
        "deltas": {
            name: {
                key: {
                    "median": float(np.median(values)),
                    "p25": float(np.quantile(values, 0.25)),
                    "p75": float(np.quantile(values, 0.75)),
                }
                for key, values in metrics.items()
            }
            for name, metrics in deltas.items()
        },
        "seeds": args.seeds,
        "queries": int(index.numel()),
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"real": real, "deltas": report["deltas"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
