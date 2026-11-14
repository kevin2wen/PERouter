from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.knowledge.msc.single_eu import MSCSingleEU


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(state_dict, prefix: str) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        if not key.startswith(prefix):
            continue
        digest.update(key.encode())
        digest.update(state_dict[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=("s0", "s1", "s2"), default="s1")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--pairwise-weight", type=float, default=0.5)
    parser.add_argument("--skip-weight", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    cache = torch.load(args.single_cache, map_location="cpu", weights_only=False)
    sample_ids = [row["sample_id"] for row in cache.get("records", [])]
    if not sample_ids:
        raise SystemExit("single cache has no records")
    dialogues = sorted(set(sample_ids))
    rng = random.Random(args.seed)
    rng.shuffle(dialogues)
    split_at = max(1, int(len(dialogues) * (1 - args.val_frac)))
    train_ids = set(dialogues[:split_at])
    val_ids = set(dialogues[split_at:]) or train_ids
    train_index = [i for i, sid in enumerate(sample_ids) if sid in train_ids]
    val_index = [i for i, sid in enumerate(sample_ids) if sid in val_ids]
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    variant = args.variant
    model = MSCSingleEU(
        cache["candidate_vectors"].size(-1),
        cache["prompt_vectors"].size(-1),
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        use_semantic=variant in {"s1", "s2"},
        use_cross_attention=variant == "s2",
        use_skip_head=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    tensors = {
        "availability": cache["observed_mask"].float(),
        "target": cache["target_modality"].float(),
        "pattern": cache["missing_pattern"].long(),
        "rate": cache["missing_rate"].float(),
        "prompt": cache["prompt_vectors"].float(),
        "candidates": cache["candidate_vectors"].float(),
        "mask": cache["candidate_mask"].bool(),
        "confidence": cache["source_confidence"].float(),
        "conflict": (
            cache["conflict_score"].float()
            if cache.get("conflict_score") is not None
            else torch.zeros_like(cache["source_confidence"])
        ),
        "utility": cache["single_utility"].float(),
        "skip": cache["skip_label"].float(),
    }

    def batch_tensors(indices):
        return {
            key: value[indices].to(device) for key, value in tensors.items()
        }

    def compute_loss(batch):
        output = model(
            batch["availability"], batch["target"], batch["pattern"], batch["rate"],
            batch["prompt"], batch["candidates"], batch["mask"],
            batch["confidence"], batch["conflict"],
        )
        utility = output["utility"]
        valid = batch["mask"].bool()
        regression = F.smooth_l1_loss(
            utility[valid], batch["utility"][valid]
        ) if valid.any() else torch.zeros((), device=device)
        pairwise_terms = []
        for row in range(utility.size(0)):
            ranks = valid[row].nonzero(as_tuple=False).flatten()
            if ranks.numel() < 2:
                continue
            diffs = batch["utility"][row, ranks].unsqueeze(1) - batch["utility"][row, ranks].unsqueeze(0)
            preds = utility[row, ranks].unsqueeze(1) - utility[row, ranks].unsqueeze(0)
            mask = diffs > 1e-6
            if mask.any():
                pairwise_terms.append(
                    torch.clamp(0.1 - preds[mask], min=0.0).mean()
                )
        pairwise = (
            torch.stack(pairwise_terms).mean()
            if pairwise_terms
            else torch.zeros((), device=device)
        )
        skip = F.binary_cross_entropy_with_logits(
            output["skip_logit"], batch["skip"]
        )
        return regression + args.pairwise_weight * pairwise + args.skip_weight * skip

    best_val, stale, history = float("inf"), 0, []
    for epoch in range(args.epochs):
        model.train()
        order = train_index[:]
        random.Random(args.seed + epoch).shuffle(order)
        total, count = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            batch = batch_tensors(order[start : start + args.batch_size])
            loss = compute_loss(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            count += 1
        model.eval()
        with torch.no_grad():
            val_loss = float(compute_loss(batch_tensors(val_index)))
        history.append({"epoch": epoch, "train_loss": total / max(1, count), "val_loss": val_loss})
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        if val_loss < best_val:
            best_val, stale = val_loss, 0
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            state = model.state_dict()
            torch.save(
                {
                    "schema_version": 1,
                    "stage": variant,
                    "state_dict": state,
                    "candidate_dim": int(cache["candidate_vectors"].size(-1)),
                    "prompt_dim": int(cache["prompt_vectors"].size(-1)),
                    "state_dim": args.state_dim,
                    "hidden_dim": args.hidden_dim,
                    "use_semantic": variant in {"s1", "s2"},
                    "use_cross_attention": variant == "s2",
                    "use_skip_head": True,
                    "candidate_feature_kind": (
                        "semantic" if cache.get("semantic_model_path") else "projection"
                    ),
                    "semantic_model_path": cache.get("semantic_model_path"),
                    "utility_scorer_hash": state_hash(state, "scorer."),
                    "state_encoder_hash": state_hash(state, "state_encoder."),
                    "single_cache_sha256": sha256_file(args.single_cache),
                    "validation_used_for_training": False,
                    "test_used": False,
                },
                output_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    summary = {
        "stage": variant,
        "best_val_loss": best_val,
        "epochs_run": len(history),
        "train_queries": len(train_index),
        "val_queries": len(val_index),
        "history": history,
    }
    (Path(args.output_dir) / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
