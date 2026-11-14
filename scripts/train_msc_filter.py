from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.knowledge.msc.cache import load_state_cache, subset_masks_batch
from ComP.knowledge.msc.policy import MSCPolicy
from ComP.knowledge.msc.state import MSCStateEncoder


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-cache", required=True)
    parser.add_argument("--subset-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--no-set-scorer", action="store_true")
    parser.add_argument("--no-skip-head", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    state = load_state_cache(args.state_cache)
    subset_payload = torch.load(
        args.subset_cache, map_location="cpu", weights_only=False
    )
    if subset_payload.get("msc_format_version") != 1:
        raise ValueError("unsupported subset cache version")
    rows = subset_payload["query_rows"]
    by_query: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_query[row["query_id"]].append(row)
    query_ids = state["query_ids"]
    index_by_query = {qid: index for index, qid in enumerate(query_ids)}
    candidate_vectors = state["candidate_vectors"]
    candidate_mask = state["candidate_mask"]

    samples = []
    for qid, subset_rows in by_query.items():
        index = index_by_query.get(qid)
        if index is None:
            continue
        vector = candidate_vectors[index]          # [K,D]
        mask = candidate_mask[index]               # [K]
        top_k = mask.size(0)
        masks = []
        utilities = []
        by_value = {int(row["subset_mask"]): float(row["utility"]) for row in subset_rows}
        valid_positions = mask.nonzero(as_tuple=False).flatten().tolist()
        from ComP.knowledge.msc.subsets import nonempty_subset_masks

        for value in nonempty_subset_masks(len(valid_positions)):
            expanded = 0
            for local in range(len(valid_positions)):
                if (value >> local) & 1:
                    expanded |= 1 << valid_positions[local]
            masks.append(expanded)
            utilities.append(by_value.get(expanded, float("-inf")))
        mask_tensor = torch.zeros(len(masks), top_k, dtype=torch.bool)
        for slot, value in enumerate(masks):
            for rank in range(top_k):
                if (value >> rank) & 1:
                    mask_tensor[slot, rank] = True
        utility_tensor = torch.tensor(utilities, dtype=torch.float32)
        utility_tensor[~torch.isfinite(utility_tensor)] = float("-inf")
        samples.append(
            {
                "index": index,
                "sample_id": subset_rows[0]["sample_id"],
                "subset_masks": mask_tensor,
                "subset_utility": utility_tensor,
                "skip_label": float((utility_tensor.max() <= 0.0).item()),
            }
        )
    if not samples:
        raise SystemExit("no subset samples available")
    rng = random.Random(args.seed)
    rng.shuffle(samples)
    split_at = int(len(samples) * (1.0 - args.val_frac))
    train_samples = samples[: max(1, split_at)]
    val_samples = samples[max(1, split_at):]

    state_encoder = MSCStateEncoder(
        modality_count=3, hidden_dim=args.state_dim, pattern_vocab_size=16
    )
    policy = MSCPolicy(
        candidate_vectors.size(-1),
        args.state_dim,
        candidate_vectors.size(1),
        hidden_dim=args.hidden_dim,
        use_set_scorer=not args.no_set_scorer,
        use_skip_head=not args.no_skip_head,
    )
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    state_encoder.to(device)
    policy.to(device)
    optimizer = torch.optim.AdamW(
        list(state_encoder.parameters()) + list(policy.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    set_criterion = nn.MSELoss(reduction="none")
    skip_criterion = nn.BCEWithLogitsLoss()

    def pack(batch):
        indices = torch.tensor([item["index"] for item in batch])
        obs = state["observed_mask"][indices]
        target = state["target_modality"][indices]
        pattern = state["missing_pattern"][indices]
        rate = state["missing_rate"][indices]
        max_s = max(item["subset_masks"].size(0) for item in batch)
        top_k = candidate_vectors.size(1)
        masks = torch.zeros(len(batch), max_s, top_k, dtype=torch.bool)
        utilities = torch.full((len(batch), max_s), float("-inf"))
        skip = torch.zeros(len(batch))
        for row, item in enumerate(batch):
            s = item["subset_masks"].size(0)
            masks[row, :s] = item["subset_masks"]
            utilities[row, :s] = item["subset_utility"]
            skip[row] = item["skip_label"]
        return (
            obs.to(device),
            target.to(device),
            pattern.to(device),
            rate.to(device),
            candidate_vectors[indices].to(device),
            candidate_mask[indices].to(device),
            masks.to(device),
            utilities.to(device),
            skip.to(device),
        )

    def evaluate(batch):
        obs, target, pattern, rate, cv, cm, masks, utilities, skip = pack(batch)
        state_vector = state_encoder(obs, target, pattern, rate)
        out = policy(
            cv, cm, state_vector, masks,
            source_confidence=None,
            conflict_score=None,
        )
        valid = torch.isfinite(utilities)
        set_loss = set_criterion(out["subset_scores"], utilities)
        set_loss = set_loss[valid].mean() if valid.any() else torch.zeros(())
        skip_loss = skip_criterion(out["skip_logit"], skip)
        return float(set_loss.item() + skip_loss.item()), state_encoder, policy

    best_val = float("inf")
    stale = 0
    history = []
    for epoch in range(args.epochs):
        state_encoder.train()
        policy.train()
        total = 0.0
        count = 0
        order = list(range(len(train_samples)))
        random.Random(args.seed + epoch).shuffle(order)
        for start in range(0, len(order), args.batch_size):
            batch = [train_samples[index] for index in order[start : start + args.batch_size]]
            obs, target, pattern, rate, cv, cm, masks, utilities, skip = pack(batch)
            state_vector = state_encoder(obs, target, pattern, rate)
            out = policy(
                cv, cm, state_vector, masks,
                source_confidence=None, conflict_score=None,
            )
            valid = torch.isfinite(utilities)
            set_loss = set_criterion(out["subset_scores"], utilities)
            set_loss = set_loss[valid].mean() if valid.any() else torch.zeros(())
            skip_loss = skip_criterion(out["skip_logit"], skip)
            loss = set_loss + skip_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().item())
            count += 1
        state_encoder.eval()
        policy.eval()
        with torch.no_grad():
            val_loss, _, _ = evaluate(val_samples)
        history.append({"epoch": epoch, "train_loss": total / max(1, count), "val_loss": val_loss})
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        if val_loss < best_val:
            best_val = val_loss
            stale = 0
            output = Path(args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": 1,
                    "stage": "l1",
                    "state_dict": {
                        **{"state_encoder." + key: value for key, value in state_encoder.state_dict().items()},
                        **{"policy." + key: value for key, value in policy.state_dict().items()},
                    },
                    "candidate_dim": int(candidate_vectors.size(-1)),
                    "state_dim": args.state_dim,
                    "top_k": int(candidate_vectors.size(1)),
                    "hidden_dim": args.hidden_dim,
                    "use_set_scorer": not args.no_set_scorer,
                    "use_skip_head": not args.no_skip_head,
                    "state_cache_sha256": sha256_file(args.state_cache),
                    "subset_cache_sha256": sha256_file(args.subset_cache),
                    "test_used": False,
                    "validation_used_for_training": False,
                },
                output / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    summary = {
        "best_val_loss": best_val,
        "epochs_run": len(history),
        "history": history,
        "train_queries": len(train_samples),
        "val_queries": len(val_samples),
    }
    (Path(args.output_dir) / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
