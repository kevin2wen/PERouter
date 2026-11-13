from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ComP.per.evidence_reranker import EvidenceReranker
from ComP.per.reranker_metrics import (
    action_metrics,
    dialogue_folds,
    listwise_loss,
    pairwise_loss,
    utility_regression_loss,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-cache", required=True)
    p.add_argument(
        "--source",
        choices=("feature", "text", "knowledge"),
        required=True,
    )
    p.add_argument("--objective", choices=("listwise", "pairwise", "utility"), required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--progress-every", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--output-dir", required=True)
    return p


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare(cache):
    utility = cache["utility"].float()
    skip_utility = cache["skip_utility"].float()
    delta = utility - skip_utility.unsqueeze(-1)
    if "candidate_mask" in cache:
        candidate_mask = cache["candidate_mask"].bool()
    else:
        candidate_mask = torch.tensor(
            [[bool(value) for value in record["candidate_ids"]] for record in cache["records"]],
            dtype=torch.bool,
        )
    best_gain, best_rank = delta.masked_fill(
        ~candidate_mask, torch.finfo(delta.dtype).min
    ).max(dim=-1)
    min_gain = float(cache.get("min_helpful_gain", 0.0))
    if "min_helpful_gain" not in cache:
        min_gain = 0.0
    use = best_gain >= min_gain
    target_action = torch.where(use, best_rank + 1, torch.zeros_like(best_rank))
    delta_with_skip = torch.cat([torch.zeros_like(delta[:, :1]), delta], dim=1)
    pattern = torch.tensor(
        [[int(char) for char in value] for value in cache["missing_patterns"]],
        dtype=torch.long,
    )
    return {
        "candidate_vectors": cache["candidate_vectors"].float(),
        "prompt_vectors": cache["prompt_vectors"].float(),
        "source_confidence": cache["source_confidence"].float(),
        "target_modalities": cache["target_modalities"].long(),
        "missing_patterns": pattern,
        "candidate_mask": candidate_mask,
        "target_action": target_action.long(),
        "delta_utility": delta_with_skip,
        "sample_ids": list(cache["records_sample_ids"]) if "records_sample_ids" in cache else [r["sample_id"] for r in cache["records"]],
        "candidate_dim": cache["candidate_vectors"].size(-1),
        "prompt_dim": cache["prompt_vectors"].size(-1),
        "top_k": cache["candidate_vectors"].size(1),
    }


def select_batch(data, indices, device):
    indices = torch.as_tensor(indices, dtype=torch.long)
    return {
        key: (value[indices].to(device) if isinstance(value, torch.Tensor) else [value[i] for i in indices.tolist()])
        for key, value in data.items()
        if key not in {"candidate_dim", "prompt_dim", "top_k", "sample_ids"}
    }


def objective_loss(logits, batch, objective):
    if objective == "listwise":
        return listwise_loss(logits, batch["target_action"])
    if objective == "pairwise":
        return pairwise_loss(logits, batch["target_action"], batch["candidate_mask"])
    return utility_regression_loss(logits, batch["delta_utility"], batch["candidate_mask"])


def main() -> int:
    cli = parser().parse_args()
    if cli.epochs <= 0 or cli.folds < 2 or cli.batch_size <= 0:
        raise ValueError("invalid training configuration")
    set_seed(cli.seed)
    cache = torch.load(cli.candidate_cache, map_location="cpu", weights_only=False)
    if cache.get("split") != "train":
        raise ValueError("candidate cache must use train split")
    data = prepare(cache)
    if data["candidate_dim"] != data["prompt_dim"]:
        raise ValueError("current reranker expects equal candidate and prompt dimensions")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_assignments = dialogue_folds(data["sample_ids"], cli.folds, cli.seed)
    output = Path(cli.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    fold_summaries = []
    all_indices = np.arange(len(data["sample_ids"]))
    oof_logits = torch.full(
        (len(data["sample_ids"]), data["top_k"] + 1), float("nan")
    )
    oof_fold_ids = torch.full((len(data["sample_ids"]),), -1, dtype=torch.long)
    for fold, val_indices in enumerate(fold_assignments):
        train_indices = np.setdiff1d(all_indices, np.asarray(val_indices), assume_unique=False)
        if len(train_indices) == 0 or len(val_indices) == 0:
            raise ValueError(f"empty fold {fold}")
        model = EvidenceReranker(
            data["candidate_dim"], data["prompt_dim"], data["top_k"],
            hidden_dim=cli.hidden_dim,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cli.learning_rate, weight_decay=cli.weight_decay
        )
        best_state = copy.deepcopy(model.state_dict())
        best_val = float("inf")
        stale = 0
        fold_history = []
        for epoch in range(cli.epochs):
            model.train()
            shuffled = np.random.permutation(train_indices)
            training_losses = []
            for start in range(0, len(shuffled), cli.batch_size):
                batch = select_batch(data, shuffled[start:start + cli.batch_size].tolist(), device)
                logits = model(
                    batch["candidate_vectors"],
                    batch["prompt_vectors"],
                    batch["source_confidence"],
                    batch["target_modalities"],
                    batch["missing_patterns"],
                    batch["candidate_mask"],
                )
                loss = objective_loss(logits, batch, cli.objective)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite reranker loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                training_losses.append(float(loss.detach().item()))
            model.eval()
            with torch.no_grad():
                val_batch = select_batch(data, val_indices, device)
                val_logits = model(
                    val_batch["candidate_vectors"], val_batch["prompt_vectors"],
                    val_batch["source_confidence"], val_batch["target_modalities"],
                    val_batch["missing_patterns"], val_batch["candidate_mask"],
                )
                val_loss = float(objective_loss(val_logits, val_batch, cli.objective).item())
            record = {
                "fold": fold,
                "epoch": epoch,
                "train_loss": float(np.mean(training_losses)),
                "oof_loss": val_loss,
            }
            fold_history.append(record)
            history.append(record)
            progress_every = max(1, int(getattr(cli, "progress_every", 20)))
            if epoch == 0 or (epoch + 1) % progress_every == 0:
                print(json.dumps(record, sort_keys=True), flush=True)
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= cli.patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_batch = select_batch(data, val_indices, device)
            logits = model(
                val_batch["candidate_vectors"], val_batch["prompt_vectors"],
                val_batch["source_confidence"], val_batch["target_modalities"],
                val_batch["missing_patterns"], val_batch["candidate_mask"],
            )
            metrics = action_metrics(
                logits, val_batch["delta_utility"], val_batch["candidate_mask"],
                val_batch["target_action"],
            )
            oof_logits[val_indices] = logits.detach().cpu()
            oof_fold_ids[val_indices] = fold
        fold_path = output / f"reranker_{cli.objective}_fold{fold}.pt"
        torch.save({
            "schema_version": 1,
            "source": cli.source,
            "objective": cli.objective,
            "fold": fold,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "candidate_dim": data["candidate_dim"],
            "prompt_dim": data["prompt_dim"],
            "top_k": data["top_k"],
            "hidden_dim": cli.hidden_dim,
            "best_oof_loss": best_val,
            "metrics": metrics,
            "cache_key": cache.get("cache_key"),
        "cache_sha256": sha256_file(cli.candidate_cache),
            "student_checkpoint": cache.get("student_checkpoint"),
            "validation_used_for_training": False,
            "test_used": False,
        }, fold_path)
        fold_summaries.append({"fold": fold, "path": str(fold_path), **metrics})
        print(
            json.dumps({
                "fold_done": fold,
                "best_oof_loss": best_val,
                "action_agreement": metrics["top1_action_agreement"],
                "regret": metrics["regret_mean"],
                "use_evidence_rate": metrics["use_evidence_rate"],
            }, sort_keys=True),
            flush=True,
        )
    if torch.isnan(oof_logits).any() or (oof_fold_ids < 0).any():
        raise RuntimeError("incomplete OOF predictions")
    global_oof_metrics = action_metrics(
        oof_logits,
        data["delta_utility"],
        data["candidate_mask"],
        data["target_action"],
    )
    torch.save({
        "logits": oof_logits,
        "fold_ids": oof_fold_ids,
        "target_action": data["target_action"],
        "candidate_mask": data["candidate_mask"],
        "delta_utility": data["delta_utility"],
        "sample_ids": data["sample_ids"],
    }, output / "oof_predictions.pt")
    payload = {
        "source": cli.source,
        "objective": cli.objective,
        "folds": cli.folds,
        "epochs_requested": cli.epochs,
        "seed": cli.seed,
        "cache_key": cache.get("cache_key"),
        "cache_sha256_source": cache.get("schema_version"),
        "simplified_crossfit": True,
        "student_seen_all_train": True,
        "validation_used_for_training": False,
        "test_used": False,
        "fold_summaries": fold_summaries,
        "global_oof_metrics": global_oof_metrics,
        "mean_action_agreement": float(np.mean([x["top1_action_agreement"] for x in fold_summaries])),
        "mean_regret": float(np.mean([x["regret_mean"] for x in fold_summaries])),
        "mean_use_evidence_rate": float(np.mean([x["use_evidence_rate"] for x in fold_summaries])),
    }
    (output / "reranker_training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (output / "reranker_training_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
