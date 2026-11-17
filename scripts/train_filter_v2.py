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

from ComP.knowledge.filter_v2.action import FilterV2ActionSelector
from ComP.knowledge.filter_v2.calibration import fit_threshold_f1
from ComP.knowledge.filter_v2.features import state_features
from ComP.knowledge.filter_v2.losses import filter_v2_loss
from ComP.knowledge.filter_v2.metrics import (
    action_regret,
    helpfulness_auc,
    query_macro_auc,
    top1_regret,
    utility_spearman,
)
from ComP.knowledge.filter_v2.model import FilterV2Model


def load_feedback(path, encoder, task):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = payload["records"]
    texts = sorted({text for r in records for text in r["candidate_texts"]})
    embeddings = {
        text: torch.from_numpy(encoder.encode([text])[0]) for text in texts
    }
    return payload, embeddings


def build_tensors(records, embeddings, task, max_k=5):
    state_vectors, candidates, masks, u_ce, u_corr, u_mae, dialogues = [], [], [], [], [], [], []
    for record in records:
        k = len(record["candidate_ids"])
        cand = torch.zeros(max_k, 384)
        mask = torch.zeros(max_k, dtype=torch.bool)
        ce = torch.zeros(max_k)
        corr = torch.zeros(max_k)
        mae = torch.zeros(max_k)
        for index, text in enumerate(record["candidate_texts"][:max_k]):
            cand[index] = embeddings.get(text, torch.zeros(384))
            mask[index] = bool(record.get("candidate_mask", [True] * k)[index])
            ce[index] = float(record["u_ce"][index])
            corr[index] = float(record["u_corr"][index])
            mae[index] = float(record["u_mae"][index])
        availability = torch.tensor(record["availability"], dtype=torch.float32)
        target = torch.nn.functional.one_hot(
            torch.tensor(record["target_modality"]), 3
        ).float()
        empty_logits = torch.as_tensor(record["empty_logits"]).reshape(1, -1)
        state = state_features(
            availability.reshape(1, 3), target.reshape(1, 3),
            torch.tensor([record["missing_pattern"]]),
            torch.tensor([record["missing_rate"]]),
            empty_logits, task,
        )[0]
        state_vectors.append(state)
        candidates.append(cand)
        masks.append(mask)
        u_ce.append(ce)
        u_corr.append(corr)
        u_mae.append(mae)
        dialogues.append(record["dialogue_id"])
    return {
        "state": torch.stack(state_vectors),
        "candidates": torch.stack(candidates),
        "mask": torch.stack(masks),
        "u_ce": torch.stack(u_ce),
        "u_corr": torch.stack(u_corr),
        "u_mae": torch.stack(u_mae),
        "dialogues": dialogues,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--eval-cache", required=True)
    parser.add_argument("--text-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(args.text_model, device="cpu")
    encoder.max_seq_length = 192
    train_payload, train_embeddings = load_feedback(args.train_cache, encoder, "classification")
    eval_payload, eval_embeddings = load_feedback(args.eval_cache, encoder, "classification")
    task = train_payload.get("task", "classification")
    train_data = build_tensors(train_payload["records"], train_embeddings, task)
    eval_data = build_tensors(eval_payload["records"], eval_embeddings, task)
    dialogues = sorted(set(train_data["dialogues"]))
    rng = random.Random(args.seed)
    rng.shuffle(dialogues)
    split_at = max(1, int(len(dialogues) * (1 - args.val_frac)))
    val_ids = set(dialogues[split_at:])
    train_ids = set(dialogues[:split_at])
    train_index = [i for i, d in enumerate(train_data["dialogues"]) if d in train_ids]
    val_index = [i for i, d in enumerate(train_data["dialogues"]) if d in val_ids]
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = FilterV2Model(
        384, train_data["state"].size(-1), hidden_dim=args.hidden_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    tensors = {
        key: value.to(device)
        for key, value in train_data.items()
        if isinstance(value, torch.Tensor)
    }
    eval_tensors = {
        key: value.to(device)
        for key, value in eval_data.items()
        if isinstance(value, torch.Tensor)
    }

    def batch(indices):
        return {
            **{key: value[indices] for key, value in tensors.items()},
            "confidence": torch.ones(len(indices), 5, device=device),
        }

    best_val, stale, history = float("inf"), 0, []
    for epoch in range(args.epochs):
        model.train()
        order = train_index[:]
        random.Random(args.seed + epoch).shuffle(order)
        total, count = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            data = batch(order[start:start + args.batch_size])
            output = model(
                data["candidates"], data["mask"], data["state"],
                data["confidence"],
            )
            losses = filter_v2_loss(
                output, data["u_ce"], data["u_corr"], data["u_mae"], data["mask"]
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(losses["loss"].detach())
            count += 1
        model.eval()
        with torch.no_grad():
            data = batch(val_index)
            output = model(
                data["candidates"], data["mask"], data["state"], data["confidence"]
            )
            val_loss = float(
                filter_v2_loss(
                    output, data["u_ce"], data["u_corr"], data["u_mae"], data["mask"]
                )["loss"]
            )
        history.append({"epoch": epoch, "train_loss": total / max(1, count), "val_loss": val_loss})
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        if val_loss < best_val:
            best_val, stale = val_loss, 0
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": 1,
                    "state_dict": model.state_dict(),
                    "candidate_dim": 384,
                    "state_dim": int(train_data["state"].size(-1)),
                    "hidden_dim": args.hidden_dim,
                    "test_used": False,
                },
                output_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(Path(args.output_dir) / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        val_out = model(
            tensors["candidates"][val_index],
            tensors["mask"][val_index],
            tensors["state"][val_index],
            torch.ones(len(val_index), 5, device=device),
        )
        max_utility = val_out["utility"].masked_fill(
            ~tensors["mask"][val_index], -1e9
        ).max(dim=-1).values.cpu().numpy()
        use_labels = (
            tensors["u_ce"][val_index].masked_fill(
                ~tensors["mask"][val_index], -1e9
            ).max(dim=-1).values > 0
        ).cpu().numpy()
    calibration = fit_threshold_f1(max_utility, use_labels)
    selector = FilterV2ActionSelector(tau=calibration["tau"])
    with torch.no_grad():
        eval_out = model(
            eval_tensors["candidates"], eval_tensors["mask"],
            eval_tensors["state"], torch.ones(eval_tensors["mask"].size(0), 5, device=device),
        )
        action = selector.select(eval_out["utility"], eval_tensors["mask"])
    prediction = eval_out["utility"].cpu().numpy()
    target = eval_tensors["u_ce"].cpu().numpy()
    mask = eval_tensors["mask"].cpu().numpy()
    metrics = {
        "queries": int(mask.shape[0]),
        "pooled_auc": helpfulness_auc(prediction, target, mask),
        "query_macro_auc": query_macro_auc(prediction, target, mask),
        "utility_spearman": utility_spearman(prediction, target, mask),
        "within_top5_regret": top1_regret(prediction, target, mask),
        "action_regret": action_regret(
            prediction, target, mask, tau=calibration["tau"]
        ),
        "utility_mae": float(
            np.abs(prediction[mask] - target[mask]).mean()
        ),
        "use_rate": float(action["use"].float().mean()),
        "skip_rate": float(action["skip"].float().mean()),
    }
    output = Path(args.output_dir)
    (output / "calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n"
    )
    (output / "eval_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    (output / "training_summary.json").write_text(
        json.dumps({"best_val_loss": best_val, "history": history}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"calibration": calibration, "metrics": metrics}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
