from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from ComP.knowledge.filter_v2.features import state_features
from ComP.knowledge.filter_v2.metrics import (
    query_macro_auc,
    top1_regret,
    utility_spearman,
)


def action_regret(prediction, target, mask, tau=0.0):
    regrets = []
    for row in range(prediction.shape[0]):
        valid = np.where(mask[row])[0]
        if valid.size == 0:
            continue
        best = max(0.0, float(target[row, valid].max()))
        chosen = valid[int(np.argmax(prediction[row, valid]))]
        use = prediction[row, chosen] > tau
        regrets.append(best - (float(target[row, chosen]) if use else 0.0))
    return float(np.mean(regrets)) if regrets else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", required=True)
    parser.add_argument("--text-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(args.text_model, device="cpu")
    payload = torch.load(args.feedback, map_location="cpu", weights_only=False)
    records = payload["records"]
    task = payload.get("task", "regression")
    texts = sorted({t for r in records for t in r["candidate_texts"]})
    emb = {t: torch.from_numpy(encoder.encode([t])[0]).numpy() for t in texts}
    n = len(records)
    max_k = 5
    candidate = np.zeros((n, max_k, emb[next(iter(emb))].shape[0]), dtype=np.float32)
    mask = np.zeros((n, max_k), dtype=bool)
    target = np.zeros((n, max_k), dtype=np.float32)
    state_vectors = []
    dialogues = []
    card_ids = []
    for row, record in enumerate(records):
        k = len(record["candidate_ids"])
        for index in range(min(k, max_k)):
            candidate[row, index] = emb.get(record["candidate_texts"][index], 0.0)
            mask[row, index] = bool(record.get("candidate_mask", [True] * k)[index])
            target[row, index] = float(record["u_ce"][index])
        availability = torch.tensor(record["availability"], dtype=torch.float32).reshape(1, 3)
        target_mod = torch.nn.functional.one_hot(
            torch.tensor(record["target_modality"]), 3
        ).float().reshape(1, 3)
        empty = torch.as_tensor(record["empty_logits"]).float().reshape(1, -1)
        state_vectors.append(state_features(
            availability, target_mod,
            torch.tensor([record["missing_pattern"]]),
            torch.tensor([record["missing_rate"]]), empty, task,
        )[0].numpy())
        dialogues.append(record["dialogue_id"])
        card_ids.append(record["candidate_ids"])
    state = np.stack(state_vectors)
    unique_dialogues = sorted(set(dialogues))
    random.Random(66).shuffle(unique_dialogues)
    fold_map = {
        dialogue: index % args.folds
        for index, dialogue in enumerate(unique_dialogues)
    }
    folds = np.array([fold_map[d] for d in dialogues])
    results = {}
    # M0 random (100 seeds)
    random_regrets = []
    rng = random.Random(66)
    for _ in range(100):
        regrets = []
        for row in range(n):
            valid = np.where(mask[row])[0]
            if valid.size == 0:
                continue
            best = max(0.0, float(target[row, valid].max()))
            chosen = valid[rng.randrange(valid.size)]
            regrets.append(best - max(0.0, float(target[row, chosen])))
        random_regrets.append(float(np.mean(regrets)))
    results["M0_random"] = {
        "action_regret_mean": float(np.mean(random_regrets)),
        "action_regret_p05": float(np.quantile(random_regrets, 0.05)),
        "action_regret_p95": float(np.quantile(random_regrets, 0.95)),
    }
    # M2 state-only query-level
    pred_max = np.zeros(n)
    for fold in range(args.folds):
        train = folds != fold
        val = folds == fold
        model = Ridge(alpha=1.0)
        model.fit(state[train], target[train].max(axis=1))
        pred_max[val] = model.predict(state[val])
    labels = target.max(axis=1) > 0
    m2_regrets = []
    for row in range(n):
        valid = np.where(mask[row])[0]
        if valid.size == 0:
            continue
        best = max(0.0, float(target[row, valid].max()))
        use = pred_max[row] > 0 and bool(mask[row, 0])
        m2_regrets.append(best - (float(target[row, 0]) if use else 0.0))
    results["M2_state"] = {
        "use_skip_auc": float(roc_auc_score(labels, pred_max))
        if labels.any() and (~labels).any() else None,
        "action_regret": float(np.mean(m2_regrets)) if m2_regrets else None,
    }
    # M3 card-text-only
    pred_card = np.zeros((n, max_k), dtype=np.float32)
    for fold in range(args.folds):
        train = folds != fold
        val = folds == fold
        x = candidate[train].reshape(-1, candidate.shape[-1])
        y = target[train].reshape(-1)
        keep = mask[train].reshape(-1)
        model = Ridge(alpha=1.0)
        model.fit(x[keep], y[keep])
        pred_card[val] = model.predict(
            candidate[val].reshape(-1, candidate.shape[-1])
        ).reshape(val.sum(), max_k)
    results["M3_text"] = {
        "query_macro_auc": query_macro_auc(pred_card, target, mask),
        "spearman": utility_spearman(pred_card, target, mask),
        "within_top5_regret": top1_regret(pred_card, target, mask),
        "action_regret": action_regret(pred_card, target, mask),
    }
    # O1 card-id cross-fitted
    o1 = np.full((n, max_k), -1e9)
    global_mean = float(target[mask].mean())
    for fold in range(args.folds):
        train = folds != fold
        val = folds == fold
        stats = defaultdict(list)
        for row in np.where(train)[0]:
            for index in np.where(mask[row])[0]:
                stats[card_ids[row][index]].append(float(target[row, index]))
        means = {
            key: (float(np.mean(v)) if len(v) >= 20 else global_mean)
            for key, v in stats.items()
        }
        for row in np.where(val)[0]:
            for index in np.where(mask[row])[0]:
                o1[row, index] = means.get(card_ids[row][index], global_mean)
    results["O1_card_id"] = {
        "action_regret": action_regret(o1, target, mask),
        "within_top5_regret": top1_regret(o1, target, mask),
    }
    # O3 source-only: retrieval rank 0
    o3 = np.full((n, max_k), -1e9)
    o3[:, 0] = 0.0
    results["O3_source_rank0"] = {
        "action_regret": action_regret(o3, target, mask),
    }
    # O2 metadata cell: target+pattern+rank bucket (coarse)
    keys = []
    for row, record in enumerate(records):
        keys.append([
            f"{record['target_modality']}|{record['missing_pattern']}|{index}"
            for index in range(max_k)
        ])
    o2 = np.full((n, max_k), -1e9)
    for fold in range(args.folds):
        train = folds != fold
        val = folds == fold
        stats = defaultdict(list)
        for row in np.where(train)[0]:
            for index in np.where(mask[row])[0]:
                stats[keys[row][index]].append(float(target[row, index]))
        means = {key: float(np.mean(v)) for key, v in stats.items()}
        for row in np.where(val)[0]:
            for index in np.where(mask[row])[0]:
                o2[row, index] = means.get(keys[row][index], global_mean)
    results["O2_metadata_cell"] = {
        "action_regret": action_regret(o2, target, mask),
    }
    random_mean = results["M0_random"]["action_regret_mean"]
    m2 = results["M2_state"]
    m3 = results["M3_text"]
    o2_regret = results["O2_metadata_cell"]["action_regret"]
    if m2.get("use_skip_auc") is not None and m2["use_skip_auc"] >= 0.60 and m2["action_regret"] < random_mean:
        decision = "KEEP_SKIP_ONLY"
    elif (m3["query_macro_auc"] or 0) >= 0.56 or o2_regret < random_mean:
        decision = "ENTER_M4"
    else:
        decision = "CLOSE_FILTER"
    report = {
        "task": task,
        "queries": n,
        "decision": decision,
        "random": results["M0_random"],
        "results": results,
        "note": "Pilot-scale 400-query feedback; metadata features B1 not rebuilt.",
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
