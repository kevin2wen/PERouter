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
from ComP.knowledge.filter_v2.metrics import query_macro_auc, top1_regret, utility_spearman


def regret(best, utility):
    return best - utility


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", required=True)
    parser.add_argument("--card-bank", required=True)
    parser.add_argument("--text-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-seeds", type=int, default=100)
    args = parser.parse_args()
    from sentence_transformers import SentenceTransformer

    payload = torch.load(args.feedback, map_location="cpu", weights_only=False)
    task = payload.get("task", "regression")
    records = payload["records"]
    cards = {}
    for line in Path(args.card_bank).open(encoding="utf-8"):
        card = json.loads(line)
        cards[str(card["knowledge_id"])] = card
    encoder = SentenceTransformer(args.text_model, device="cpu")
    texts = sorted({t for r in records for t in r["candidate_texts"]})
    emb = {t: torch.from_numpy(encoder.encode([t])[0]).numpy() for t in texts}
    n, max_k = len(records), 5
    candidate = np.zeros((n, max_k, 384), dtype=np.float32)
    mask = np.zeros((n, max_k), dtype=bool)
    target = np.zeros((n, max_k), dtype=np.float32)
    state = []
    dialogues, ids = [], []
    meta_features = []
    relations = sorted({str(c.get("relation_type") or "") for c in cards.values()})
    affects = sorted({str(c.get("affect_prior") or "") for c in cards.values()})
    for row, record in enumerate(records):
        k = len(record["candidate_ids"])
        ids.append([str(v) for v in record["candidate_ids"]])
        for index in range(min(k, max_k)):
            candidate[row, index] = emb.get(record["candidate_texts"][index], 0.0)
            mask[row, index] = bool(record.get("candidate_mask", [True] * k)[index])
            target[row, index] = float(record.get("u_reg", record.get("u_ce"))[index])
        card = cards.get(ids[row][0], {})
        feature = [0.0] * (len(relations) + len(affects) + max_k)
        relation = str(card.get("relation_type") or "")
        affect = str(card.get("affect_prior") or "")
        if relation in relations:
            feature[relations.index(relation)] = 1.0
        if affect in affects:
            feature[len(relations) + affects.index(affect)] = 1.0
        meta_features.append(feature)
        availability = torch.tensor(record["availability"], dtype=torch.float32).reshape(1, 3)
        target_mod = torch.nn.functional.one_hot(
            torch.tensor(record["target_modality"]), 3
        ).float().reshape(1, 3)
        empty = torch.as_tensor(record["empty_logits"]).float().reshape(1, -1)
        state.append(state_features(
            availability, target_mod, torch.tensor([record["missing_pattern"]]),
            torch.tensor([record["missing_rate"]]), empty, task,
        )[0].numpy())
        dialogues.append(record["dialogue_id"])
    meta_features = np.asarray(meta_features, dtype=np.float32)
    state = np.stack(state)
    unique = sorted(set(dialogues))
    random.Random(66).shuffle(unique)
    fold_of = {d: i % args.folds for i, d in enumerate(unique)}
    folds = np.array([fold_of[d] for d in dialogues])
    best = np.maximum(0.0, target.max(axis=1))
    results = {}

    def summarize(values):
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "p05": float(np.quantile(values, 0.05)),
            "p95": float(np.quantile(values, 0.95)),
        }

    # correct baselines
    results["B_skip"] = summarize([best.mean()])
    top1_utility = np.where(mask[:, 0], target[:, 0], 0.0)
    results["B_top1"] = summarize([np.mean(best - top1_utility)])
    rng = random.Random(66)
    rand_card, rand_action = [], []
    for _ in range(args.random_seeds):
        card_regrets, action_regrets = [], []
        for row in range(n):
            valid = np.where(mask[row])[0]
            if valid.size == 0:
                continue
            chosen = valid[rng.randrange(valid.size)]
            card_regrets.append(best[row] - float(target[row, chosen]))
            options = [0.0] + [float(target[row, i]) for i in valid]
            action_regrets.append(best[row] - options[rng.randrange(len(options))])
        rand_card.append(float(np.mean(card_regrets)))
        rand_action.append(float(np.mean(action_regrets)))
    results["B_rand_card"] = summarize(rand_card)
    results["B_rand_action"] = summarize(rand_action)
    # M1 metadata ridge
    pred_meta = np.zeros(n)
    for fold in range(args.folds):
        tr, va = folds != fold, folds == fold
        model = Ridge(alpha=1.0).fit(meta_features[tr], best[tr])
        pred_meta[va] = model.predict(meta_features[va])
    results["M1_metadata"] = {
        "use_skip_auc": float(roc_auc_score(best > 0, pred_meta)),
        "action_regret": float(np.mean(best - np.where(pred_meta > 0, top1_utility, 0.0))),
    }
    # M2 state with calibrated threshold
    pred_max = np.zeros(n)
    m2_regret = []
    for fold in range(args.folds):
        tr, va = folds != fold, folds == fold
        model = Ridge(alpha=1.0).fit(state[tr], best[tr])
        train_pred = model.predict(state[tr])
        taus = np.quantile(train_pred, np.linspace(0.05, 0.95, 19))
        best_tau, best_regret = 0.0, float("inf")
        for tau in taus:
            regret_train = np.mean(
                best[tr] - np.where(train_pred > tau, top1_utility[tr], 0.0)
            )
            if regret_train < best_regret:
                best_tau, best_regret = float(tau), float(regret_train)
        val_pred = model.predict(state[va])
        pred_max[va] = val_pred
        m2_regret.append(
            float(np.mean(best[va] - np.where(val_pred > best_tau, top1_utility[va], 0.0)))
        )
    results["M2_state_calibrated"] = {
        "use_skip_auc": float(roc_auc_score(best > 0, pred_max)),
        "action_regret_fold_mean": float(np.mean(m2_regret)),
    }
    # M3 card text ridge
    pred_card = np.zeros((n, max_k), dtype=np.float32)
    for fold in range(args.folds):
        tr, va = folds != fold, folds == fold
        x = candidate[tr].reshape(-1, 384)
        y = target[tr].reshape(-1)
        keep = mask[tr].reshape(-1)
        model = Ridge(alpha=1.0).fit(x[keep], y[keep])
        pred_card[va] = model.predict(candidate[va].reshape(-1, 384)).reshape(
            va.sum(), max_k
        )
    results["M3_text"] = {
        "query_macro_auc": query_macro_auc(pred_card, target, mask),
        "spearman": utility_spearman(pred_card, target, mask),
        "within_top5_regret": top1_regret(pred_card, target, mask),
        "action_regret": float(
            np.mean(best - np.where(
                pred_card.max(axis=1) > 0,
                np.take_along_axis(
                    target, pred_card.argmax(axis=1)[:, None], axis=1
                ).squeeze(1),
                0.0,
            ))
        ),
    }
    # O3a top1 always; O3b source policy by fold
    results["O3a_top1"] = summarize([np.mean(best - top1_utility)])
    o3b = []
    for fold in range(args.folds):
        tr, va = folds != fold, folds == fold
        mean_top1 = float(np.mean(target[tr, 0][mask[tr, 0]]))
        o3b.append(float(np.mean(best[va] - (top1_utility[va] if mean_top1 > 0 else 0.0))))
    results["O3b_source"] = summarize(o3b)
    # O2 full cells with relation/affect/rank; fallback to global
    keys = []
    for row, record in enumerate(records):
        card = cards.get(ids[row][0], {})
        relation = str(card.get("relation_type") or "")
        affect = str(card.get("affect_prior") or "")
        keys.append([
            f"{record['target_modality']}|{record['missing_pattern']}|{index}|{relation}|{affect}"
            for index in range(max_k)
        ])
    o2 = np.full((n, max_k), -1e9)
    global_mean = float(target[mask].mean())
    for fold in range(args.folds):
        tr, va = folds != fold, folds == fold
        stats = defaultdict(list)
        for row in np.where(tr)[0]:
            for index in np.where(mask[row])[0]:
                stats[keys[row][index]].append(float(target[row, index]))
        means = {
            key: float(np.mean(v)) if len(v) >= 20 else global_mean
            for key, v in stats.items()
        }
        for row in np.where(va)[0]:
            for index in np.where(mask[row])[0]:
                o2[row, index] = means.get(keys[row][index], global_mean)
    results["O2_metadata_cell"] = {
        "action_regret": float(
            np.mean(best - np.where(
                o2.max(axis=1) > 0,
                np.take_along_axis(target, o2.argmax(axis=1)[:, None], axis=1).squeeze(1),
                0.0,
            ))
        )
    }
    random_card_p05 = results["B_rand_card"]["p05"]
    decision = "CLOSE_FILTER"
    if (results["M1_metadata"]["use_skip_auc"] >= 0.56
            or (results["M3_text"]["query_macro_auc"] or 0) >= 0.56
            or results["O2_metadata_cell"]["action_regret"] < random_card_p05):
        decision = "ENTER_M4"
    elif results["M2_state_calibrated"]["use_skip_auc"] >= 0.58:
        decision = "KEEP_SKIP_ONLY"
    report = {
        "task": task,
        "queries": n,
        "decision": decision,
        "baselines": {
            key: results[key]
            for key in ("B_skip", "B_top1", "B_rand_card", "B_rand_action")
        },
        "methods": {
            key: results[key]
            for key in ("M1_metadata", "M2_state_calibrated", "M3_text",
                        "O2_metadata_cell", "O3a_top1", "O3b_source")
        },
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
