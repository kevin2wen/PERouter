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

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from ComP.knowledge.filter_v2.metrics import query_macro_auc, top1_regret, utility_spearman


def bootstrap_cluster(values_by_cluster, n_boot=2000, seed=66):
    clusters = [np.asarray(v, dtype=np.float64) for v in values_by_cluster.values()]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        picks = rng.integers(0, len(clusters), size=len(clusters))
        draws.append(np.mean([clusters[i].mean() for i in picks]))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(np.mean([c.mean() for c in clusters])), float(lo), float(hi)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", required=True)
    parser.add_argument("--card-bank", required=True)
    parser.add_argument("--text-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-seeds", type=int, default=100)
    args = parser.parse_args()
    from sentence_transformers import SentenceTransformer

    payload = torch.load(args.feedback, map_location="cpu", weights_only=False)
    records = payload["records"]
    cards = {}
    for line in Path(args.card_bank).open(encoding="utf-8"):
        card = json.loads(line)
        cards[str(card["knowledge_id"])] = card
    relations = sorted({str(c["relation_type"] or "") for c in cards.values()})
    affects = sorted({str(c["affect_prior"] or "") for c in cards.values()})
    encoder = SentenceTransformer(args.text_model, device="cpu")
    texts = sorted({t for r in records for t in r["candidate_texts"]})
    emb = {t: torch.from_numpy(encoder.encode([t])[0]).numpy() for t in texts}
    n, k_max = len(records), 5
    utility = np.zeros((n, k_max), dtype=np.float32)
    mask = np.zeros((n, k_max), dtype=bool)
    candidate_emb = np.zeros((n, k_max, 384), dtype=np.float32)
    meta = np.zeros((n, k_max, 3 + len(relations) + len(affects)), dtype=np.float32)
    dialogues, ids = [], []
    for row, record in enumerate(records):
        k = len(record["candidate_ids"])
        ids.append([str(v) for v in record["candidate_ids"]])
        for index in range(min(k, k_max)):
            mask[row, index] = bool(record.get("candidate_mask", [True] * k)[index])
            utility[row, index] = float(record.get("u_reg", record.get("u_ce"))[index])
            candidate_emb[row, index] = emb.get(record["candidate_texts"][index], 0.0)
            card = cards.get(ids[row][index], {})
            relation = str(card.get("relation_type") or "")
            affect = str(card.get("affect_prior") or "")
            meta[row, index, 0] = index / 4.0
            meta[row, index, 1] = 1.0
            meta[row, index, 2] = float(card.get("source_confidence") or 0.0)
            if relation in relations:
                meta[row, index, 3 + relations.index(relation)] = 1.0
            if affect in affects:
                meta[row, index, 3 + len(relations) + affects.index(affect)] = 1.0
        dialogues.append(record["dialogue_id"])
    unique = sorted(set(dialogues))
    random.Random(66).shuffle(unique)
    fold_of = {d: i % args.folds for i, d in enumerate(unique)}
    folds = np.array([fold_of[d] for d in dialogues])
    best = np.maximum(0.0, utility.max(axis=1))
    skip_regret = float(best.mean())
    results = {}
    # per-card M1
    m1 = np.zeros((n, k_max), dtype=np.float32)
    for fold in range(args.folds):
        tr, va = folds != fold, folds == fold
        x = meta[tr].reshape(-1, meta.shape[-1])
        y = utility[tr].reshape(-1)
        keep = mask[tr].reshape(-1)
        model = HistGradientBoostingRegressor(max_iter=80, max_depth=3)
        model.fit(x[keep], y[keep])
        m1[va] = model.predict(meta[va].reshape(-1, meta.shape[-1])).reshape(
            va.sum(), k_max
        )
    results["M1_per_card"] = {
        "query_macro_auc": query_macro_auc(m1, utility, mask),
        "spearman": utility_spearman(m1, utility, mask),
        "within_top5_regret": top1_regret(m1, utility, mask),
        "action_regret": float(
            np.mean(best - np.where(
                m1.max(axis=1) > 0,
                np.take_along_axis(utility, m1.argmax(axis=1)[:, None], axis=1).squeeze(1),
                0.0,
            ))
        ),
    }
    # candidate-specific O2 with shrinkage
    keys = []
    for row, record in enumerate(records):
        row_keys = []
        for index in range(k_max):
            card = cards.get(ids[row][index], {})
            row_keys.append(
                f"{record['target_modality']}|{record['missing_pattern']}|{index}|"
                f"{card.get('relation_type') or ''}|{card.get('affect_prior') or ''}"
            )
        keys.append(row_keys)
    o2 = np.full((n, k_max), -1e9)
    global_mean = float(utility[mask].mean())
    for fold in range(args.folds):
        tr, va = folds != fold, folds == fold
        stats = defaultdict(list)
        for row in np.where(tr)[0]:
            for index in np.where(mask[row])[0]:
                stats[keys[row][index]].append(float(utility[row, index]))
        means = {
            key: (len(v) * float(np.mean(v)) + 20 * global_mean) / (len(v) + 20)
            for key, v in stats.items()
        }
        for row in np.where(va)[0]:
            for index in np.where(mask[row])[0]:
                o2[row, index] = means.get(keys[row][index], global_mean)
    results["O2_candidate_cell"] = {
        "action_regret": float(
            np.mean(best - np.where(
                o2.max(axis=1) > 0,
                np.take_along_axis(utility, o2.argmax(axis=1)[:, None], axis=1).squeeze(1),
                0.0,
            ))
        )
    }
    # nested M2
    m2_regrets, m2_pred = [], np.zeros(n)
    for fold in range(args.folds):
        train = np.where(folds != fold)[0]
        val = np.where(folds == fold)[0]
        train_dialogues = sorted({dialogues[i] for i in train})
        random.Random(66 + fold).shuffle(train_dialogues)
        cut = max(1, int(len(train_dialogues) * 0.75))
        inner_train = set(train_dialogues[:cut])
        inner_cal = set(train_dialogues[cut:])
        it = np.array([i for i in train if dialogues[i] in inner_train])
        ic = np.array([i for i in train if dialogues[i] in inner_cal])
        state = np.stack([
            np.concatenate([
                np.asarray(records[i]["availability"], dtype=np.float32),
                np.eye(3)[records[i]["target_modality"]].astype(np.float32),
                np.eye(8)[records[i]["missing_pattern"]].astype(np.float32),
                [float(records[i]["missing_rate"]),
                 float(np.asarray(records[i]["empty_logits"]).reshape(-1)[0])],
            ])
            for i in range(n)
        ])
        model = Ridge(alpha=1.0).fit(state[it], best[it])
        cal_pred = model.predict(state[ic])
        top1 = np.where(mask[ic, 0], utility[ic, 0], 0.0)
        taus = np.quantile(cal_pred, np.linspace(0.05, 0.95, 19))
        best_tau, best_reg = 0.0, float("inf")
        for tau in taus:
            value = np.mean(best[ic] - np.where(cal_pred > tau, top1, 0.0))
            if value < best_reg:
                best_tau, best_reg = float(tau), float(value)
        val_pred = model.predict(state[val])
        m2_pred[val] = val_pred
        val_top1 = np.where(mask[val, 0], utility[val, 0], 0.0)
        m2_regrets.append(float(np.mean(best[val] - np.where(val_pred > best_tau, val_top1, 0.0))))
    results["M2_nested"] = {
        "use_skip_auc": float(roc_auc_score(best > 0, m2_pred)),
        "action_regret_fold_mean": float(np.mean(m2_regrets)),
    }
    # M3 paired action regret vs skip and same-use random
    m3 = np.zeros((n, k_max), dtype=np.float32)
    for fold in range(args.folds):
        tr, va = folds != fold, folds == fold
        x = candidate_emb[tr].reshape(-1, 384)
        y = utility[tr].reshape(-1)
        keep = mask[tr].reshape(-1)
        model = Ridge(alpha=1.0).fit(x[keep], y[keep])
        m3[va] = model.predict(candidate_emb[va].reshape(-1, 384)).reshape(va.sum(), k_max)
    use = m3.max(axis=1) > 0
    chosen = m3.argmax(axis=1)
    action_utility = np.where(use, utility[np.arange(n), chosen], 0.0)
    m3_regret = best - action_utility
    diffs = defaultdict(list)
    for i in range(n):
        diffs[dialogues[i]].append(m3_regret[i] - best[i])
    mean, lo, hi = bootstrap_cluster(diffs)
    results["M3_action"] = {
        "use_rate": float(use.mean()),
        "skip_rate": float(1 - use.mean()),
        "action_regret": float(m3_regret.mean()),
        "delta_vs_skip": mean,
        "delta_vs_skip_ci_low": lo,
        "delta_vs_skip_ci_high": hi,
    }
    rng = random.Random(66)
    same_use = []
    for _ in range(100):
        regrets = []
        for row in range(n):
            valid = np.where(mask[row])[0]
            if valid.size == 0:
                continue
            if use[row]:
                pick = valid[rng.randrange(valid.size)]
                regrets.append(best[row] - float(utility[row, pick]))
            else:
                regrets.append(best[row])
        same_use.append(float(np.mean(regrets)))
    results["same_use_random"] = {
        "mean": float(np.mean(same_use)),
        "p05": float(np.quantile(same_use, 0.05)),
        "p95": float(np.quantile(same_use, 0.95)),
    }
    decision = "CLOSE_FILTER"
    if results["M1_per_card"]["query_macro_auc"] >= 0.56 or results["M3_action"]["delta_vs_skip_ci_high"] < 0:
        decision = "ENTER_M4"
    elif results["M2_nested"]["use_skip_auc"] >= 0.58:
        decision = "KEEP_SKIP_ONLY"
    report = {
        "queries": n,
        "decision": decision,
        "baseline_skip": skip_regret,
        "results": results,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "p0_p4_decision.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
