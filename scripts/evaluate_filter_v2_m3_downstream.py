from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def metrics(pred, label):
    valid = label != 0
    y = (label[valid] > 0).astype(int)
    p = (pred[valid] > 0).astype(int)
    return {
        "acc": float(accuracy_score(y, p)) if y.size else None,
        "f1": float(f1_score(y, p, average="weighted")) if y.size else None,
        "mae": float(np.mean(np.abs(pred[valid] - label[valid]))) if y.size else None,
        "corr": float(np.corrcoef(label[valid], pred[valid])[0, 1]) if y.size > 1 else None,
        "count": int(y.size),
    }


def cluster_bootstrap(per_dialogue, n_boot=2000, seed=66):
    groups = [np.asarray(v) for v in per_dialogue.values()]
    rng = np.random.default_rng(seed)
    draws = np.array([
        np.mean([groups[i].mean() for i in rng.integers(0, len(groups), len(groups))])
        for _ in range(n_boot)
    ])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", required=True)
    parser.add_argument("--text-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-seeds", type=int, default=100)
    args = parser.parse_args()
    from sentence_transformers import SentenceTransformer

    payload = torch.load(args.feedback, map_location="cpu", weights_only=False)
    records = payload["records"]
    if "candidate_regression" not in records[0]:
        raise SystemExit("feedback has no candidate_regression; rebuild first")
    encoder = SentenceTransformer(args.text_model, device="cpu")
    texts = sorted({t for r in records for t in r["candidate_texts"]})
    emb = {t: torch.from_numpy(encoder.encode([t])[0]).numpy() for t in texts}
    n, kmax = len(records), 5
    util = np.zeros((n, kmax), dtype=np.float32)
    reg = np.zeros((n, kmax), dtype=np.float32)
    mask = np.zeros((n, kmax), dtype=bool)
    cand = np.zeros((n, kmax, 384), dtype=np.float32)
    empty = np.zeros(n, dtype=np.float32)
    label = np.zeros(n, dtype=np.float32)
    dialogues = []
    for row, record in enumerate(records):
        k = len(record["candidate_ids"])
        empty[row] = record["empty_regression"]
        label[row] = record["label"]
        for index in range(min(k, kmax)):
            mask[row, index] = bool(record.get("candidate_mask", [True] * k)[index])
            util[row, index] = float(record.get("u_reg", record.get("u_ce"))[index])
            reg[row, index] = float(record["candidate_regression"][index])
            cand[row, index] = emb.get(record["candidate_texts"][index], 0.0)
        dialogues.append(record["dialogue_id"])
    unique = sorted(set(dialogues))
    random.Random(66).shuffle(unique)
    fold_of = {d: i % args.folds for i, d in enumerate(unique)}
    folds = np.array([fold_of[d] for d in dialogues])
    pred = np.zeros((n, kmax), dtype=np.float32)
    shuffled_pred = np.zeros((n, kmax), dtype=np.float32)
    for fold in range(args.folds):
        tr, va = folds != fold, folds == fold
        x = cand[tr].reshape(-1, 384)
        y = util[tr].reshape(-1)
        keep = mask[tr].reshape(-1)
        model = Ridge(alpha=1.0).fit(x[keep], y[keep])
        pred[va] = model.predict(cand[va].reshape(-1, 384)).reshape(va.sum(), kmax)
        rng = np.random.default_rng(1000 + fold)
        perm = np.argsort(rng.random(cand[va].shape[:2]), axis=1)
        shuffled = np.take_along_axis(cand[va], perm[:, :, None], axis=1)
        shuffled_pred[va] = model.predict(shuffled.reshape(-1, 384)).reshape(
            va.sum(), kmax
        )
    # cross-fitted thresholds on downstream F1
    taus = np.zeros(args.folds)
    for fold in range(args.folds):
        other = folds != fold
        candidates = np.quantile(pred[other].max(axis=1), np.linspace(0.05, 0.95, 19))
        best = (0.0, -1.0)
        for tau in candidates:
            chosen = pred[other].argmax(axis=1)
            prediction = np.where(
                pred[other].max(axis=1) > tau,
                np.take_along_axis(reg[other], chosen[:, None], axis=1).squeeze(1),
                empty[other],
            )
            score = metrics(prediction, label[other])["f1"] or 0.0
            if score > best[1]:
                best = (float(tau), score)
        taus[fold] = best[0]
    def action(prediction, mode, tau=None):
        chosen = prediction.argmax(axis=1)
        if mode == "raw":
            use = prediction.max(axis=1) > 0
        else:
            use = np.array([prediction[i].max() > taus[folds[i]] for i in range(n)])
        selected = np.take_along_axis(reg, chosen[:, None], axis=1).squeeze(1)
        return np.where(use, selected, empty), use, chosen
    results = {}
    baseline = {
        "empty": empty,
        "skip": empty,
        "top1": np.where(mask[:, 0], reg[:, 0], empty),
    }
    for name, prediction in baseline.items():
        results[name] = metrics(prediction, label)
    for mode in ("raw", "calibrated"):
        prediction, use, chosen = action(pred, mode)
        results[f"M3_{mode}"] = {
            **metrics(prediction, label),
            "use_rate": float(use.mean()),
            "skip_rate": float(1 - use.mean()),
        }
        results[f"M3_{mode}"]["_prediction"] = prediction
        results[f"M3_{mode}"]["_use"] = use
    shuf_pred, shuf_use, _ = action(shuffled_pred, "raw")
    results["M3_shuffled"] = {
        **metrics(shuf_pred, label),
        "use_rate": float(shuf_use.mean()),
    }
    use = results["M3_raw"]["_use"]
    rng = random.Random(66)
    same_use = []
    matched = []
    for _ in range(args.random_seeds):
        same_pred = empty.copy()
        matched_pred = empty.copy()
        for row in range(n):
            valid = np.where(mask[row])[0]
            if valid.size == 0:
                continue
            if use[row]:
                same_pred[row] = reg[row, valid[rng.randrange(valid.size)]]
            if rng.random() < use.mean():
                matched_pred[row] = reg[row, valid[rng.randrange(valid.size)]]
        same_use.append(metrics(same_pred, label))
        matched.append(metrics(matched_pred, label))
    for name, rows in (("same_use_random", same_use), ("matched_use_random", matched)):
        results[name] = {
            key: float(np.mean([r[key] for r in rows if r[key] is not None]))
            for key in ("acc", "f1", "mae", "corr")
        }
    m3_pred = results["M3_raw"]["_prediction"]
    comparisons = {}
    for name, other in (("empty", empty), ("same_use_random", same_use[0] if False else None)):
        if other is None:
            continue
    per_dialogue = defaultdict(lambda: {"dacc": [], "df1": []})
    for row in range(n):
        if label[row] == 0:
            continue
        d = dialogues[row]
        per_dialogue[d]["dacc"].append(
            int((m3_pred[row] > 0) == (label[row] > 0))
            - int((empty[row] > 0) == (label[row] > 0))
        )
    lo, hi = cluster_bootstrap({k: v["dacc"] for k, v in per_dialogue.items()})
    comparisons["M3_minus_empty_acc"] = {"delta": float(np.mean([x for v in per_dialogue.values() for x in v["dacc"]])), "ci_low": lo, "ci_high": hi}
    clean = {k: v for k, v in results.items() if not k.startswith("_")}
    for key in list(clean):
        if isinstance(clean[key], dict):
            clean[key] = {k: v for k, v in clean[key].items() if not k.startswith("_")}
    report = {
        "queries": n,
        "taus": taus.tolist(),
        "results": clean,
        "comparisons": comparisons,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "m3_downstream.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
