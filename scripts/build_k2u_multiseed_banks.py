from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.per.batch import PERDatasetAdapter
from ComP.per.missing import MaskManifest
from ComP.per.types import MODALITIES
from ComP.train_per import _apply_manifest, _dataset_and_splits, _make_loader


def l2(values: np.ndarray) -> np.ndarray:
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-9, None)


def cost_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = l2(a)
    b = l2(b)
    return 1.0 - a @ b.T


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--mask-manifest", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--bank-seeds", default="66,67,68,69,70,71,72,73,74,75")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    payload = torch.load(args.student_checkpoint, map_location="cpu", weights_only=False)
    from argparse import Namespace

    cli = Namespace(**payload["args"])
    cli.no_cuda = True
    dataset, splits, _ = _dataset_and_splits(cli)
    loader = _make_loader(
        PERDatasetAdapter(dataset, splits[0]["train"]), cli, training=False
    )
    manifest = MaskManifest.load(args.mask_manifest)
    records = []
    for batch in loader:
        batch = _apply_manifest(batch, manifest, torch.device("cpu"))
        for b in range(batch.sequence_mask.size(0)):
            for t in range(batch.sequence_mask.size(1)):
                if not bool(batch.sequence_mask[b, t]):
                    continue
                records.append(
                    {
                        "sample_id": str(batch.sample_ids[b]),
                        "dialogue_id": str(batch.sample_ids[b]),
                        "features": {
                            name: batch.features[name][b, t].numpy().astype(np.float32)
                            for name in MODALITIES
                        },
                    }
                )
    seeds = [int(value) for value in args.bank_seeds.split(",") if value]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dialogues = sorted({record["dialogue_id"] for record in records})
    summary = {
        "repeats": [],
        "overall": {},
        "n_clusters": args.n_clusters,
        "seeds": seeds,
        "test_accessed": False,
    }
    for repeat in range(args.repeats):
        order = dialogues[:]
        random.Random(66 + repeat).shuffle(order)
        n = len(order)
        bank_end = int(n * 0.4)
        scorer_end = int(n * 0.8)
        bank_dialogues = set(order[:bank_end])
        scorer_dialogues = set(order[bank_end:scorer_end])
        eval_dialogues = set(order[scorer_end:])
        bank_records = [
            record for record in records if record["dialogue_id"] in bank_dialogues
        ]
        probe_records = [
            record for record in records if record["dialogue_id"] in scorer_dialogues
        ][:200]
        bank_by_target = {}
        probe_by_target = {}
        for target in MODALITIES:
            observed = [name for name in MODALITIES if name != target]
            bank_by_target[target] = np.stack(
                [
                    np.concatenate(
                        [l2(record["features"][name].reshape(1, -1))[0] for name in observed]
                    )
                    for record in bank_records
                ]
            )
            probe_by_target[target] = np.stack(
                [
                    np.concatenate(
                        [l2(record["features"][name].reshape(1, -1))[0] for name in observed]
                    )
                    for record in probe_records
                ]
            )
        repeat_report = {
            "repeat": repeat,
            "bank_dialogues": len(bank_dialogues),
            "scorer_dialogues": len(scorer_dialogues),
            "eval_dialogues": len(eval_dialogues),
            "targets": {},
        }
        for target in MODALITIES:
            fits = []
            for seed in seeds:
                model = KMeans(
                    n_clusters=args.n_clusters, random_state=seed, n_init=3
                ).fit(bank_by_target[target])
                fits.append(model)
            aris, cosines, jaccards, supports = [], [], [], []
            assignments = [fit.labels_ for fit in fits]
            for a, b in itertools.combinations(range(len(fits)), 2):
                aris.append(
                    adjusted_rand_score(assignments[a], assignments[b])
                )
                cost = cost_matrix(
                    fits[a].cluster_centers_, fits[b].cluster_centers_
                )
                rows, cols = linear_sum_assignment(cost)
                matched = 1.0 - cost[rows, cols]
                cosines.append(float(matched.mean()))
                top_a = np.argsort(
                    ((probe_by_target[target][:, None, :] - fits[a].cluster_centers_[None])
                     .__pow__(2)).sum(-1), axis=1
                )[:, :5]
                top_b = np.argsort(
                    ((probe_by_target[target][:, None, :] - fits[b].cluster_centers_[None])
                     .__pow__(2)).sum(-1), axis=1
                )[:, :5]
                jaccards.append(
                    float(np.mean([
                        len(set(x) & set(y)) / len(set(x) | set(y))
                        for x, y in zip(top_a, top_b)
                    ]))
                )
            for fit in fits:
                counts = np.bincount(fit.labels_)
                supports.extend(counts.tolist())
            supports = np.asarray(supports)
            repeat_report["targets"][target] = {
                "ari_mean": float(np.mean(aris)),
                "ari_min": float(np.min(aris)),
                "matched_cosine_mean": float(np.mean(cosines)),
                "matched_cosine_min": float(np.min(cosines)),
                "top5_jaccard_mean": float(np.mean(jaccards)),
                "top5_jaccard_min": float(np.min(jaccards)),
                "support_min": int(supports.min()),
                "support_median": float(np.median(supports)),
                "small_cluster_rate": float((supports < 10).mean()),
            }
        summary["repeats"].append(repeat_report)
    targets = MODALITIES
    overall = {}
    for target in targets:
        rows = [rep["targets"][target] for rep in summary["repeats"]]
        overall[target] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("ari_mean", "matched_cosine_mean", "top5_jaccard_mean", "small_cluster_rate")
        }
    summary["overall"] = overall
    (output / "stability.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(overall, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
