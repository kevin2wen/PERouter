from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.per.batch import PERDatasetAdapter
from ComP.per.missing import MaskManifest
from ComP.per.types import MODALITIES
from ComP.train_per import _apply_manifest, _dataset_and_splits, _make_loader


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def l2(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-9, None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--mask-manifest", required=True)
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
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
                        "utterance_id": str(batch.utterance_ids[b][t]),
                        "features": {
                            name: batch.features[name][b, t].numpy().astype(np.float32)
                            for name in MODALITIES
                        },
                    }
                )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dialogues = sorted({record["dialogue_id"] for record in records})
    repeat_reports = []
    overlap_report = {"overlaps": [], "total_overlap": 0}
    stability = {"prototype_cosine_by_target": {}}
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
        eval_records = [
            record for record in records if record["dialogue_id"] in eval_dialogues
        ]
        bank_payload = {
            "repeat": repeat,
            "bank_dialogues": sorted(bank_dialogues),
            "scorer_dialogues": sorted(scorer_dialogues),
            "evaluation_dialogues": sorted(eval_dialogues),
            "targets": {},
        }
        target_prototypes = defaultdict(list)
        for target in MODALITIES:
            observed = [name for name in MODALITIES if name != target]
            matrix = np.stack(
                [
                    np.concatenate(
                        [l2(record["features"][name].reshape(1, -1))[0] for name in observed]
                    )
                    for record in bank_records
                ]
            )
            target_matrix = np.stack(
                [record["features"][target] for record in bank_records]
            )
            kmeans = KMeans(
                n_clusters=args.n_clusters, random_state=66 + repeat, n_init=10
            ).fit(matrix)
            clusters = []
            for cluster in range(args.n_clusters):
                indices = np.where(kmeans.labels_ == cluster)[0]
                if indices.size == 0:
                    continue
                target_mean = target_matrix[indices].mean(axis=0)
                source_samples = sorted(
                    {bank_records[i]["sample_id"] for i in indices}
                )
                clusters.append(
                    {
                        "cluster": cluster,
                        "support": int(indices.size),
                        "prototype": torch.from_numpy(target_mean),
                        "source_sample_ids": source_samples,
                        "source_dialogue_ids": sorted(
                            {bank_records[i]["dialogue_id"] for i in indices}
                        ),
                    }
                )
                target_prototypes[target].append(target_mean)
            bank_payload["targets"][target] = {
                "centers": torch.from_numpy(kmeans.cluster_centers_),
                "clusters": clusters,
            }
        torch.save(bank_payload, output / f"bank_split_{repeat}.pt")
        # overlap audit
        bank_dialogue_set = bank_dialogues
        overlaps = [
            record["utterance_id"]
            for record in eval_records
            if record["dialogue_id"] in bank_dialogue_set
        ]
        overlap_report["overlaps"].append(
            {"repeat": repeat, "evaluation_queries": len(eval_records), "overlap": len(overlaps)}
        )
        overlap_report["total_overlap"] += len(overlaps)
        repeat_reports.append(
            {
                "repeat": repeat,
                "bank_dialogues": len(bank_dialogues),
                "scorer_dialogues": len(scorer_dialogues),
                "evaluation_dialogues": len(eval_dialogues),
                "bank_records": len(bank_records),
            }
        )
        for target, prototypes in target_prototypes.items():
            if len(prototypes) < 2:
                continue
            values = np.stack(prototypes)
            values = values / np.clip(
                np.linalg.norm(values, axis=1, keepdims=True), 1e-9, None
            )
            similarity = values @ values.T
            upper = similarity[np.triu_indices(values.shape[0], k=1)]
            stability["prototype_cosine_by_target"].setdefault(target, []).append(
                {
                    "repeat": repeat,
                    "mean": float(upper.mean()),
                    "min": float(upper.min()),
                }
            )
    (output / "support_overlap_audit.json").write_text(
        json.dumps(overlap_report, indent=2, sort_keys=True) + "\n"
    )
    (output / "bank_seed_stability.json").write_text(
        json.dumps(stability, indent=2, sort_keys=True) + "\n"
    )
    report = {
        "repeats": repeat_reports,
        "total_overlap": overlap_report["total_overlap"],
        "bank_files": args.repeats,
        "mask_manifest_sha256": sha256_file(args.mask_manifest),
        "student_checkpoint_sha256": sha256_file(args.student_checkpoint),
        "test_accessed": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
