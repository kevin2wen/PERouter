from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ComP.per.batch import PERDatasetAdapter, move_batch_to_device
from ComP.per.evidence_adapter import load_adapter_checkpoint
from ComP.per.evidence_oracle import isolate_rank_evidence
from ComP.per.evidence_quality import (
    binary_label_consistency,
    candidate_statistics,
    rank_histogram,
    recall_at_k,
)
from ComP.per.missing import MaskManifest
from ComP.per.text_evidence import TextNeighborRetriever
from ComP.per.types import MODALITIES
from ComP.train_per import (
    _apply_manifest,
    _build_retriever,
    _build_student,
    _check_feature_paths,
    _dataset_and_splits,
    _load_evidence,
    _load_teacher,
    _make_loader,
)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--student-checkpoint", required=True)
    p.add_argument("--teacher-checkpoint", required=True)
    p.add_argument("--mask-manifest", required=True)
    p.add_argument("--split", choices=("validation",), default="validation")
    p.add_argument("--evidence-source", choices=("feature", "text"), required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--adapter-checkpoint")
    p.add_argument("--gpu", type=int, default=0)
    return p


def build_label_map(loader) -> Dict[str, float]:
    labels = {}
    for batch in loader:
        for b, sample_id in enumerate(batch.sample_ids):
            for t, utterance_id in enumerate(batch.utterance_ids[b]):
                if bool(batch.sequence_mask[b, t]):
                    labels[f"{sample_id}:{utterance_id}"] = float(batch.labels[b, t])
    return labels


def main() -> int:
    cli = parser().parse_args()
    if cli.split != "validation":
        raise PermissionError("candidate quality is validation-only")
    if not Path(cli.mask_manifest).is_file():
        raise FileNotFoundError(cli.mask_manifest)
    payload = torch.load(cli.student_checkpoint, map_location="cpu", weights_only=False)
    saved = dict(payload["args"])
    device = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    saved.update({"no_cuda": device.type == "cpu", "gpu": cli.gpu, "device": device})
    args = Namespace(**saved)
    _check_feature_paths(args)
    dataset, splits, _ = _dataset_and_splits(args)
    split = next(item for item in splits if int(item["fold"]) == 0)
    adim, tdim, vdim = dataset.get_featDim()
    train_loader = _make_loader(PERDatasetAdapter(dataset, split["train"]), args, training=False)
    val_loader = _make_loader(PERDatasetAdapter(dataset, split["val"]), args, training=False)

    train_labels = build_label_map(train_loader)
    evidence_dim = adim + tdim + vdim
    if cli.evidence_source == "text":
        retriever = TextNeighborRetriever(
            {"audio": adim, "text": tdim, "video": vdim},
            top_k=args.top_k,
            device=torch.device("cpu"),
        )
        retriever.add_loader(train_loader)
    else:
        retriever = _build_retriever(train_loader, adim, tdim, vdim, args.top_k)
    model = _build_student(args, device, adim, tdim, vdim, evidence_dim)
    model.load_state_dict(payload["model"], strict=True)
    if cli.adapter_checkpoint:
        load_adapter_checkpoint(
            model, cli.adapter_checkpoint, expected_source=cli.evidence_source
        )
    model.eval()
    teacher = _load_teacher(args, device, adim, tdim, vdim, 0)
    manifest = MaskManifest.load(cli.mask_manifest)

    rows = []
    all_confidence = []
    all_utility = []
    all_ranks = []
    all_active = []
    candidate_recall = []
    prompt_similarity = {rank: [[] for _ in MODALITIES] for rank in range(args.top_k)}
    with torch.no_grad():
        for batch_index, full_cpu in enumerate(val_loader):
            batch = _apply_manifest(full_cpu, manifest, device)
            full_batch = move_batch_to_device(full_cpu, device)
            if cli.evidence_source == "text":
                reference = retriever.retrieve(
                    batch.features, batch.availability, batch.sequence_mask,
                    batch.sample_ids, batch.utterance_ids, raw_texts=batch.raw_texts,
                )
            else:
                reference = _load_evidence(
                    batch, retriever, "feature", evidence_dim, args.top_k, seed=args.seed
                )
            base_output = model(batch.features, batch.availability, evidence=reference)
            evidence_utility = []
            rank_vectors = []
            for rank in range(args.top_k):
                probe = isolate_rank_evidence(reference, rank)
                vectors = model.evidence_encoder(
                    probe, batch.availability, ~batch.availability
                )
                rank_vectors.append(vectors)
                scores = teacher.score_candidates(
                    full_batch.features,
                    batch.availability,
                    base_output.prompt_candidates,
                    vectors,
                    batch.labels,
                )
                evidence_utility.append(scores.utility[..., 1])
            utility = torch.stack(evidence_utility, dim=-1)
            first_probe = isolate_rank_evidence(reference, 0)
            first_vectors = model.evidence_encoder(
                first_probe, batch.availability, ~batch.availability
            )
            first_scores = teacher.score_candidates(
                full_batch.features,
                batch.availability,
                base_output.prompt_candidates,
                first_vectors,
                batch.labels,
            )
            prompt_utility = first_scores.utility[..., 0]
            active = (
                (~batch.availability)
                & batch.sequence_mask.unsqueeze(-1)
                & reference.mask.any(dim=-1, keepdim=True)
            )
            stats = candidate_statistics(utility, prompt_utility, active)
            all_confidence.append(reference.source_confidence.detach().cpu())
            all_utility.append(utility.detach().cpu())
            all_ranks.append(stats["best_rank"].detach().cpu())
            all_active.append(active.detach().cpu())

            conf_cpu = reference.source_confidence.detach().cpu()
            util_cpu = utility.detach().cpu()
            active_cpu = active.detach().cpu()
            best_rank_cpu = stats["best_rank"].detach().cpu()
            gain_cpu = stats["oracle_gain"].detach().cpu()
            helpful_cpu = stats["has_helpful_candidate"].detach().cpu()
            availability_cpu = batch.availability.detach().cpu()
            labels_cpu = batch.labels.detach().cpu()
            seq_cpu = batch.sequence_mask.detach().cpu()
            ids = reference.ids

            for b, sample_id in enumerate(batch.sample_ids):
                for t, utterance_id in enumerate(batch.utterance_ids[b]):
                    if not bool(seq_cpu[b, t]):
                        continue
                    for m, modality in enumerate(MODALITIES):
                        if not bool(active_cpu[b, t, m]):
                            continue
                        for rank in range(args.top_k):
                            candidate_id = ids[b][t][rank]
                            candidate_label = train_labels.get(candidate_id)
                            target_label = float(labels_cpu[b, t])
                            label_consistent = (
                                (candidate_label > 0) == (target_label > 0)
                                if candidate_label is not None
                                else None
                            )
                            vector = rank_vectors[rank][modality][b, t]
                            prompt = base_output.prompt_candidates[modality][b, t]
                            cosine = float(F.cosine_similarity(vector, prompt, dim=0).item())
                            prompt_similarity[rank][m].append(cosine)
                            rows.append({
                                "batch": batch_index,
                                "split": cli.split,
                                "evidence_source": cli.evidence_source,
                                "mask_rate": args.mask_rate,
                                "sample_id": sample_id,
                                "utterance_id": utterance_id,
                                "target_modality": modality,
                                "rank": rank,
                                "candidate_id": candidate_id,
                                "source_confidence": float(conf_cpu[b, t, rank]),
                                "utility": float(util_cpu[b, t, m, rank]),
                                "prompt_utility": float(prompt_utility[b, t, m].detach().cpu()),
                                "best_rank": int(best_rank_cpu[b, t, m]),
                                "oracle_gain": float(gain_cpu[b, t, m]),
                                "has_helpful_candidate": bool(helpful_cpu[b, t, m]),
                                "candidate_label": candidate_label,
                                "target_label": target_label,
                                "label_consistent": label_consistent,
                                "candidate_prompt_cosine": cosine,
                            })

    confidence = torch.cat(all_confidence, dim=0)
    utility = torch.cat(all_utility, dim=0)
    ranks = torch.cat(all_ranks, dim=0)
    active = torch.cat(all_active, dim=0)
    recall = recall_at_k(utility, prompt_utility=torch.tensor(0.0), active=active) if False else None
    # Recompute recall from stored rows without changing teacher semantics.
    recall_by_k = {}
    for k in range(1, args.top_k + 1):
        selected = []
        for row in rows:
            if row["rank"] < k:
                selected.append(row)
        grouped = {}
        for row in selected:
            key = (row["batch"], row["sample_id"], row["utterance_id"], row["target_modality"])
            grouped.setdefault(key, []).append(row)
        helpful = sum(any(item["utility"] > item["prompt_utility"] for item in items) for items in grouped.values())
        recall_by_k[k] = helpful / len(grouped) if grouped else 0.0

    conf_values = confidence[reference.mask].tolist() if False else [row["source_confidence"] for row in rows]
    utility_values = [row["utility"] for row in rows]
    rank_values = [row["rank"] for row in rows]
    confidence_rho = spearmanr(conf_values, utility_values)
    rank_rho = spearmanr(rank_values, utility_values)
    active_count = sum(1 for row in rows if row["rank"] == 0)
    helpful_count = sum(1 for row in rows if row["rank"] == 0 and row["has_helpful_candidate"])
    output = Path(cli.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (output / "candidate_quality_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "candidate_recall_at_k.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["evidence_source", "total_k", "prefix_k", "active_positions", "helpful_positions", "helpful_rate"])
        writer.writeheader()
        for k, rate in recall_by_k.items():
            writer.writerow({
                "evidence_source": cli.evidence_source,
                "total_k": args.top_k,
                "prefix_k": k,
                "active_positions": active_count,
                "helpful_positions": int(round(rate * active_count)),
                "helpful_rate": rate,
            })
    with (output / "candidate_confidence_utility_corr.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["comparison", "spearman_rho", "pvalue", "count"])
        writer.writeheader()
        writer.writerow({"comparison": "confidence_vs_utility", "spearman_rho": confidence_rho.statistic, "pvalue": confidence_rho.pvalue, "count": len(rows)})
        writer.writerow({"comparison": "rank_vs_utility", "spearman_rho": rank_rho.statistic, "pvalue": rank_rho.pvalue, "count": len(rows)})
    with (output / "candidate_prompt_similarity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["evidence_source", "rank", "target_modality", "count", "mean_cosine", "min_cosine", "max_cosine"])
        writer.writeheader()
        for rank, per_modality in prompt_similarity.items():
            for m, modality in enumerate(MODALITIES):
                values = per_modality[m]
                writer.writerow({
                    "evidence_source": cli.evidence_source, "rank": rank,
                    "target_modality": modality, "count": len(values),
                    "mean_cosine": sum(values) / len(values) if values else math.nan,
                    "min_cosine": min(values) if values else math.nan,
                    "max_cosine": max(values) if values else math.nan,
                })
    summary = {
        "schema_version": 1,
        "split": cli.split,
        "test_access": "forbidden",
        "evidence_source": cli.evidence_source,
        "uses_validation_labels_for_teacher_utility": True,
        "deployable": False,
        "student_checkpoint_sha256": sha256_file(cli.student_checkpoint),
        "teacher_checkpoint_sha256": sha256_file(cli.teacher_checkpoint),
        "mask_manifest_sha256": sha256_file(cli.mask_manifest),
        "top_k": args.top_k,
        "active_positions": active_count,
        "helpful_positions": helpful_count,
        "helpful_candidate_rate": helpful_count / active_count if active_count else 0.0,
        "best_rank_distribution": rank_histogram(ranks, active, args.top_k),
        "recall_at_k": recall_by_k,
        "confidence_utility_spearman": {"rho": confidence_rho.statistic, "pvalue": confidence_rho.pvalue},
        "rank_utility_spearman": {"rho": rank_rho.statistic, "pvalue": rank_rho.pvalue},
        "label_consistency_rate": (
            sum(1 for row in rows if row["rank"] == 0 and row["label_consistent"] is True) / active_count
            if active_count else 0.0
        ),
        "mean_oracle_gain": (
            sum(row["oracle_gain"] for row in rows if row["rank"] == 0) / active_count
            if active_count else 0.0
        ),
    }
    (output / "candidate_quality_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
