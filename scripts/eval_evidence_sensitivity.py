from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ComP.per.batch import PERDatasetAdapter, move_batch_to_device
from ComP.per.evidence import randomize_evidence, shuffled_evidence, source_only_evidence
from ComP.per.evidence_adapter import load_adapter_checkpoint
from ComP.per.evidence_oracle import (
    apply_selected_ranks,
    isolate_rank_evidence,
    select_confidence_ranks,
    select_strong_ranks,
)
from ComP.per.evidence_sensitivity import (
    masked_mean,
    masked_rate,
    normalized_gradient,
    paired_logit_deltas,
)
from ComP.per.missing import MaskManifest
from ComP.per.text_evidence import TextNeighborRetriever
from ComP.per.types import MODALITIES, EvidenceBatch
from ComP.train_per import (
    _apply_manifest,
    _build_retriever,
    _build_student,
    _check_feature_paths,
    _dataset_and_splits,
    _load_evidence,
    _load_teacher,
    _make_loader,
    _metrics,
    masked_task_loss,
)

MODES = ("empty", "feature", "random", "shuffled", "source_only", "oracle_weak", "oracle_strong")
PAIRS = (
    ("feature", "empty"),
    ("feature", "random"),
    ("feature", "shuffled"),
    ("feature", "source_only"),
    ("oracle_weak", "random"),
    ("oracle_strong", "random"),
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


def reference_evidence(batch, retriever, source: str, evidence_dim: int, top_k: int, seed: int):
    if source == "text":
        return retriever.retrieve(
            batch.features,
            batch.availability,
            batch.sequence_mask,
            batch.sample_ids,
            batch.utterance_ids,
            raw_texts=batch.raw_texts,
        )
    return _load_evidence(batch, retriever, "feature", evidence_dim, top_k, seed=seed)


def mode_evidence(
    batch,
    full_batch,
    retriever,
    source: str,
    mode: str,
    evidence_dim: int,
    top_k: int,
    seed: int,
    model,
    teacher,
) -> EvidenceBatch:
    if mode == "empty":
        return _load_evidence(batch, None, "empty", evidence_dim, top_k, seed=seed)
    reference = reference_evidence(batch, retriever, source, evidence_dim, top_k, seed)
    if mode == "feature":
        return reference
    if mode == "random":
        return randomize_evidence(
            reference,
            retriever,
            seed=seed,
            query_sample_ids=batch.sample_ids,
        )
    if mode == "shuffled":
        return shuffled_evidence(
            reference,
            seed=seed,
            query_sample_ids=batch.sample_ids,
        )
    if mode == "source_only":
        return source_only_evidence(reference)
    if mode == "oracle_weak":
        return apply_selected_ranks(reference, select_confidence_ranks(reference))
    if mode == "oracle_strong":
        base_output = model(batch.features, batch.availability, evidence=reference)
        utilities = []
        for rank in range(top_k):
            probe = isolate_rank_evidence(reference, rank)
            probe_vectors = model.evidence_encoder(
                probe, batch.availability, ~batch.availability
            )
            scores = teacher.score_candidates(
                full_batch.features,
                batch.availability,
                base_output.prompt_candidates,
                probe_vectors,
                batch.labels,
            )
            utilities.append(scores.utility[..., 1])
        utility_by_rank = torch.stack(utilities, dim=-1)
        ranks, _ = select_strong_ranks(
            utility_by_rank,
            batch.availability,
            batch.sequence_mask,
            reference.mask,
        )
        return apply_selected_ranks(reference, ranks)
    raise ValueError(f"unknown mode: {mode}")


def differentiable_evidence(evidence: EvidenceBatch) -> EvidenceBatch:
    return EvidenceBatch(
        ids=evidence.ids,
        features=evidence.features.detach().clone().requires_grad_(True),
        mask=evidence.mask.clone(),
        source_confidence=evidence.source_confidence.clone(),
        conflict_score=(
            evidence.conflict_score.clone()
            if evidence.conflict_score is not None
            else None
        ),
    )


def run_sensitivity(model, batch, evidence: EvidenceBatch, weight: float, task: str):
    run_evidence = differentiable_evidence(evidence)
    model.zero_grad(set_to_none=True)
    route = torch.full_like(
        torch.zeros(batch.availability.size(0), batch.availability.size(1), 3,
                    device=batch.availability.device),
        float(weight),
    )
    output = model(
        batch.features,
        batch.availability,
        evidence=run_evidence,
        route_override=route,
    )
    loss = masked_task_loss(output.logits, batch.labels, batch.sequence_mask, task)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite sensitivity loss")

    evidence_grad = torch.autograd.grad(
        loss, run_evidence.features, retain_graph=True, allow_unused=True
    )[0]
    # Fusion route is a fixed constant under route_override; its gradient is zero by definition.
    route_grad = None
    encoder_params = list(model.evidence_encoder.parameters())
    encoder_grads = torch.autograd.grad(
        loss, encoder_params, retain_graph=True, allow_unused=True
    )
    projection_params = list(model.evidence_encoder.projections.parameters())
    projection_grads = torch.autograd.grad(
        loss, projection_params, retain_graph=False, allow_unused=True
    )
    return {
        "logits": output.logits.detach().cpu(),
        "evidence_gradient_norm": float(normalized_gradient(evidence_grad).item()),
        "route_gradient_norm": float(normalized_gradient(route_grad).item()),
        "encoder_gradient_norm": float(
            torch.sqrt(sum(grad.detach().float().pow(2).sum() for grad in encoder_grads if grad is not None)).item()
        ),
        "projection_gradient_norm": float(
            torch.sqrt(sum(grad.detach().float().pow(2).sum() for grad in projection_grads if grad is not None)).item()
        ),
    }


def main() -> int:
    cli = parser().parse_args()
    if cli.split != "validation":
        raise PermissionError("E1.5 sensitivity is validation-only")
    if not Path(cli.mask_manifest).is_file():
        raise FileNotFoundError(cli.mask_manifest)
    payload = torch.load(cli.student_checkpoint, map_location="cpu", weights_only=False)
    saved = dict(payload["args"])
    device = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    saved.update({"no_cuda": device.type == "cpu", "gpu": cli.gpu, "device": device})
    saved["evidence_source"] = cli.evidence_source
    args = Namespace(**saved)
    _check_feature_paths(args)
    dataset, splits, _ = _dataset_and_splits(args)
    split = next(item for item in splits if int(item["fold"]) == 0)
    adim, tdim, vdim = dataset.get_featDim()
    train_loader = _make_loader(PERDatasetAdapter(dataset, split["train"]), args, training=False)
    target_loader = _make_loader(PERDatasetAdapter(dataset, split["val"]), args, training=False)
    retriever = None
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
    task = "regression" if args.dataset in {"CMUMOSI", "CMUMOSEI"} else "classification"

    rows = []
    gradient_rows = []
    predictions_by_mode = {mode: [] for mode in MODES}
    labels_all = []
    sequence_all = []
    aggregate = {pair: [] for pair in PAIRS}

    for batch_index, full_cpu in enumerate(target_loader):
        with torch.no_grad():
            batch = _apply_manifest(full_cpu, manifest, device)
            full_batch = move_batch_to_device(full_cpu, device)
            modes = {}
            for mode in MODES:
                modes[mode] = mode_evidence(
                    batch, full_batch, retriever, cli.evidence_source,
                    mode, evidence_dim, args.top_k, args.seed, model, teacher,
                )

        labels = batch.labels.detach().cpu()
        mask = batch.sequence_mask.detach().cpu()
        labels_all.append(labels)
        sequence_all.append(mask)
        outputs = {}
        for mode in MODES:
            outputs[mode] = run_sensitivity(model, batch, modes[mode], 0.5, task)
            predictions_by_mode[mode].append(outputs[mode]["logits"])
            gradient_rows.append({
                "batch": batch_index,
                "evidence_source": cli.evidence_source,
                "evidence_mode": mode,
                "evidence_gradient_norm": outputs[mode]["evidence_gradient_norm"],
                "route_gradient_norm": outputs[mode]["route_gradient_norm"],
                "encoder_gradient_norm": outputs[mode]["encoder_gradient_norm"],
                "projection_gradient_norm": outputs[mode]["projection_gradient_norm"],
            })

        for reference_mode, comparison_mode in PAIRS:
            delta = paired_logit_deltas(
                outputs[reference_mode]["logits"],
                outputs[comparison_mode]["logits"],
                labels,
                mask,
                task,
            )
            ref_available = modes[reference_mode].mask.any(dim=-1).detach().cpu()
            cmp_available = modes[comparison_mode].mask.any(dim=-1).detach().cpu()
            availability = batch.availability.detach().cpu()
            for b, sample_id in enumerate(batch.sample_ids):
                for t, utterance_id in enumerate(batch.utterance_ids[b]):
                    if not bool(mask[b, t]):
                        continue
                    for m, modality in enumerate(MODALITIES):
                        if bool(availability[b, t, m]):
                            continue
                        row = {
                            "batch": batch_index,
                            "split": cli.split,
                            "evidence_source": cli.evidence_source,
                            "comparison": f"{reference_mode}_vs_{comparison_mode}",
                            "sample_id": sample_id,
                            "utterance_id": utterance_id,
                            "target_modality": modality,
                            "reference_evidence_available": bool(ref_available[b, t]),
                            "comparison_evidence_available": bool(cmp_available[b, t]),
                            "logits_delta_l1": float(delta["logits_delta_l1"][b, t]),
                            "logits_delta_l2": float(delta["logits_delta_l2"][b, t]),
                            "prediction_delta": float(delta["prediction_delta"][b, t]),
                            "sign_flip": bool(delta["sign_flip"][b, t]),
                            "classification_margin_delta": float(delta["classification_margin_delta"][b, t]),
                            "reference_evidence_gradient_norm": outputs[reference_mode]["evidence_gradient_norm"],
                            "comparison_evidence_gradient_norm": outputs[comparison_mode]["evidence_gradient_norm"],
                            "reference_encoder_gradient_norm": outputs[reference_mode]["encoder_gradient_norm"],
                            "comparison_encoder_gradient_norm": outputs[comparison_mode]["encoder_gradient_norm"],
                            "reference_projection_gradient_norm": outputs[reference_mode]["projection_gradient_norm"],
                            "comparison_projection_gradient_norm": outputs[comparison_mode]["projection_gradient_norm"],
                            "reference_route_gradient_norm": outputs[reference_mode]["route_gradient_norm"],
                            "comparison_route_gradient_norm": outputs[comparison_mode]["route_gradient_norm"],
                        }
                        rows.append(row)
                        aggregate[(reference_mode, comparison_mode)].append(row)

    labels = torch.cat(labels_all, dim=0)
    sequence = torch.cat(sequence_all, dim=0)
    metrics_by_mode = {}
    for mode in MODES:
        logits = torch.cat(predictions_by_mode[mode], dim=0)
        prediction = logits.squeeze(-1) if task == "regression" else logits.argmax(dim=-1)
        metrics_by_mode[mode] = _metrics(labels, prediction, sequence, args.dataset)

    output = Path(cli.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    row_fields = sorted({key for row in rows for key in row})
    with (output / "evidence_sensitivity_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fields)
        writer.writeheader()
        writer.writerows(rows)
    grad_fields = sorted({key for row in gradient_rows for key in row})
    with (output / "evidence_gradient_norm.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=grad_fields)
        writer.writeheader()
        writer.writerows(gradient_rows)

    summary_pairs = {}
    for pair, pair_rows in aggregate.items():
        if not pair_rows:
            continue
        summary_pairs[f"{pair[0]}_vs_{pair[1]}"] = {
            "row_count": len(pair_rows),
            "mean_l1": sum(row["logits_delta_l1"] for row in pair_rows) / len(pair_rows),
            "mean_l2": sum(row["logits_delta_l2"] for row in pair_rows) / len(pair_rows),
            "mean_prediction_delta": sum(row["prediction_delta"] for row in pair_rows) / len(pair_rows),
            "sign_flip_rate": sum(row["sign_flip"] for row in pair_rows) / len(pair_rows),
            "mean_classification_margin_delta": sum(row["classification_margin_delta"] for row in pair_rows) / len(pair_rows),
            "mean_reference_evidence_gradient_norm": sum(row["reference_evidence_gradient_norm"] for row in pair_rows) / len(pair_rows),
            "mean_comparison_evidence_gradient_norm": sum(row["comparison_evidence_gradient_norm"] for row in pair_rows) / len(pair_rows),
            "mean_reference_encoder_gradient_norm": sum(row["reference_encoder_gradient_norm"] for row in pair_rows) / len(pair_rows),
            "mean_comparison_encoder_gradient_norm": sum(row["comparison_encoder_gradient_norm"] for row in pair_rows) / len(pair_rows),
        }
    summary = {
        "schema_version": 1,
        "split": cli.split,
        "test_access": "forbidden",
        "evidence_source": cli.evidence_source,
        "evidence_source_version": (
            "feature_neighbor_v1" if cli.evidence_source == "feature" else "text_tfidf_ngram12_v1"
        ),
        "student_checkpoint": cli.student_checkpoint,
        "student_checkpoint_sha256": sha256_file(cli.student_checkpoint),
        "teacher_checkpoint": cli.teacher_checkpoint,
        "teacher_checkpoint_sha256": sha256_file(cli.teacher_checkpoint),
        "mask_manifest": cli.mask_manifest,
        "mask_manifest_sha256": sha256_file(cli.mask_manifest),
        "comparisons": summary_pairs,
        "metrics_by_mode": metrics_by_mode,
    }
    (output / "evidence_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
