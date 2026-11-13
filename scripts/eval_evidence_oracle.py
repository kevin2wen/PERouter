from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ComP.per.batch import PERDatasetAdapter, move_batch_to_device
from ComP.per.evidence_audit import EvidenceCoverageAccumulator, evidence_trace_rows
from ComP.per.evidence import randomize_evidence, shuffled_evidence, source_only_evidence
from ComP.per.evidence_adapter import load_adapter_checkpoint
from ComP.per.evidence_metrics import grouped_metrics
from ComP.per.evidence_oracle import (
    apply_selected_ranks,
    isolate_rank_evidence,
    select_confidence_ranks,
    select_strong_ranks,
    validate_evidence_oracle_split,
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
    _metrics,
    masked_task_loss,
)

EVIDENCE_MODES = (
    "empty",
    "feature",
    "random",
    "shuffled",
    "source_only",
    "oracle_weak",
    "oracle_strong",
)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--mask-manifest", required=True)
    parser.add_argument("--evidence-mode", choices=EVIDENCE_MODES, required=True)
    parser.add_argument("--evidence-source", choices=("feature", "text"), default="feature")
    parser.add_argument("--fixed-source-weight", type=float, default=0.5)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter-checkpoint")
    parser.add_argument("--gpu", type=int, default=0)
    return parser


def main() -> int:
    cli = build_parser().parse_args()
    validate_evidence_oracle_split(cli.evidence_mode, cli.split)
    if not 0.0 <= cli.fixed_source_weight <= 1.0:
        raise ValueError("fixed source weight must be in [0,1]")
    if cli.evidence_mode == "oracle_strong" and not cli.teacher_checkpoint:
        raise ValueError("oracle_strong requires --teacher-checkpoint")
    manifest_path = Path(cli.mask_manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    payload = torch.load(cli.student_checkpoint, map_location="cpu", weights_only=False)
    saved = dict(payload["args"])
    device = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    saved.update({"no_cuda": device.type == "cpu", "gpu": cli.gpu, "device": device})
    saved["evidence_mode"] = cli.evidence_mode
    saved["fixed_source_weight"] = cli.fixed_source_weight
    if cli.top_k is not None:
        saved["top_k"] = cli.top_k
    args = Namespace(**saved)
    _check_feature_paths(args)

    dataset, splits, _ = _dataset_and_splits(args)
    split = next(item for item in splits if int(item["fold"]) == 0)
    adim, tdim, vdim = dataset.get_featDim()
    train_adapter = PERDatasetAdapter(dataset, split["train"])
    target_key = "val" if cli.split == "validation" else "test"
    target_adapter = PERDatasetAdapter(dataset, split[target_key])
    train_loader = _make_loader(train_adapter, args, training=False)
    target_loader = _make_loader(target_adapter, args, training=False)

    retriever = None
    evidence_dim = adim + tdim + vdim
    if cli.evidence_mode != "empty":
        if cli.evidence_source == "text":
            retriever = TextNeighborRetriever(
                {"audio": adim, "text": tdim, "video": vdim},
                top_k=args.top_k,
                device=torch.device("cpu"),
            )
            retriever.add_loader(train_loader)
        else:
            retriever = _build_retriever(train_loader, adim, tdim, vdim, args.top_k)
        evidence_dim = retriever.evidence_dim
    model = _build_student(args, device, adim, tdim, vdim, evidence_dim)
    model.load_state_dict(payload["model"], strict=True)
    if cli.adapter_checkpoint:
        load_adapter_checkpoint(
            model, cli.adapter_checkpoint, expected_source=cli.evidence_source
        )
    model.eval()
    teacher = None
    if cli.evidence_mode == "oracle_strong":
        teacher = _load_teacher(args, device, adim, tdim, vdim, 0)
    manifest = MaskManifest.load(str(manifest_path))

    labels_all = []
    predictions_all = []
    sequence_all = []
    availability_all = []
    selected_rank_counts = {str(rank): 0 for rank in range(args.top_k)}
    selection_rows = []
    trace_rows = []
    coverage = EvidenceCoverageAccumulator()

    with torch.no_grad():
        for batch_index, full_cpu in enumerate(target_loader):
            batch = _apply_manifest(full_cpu, manifest, device)
            full_batch = move_batch_to_device(full_cpu, device)
            reference = None
            selected_ranks = None
            utilities_by_rank = None

            if cli.evidence_mode == "empty":
                evidence = _load_evidence(
                    batch, None, "empty", evidence_dim, args.top_k, seed=args.seed
                )
            else:
                if cli.evidence_source == "text":
                    reference = retriever.retrieve(
                        batch.features,
                        batch.availability,
                        batch.sequence_mask,
                        batch.sample_ids,
                        batch.utterance_ids,
                        raw_texts=batch.raw_texts,
                    )
                else:
                    reference = _load_evidence(
                        batch, retriever, "feature", evidence_dim, args.top_k, seed=args.seed
                    )
                if cli.evidence_mode == "feature":
                    evidence = reference
                elif cli.evidence_mode == "random":
                    evidence = randomize_evidence(
                        reference,
                        retriever,
                        seed=args.seed,
                        query_sample_ids=batch.sample_ids,
                    )
                elif cli.evidence_mode == "shuffled":
                    evidence = shuffled_evidence(
                        reference,
                        seed=args.seed,
                        query_sample_ids=batch.sample_ids,
                    )
                elif cli.evidence_mode == "source_only":
                    evidence = source_only_evidence(reference)
                elif cli.evidence_mode == "oracle_weak":
                    selected_ranks = select_confidence_ranks(reference)
                    evidence = apply_selected_ranks(reference, selected_ranks)
                else:
                    base_output = model(
                        batch.features, batch.availability, evidence=reference
                    )
                    utilities = []
                    for rank in range(args.top_k):
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
                    utilities_by_rank = torch.stack(utilities, dim=-1)
                    selected_ranks, _ = select_strong_ranks(
                        utilities_by_rank,
                        batch.availability,
                        batch.sequence_mask,
                        reference.mask,
                    )
                    evidence = apply_selected_ranks(reference, selected_ranks)

            coverage.add(evidence, batch.availability, batch.sequence_mask)
            trace_mode = (
                cli.evidence_mode
                if cli.evidence_mode == "empty"
                else f"{cli.evidence_source}_{cli.evidence_mode}"
            )
            trace_rows.extend(
                evidence_trace_rows(
                    evidence,
                    batch.availability,
                    batch.sequence_mask,
                    split=cli.split,
                    sample_ids=batch.sample_ids,
                    utterance_ids=batch.utterance_ids,
                    evidence_mode=trace_mode,
                    mask_rate=args.mask_rate,
                    seed=args.seed,
                    sampler_version=args.missing_protocol,
                )
            )

            route = torch.full_like(
                torch.zeros(
                    batch.availability.size(0),
                    batch.availability.size(1),
                    3,
                    device=device,
                ),
                cli.fixed_source_weight,
            )
            output = model(
                batch.features,
                batch.availability,
                evidence=evidence,
                route_override=route,
            )
            task = "regression" if args.dataset in {"CMUMOSI", "CMUMOSEI"} else "classification"
            loss = masked_task_loss(
                output.logits, batch.labels, batch.sequence_mask, task
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite validation/test loss")
            labels_all.append(batch.labels.detach().cpu())
            sequence_all.append(batch.sequence_mask.detach().cpu())
            availability_all.append(batch.availability.detach().cpu())
            prediction = (
                output.logits.squeeze(-1)
                if task == "regression"
                else output.logits.argmax(dim=-1)
            )
            predictions_all.append(prediction.detach().cpu())

            if reference is not None and selected_ranks is not None:
                rank_rows = selected_ranks.detach().cpu().tolist()
                valid_rows = reference.mask.any(dim=-1).detach().cpu().tolist()
                availability_rows = batch.availability.detach().cpu().tolist()
                sequence_rows = batch.sequence_mask.detach().cpu().tolist()
                for b, sample_id in enumerate(batch.sample_ids):
                    for t, utterance_id in enumerate(batch.utterance_ids[b]):
                        if not sequence_rows[b][t] or not valid_rows[b][t]:
                            continue
                        rank = int(rank_rows[b][t])
                        selected_rank_counts[str(rank)] = selected_rank_counts.get(str(rank), 0) + 1
                        selection_rows.append(
                            {
                                "batch": batch_index,
                                "split": cli.split,
                                "sample_id": sample_id,
                                "utterance_id": utterance_id,
                                "selected_rank": rank,
                                "available_ranks": int(reference.mask[b, t].sum().item()),
                                "missing_target_modalities": [
                                    MODALITIES[index]
                                    for index, value in enumerate(availability_rows[b][t])
                                    if not value
                                ],
                                "evidence_source": cli.evidence_source,
                                "uses_validation_labels": cli.evidence_mode == "oracle_strong",
                                "deployable": cli.evidence_mode == "oracle_weak",
                                "shared_across_target_modalities": True,
                            }
                        )

    labels = torch.cat(labels_all, dim=0)
    predictions = torch.cat(predictions_all, dim=0)
    sequence_mask = torch.cat(sequence_all, dim=0)
    metrics = _metrics(labels, predictions, sequence_mask, args.dataset)
    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_fields = list(trace_rows[0]) if trace_rows else ["split"]
    with (output_dir / "evidence_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trace_fields)
        writer.writeheader()
        writer.writerows(trace_rows)
    with (output_dir / "oracle_selection_trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in selection_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output_dir / "evidence_gate_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["evidence_mode", "split", *metrics.keys()])
        writer.writeheader()
        writer.writerow({"evidence_mode": cli.evidence_mode, "split": cli.split, **metrics})

    coverage_payload = {
        "split": cli.split,
        "evidence_mode": cli.evidence_mode,
        "evidence_source": cli.evidence_source,
        "evidence_source_version": (
            "feature_neighbor_v1"
            if cli.evidence_source == "feature"
            else "text_tfidf_ngram12_v1"
        ),
        "evidence_source_fingerprint": (
            retriever.source_fingerprint()
            if retriever is not None and cli.evidence_source == "text"
            else None
        ),
        "checkpoint": cli.student_checkpoint,
        "mask_manifest": cli.mask_manifest,
        "coverage": coverage.summary(),
    }
    (output_dir / "evidence_coverage.json").write_text(
        json.dumps(coverage_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = {
        "schema_version": 2,
        "evidence_mode": cli.evidence_mode,
        "evidence_source": cli.evidence_source,
        "evidence_source_version": (
            "feature_neighbor_v1"
            if cli.evidence_source == "feature"
            else "text_tfidf_ngram12_v1"
        ),
        "evidence_source_fingerprint": (
            retriever.source_fingerprint()
            if retriever is not None and cli.evidence_source == "text"
            else None
        ),
        "split": cli.split,
        "fixed_source_weight": cli.fixed_source_weight,
        "top_k": args.top_k,
        "deployable": cli.evidence_mode != "oracle_strong",
        "uses_validation_labels": cli.evidence_mode == "oracle_strong",
        "diagnostic_only": cli.evidence_mode == "oracle_strong",
        "shared_across_target_modalities": True,
        "student_checkpoint": cli.student_checkpoint,
        "student_checkpoint_sha256": sha256_file(cli.student_checkpoint),
        "teacher_checkpoint": cli.teacher_checkpoint,
        "teacher_checkpoint_sha256": (
            sha256_file(cli.teacher_checkpoint) if cli.teacher_checkpoint else None
        ),
        "mask_manifest": cli.mask_manifest,
        "mask_manifest_sha256": sha256_file(cli.mask_manifest),
        "metrics": metrics,
        "metrics_by_missing_pattern_and_target": grouped_metrics(
            labels, predictions, sequence_mask,
            torch.cat(availability_all, dim=0), args.dataset, _metrics,
        ),
        "coverage": coverage.summary(),
        "selected_rank_counts": selected_rank_counts,
        "selection_count": len(selection_rows),
    }
    (output_dir / "result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
