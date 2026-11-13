from __future__ import annotations

import argparse
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ComP.per.batch import PERDatasetAdapter, move_batch_to_device
from ComP.per.losses import build_route_target
from ComP.per.missing import MaskManifest
from ComP.per.teacher_diagnostics import (
    compute_route_quality,
    route_quality_groups,
    validate_route_selection,
)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-student", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--split", choices=("validation",), required=True)
    parser.add_argument("--mask-manifest", required=True)
    parser.add_argument("--route-override", choices=("oracle", "student", "fixed", "prompt"), required=True)
    parser.add_argument("--fixed-source-weight", type=float, default=0.5)
    parser.add_argument("--evidence-mode", choices=("empty", "feature", "random", "shuffled", "source_only"), default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    cli = parser.parse_args()
    validate_route_selection(cli.split, cli.route_override)

    manifest_path = Path(cli.mask_manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    checkpoint = torch.load(cli.checkpoint_student, map_location="cpu", weights_only=False)
    saved = dict(checkpoint["args"])
    device = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    saved.update({
        "no_cuda": device.type == "cpu",
        "gpu": cli.gpu,
        "device": device,
        "teacher_checkpoint": cli.teacher_checkpoint,
    })
    if cli.evidence_mode is not None:
        saved["evidence_mode"] = cli.evidence_mode
    args = Namespace(**saved)
    if not 0.0 <= cli.fixed_source_weight <= 1.0:
        raise ValueError("fixed source weight must be in [0,1]")
    _check_feature_paths(args)
    dataset, splits, _ = _dataset_and_splits(args)
    split = next(item for item in splits if int(item["fold"]) == 0)
    adim, tdim, vdim = dataset.get_featDim()
    train_adapter = PERDatasetAdapter(dataset, split["train"])
    val_adapter = PERDatasetAdapter(dataset, split["val"])
    train_loader = _make_loader(train_adapter, args, training=False)
    val_loader = _make_loader(val_adapter, args, training=False)
    evidence_dim = adim + tdim + vdim
    retriever = None
    if args.evidence_mode != "empty":
        retriever = _build_retriever(train_loader, adim, tdim, vdim, args.top_k)
        evidence_dim = retriever.evidence_dim
    model = _build_student(
        args, device, adim, tdim, vdim, evidence_dim
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    teacher = _load_teacher(args, device, adim, tdim, vdim, 0)
    manifest = MaskManifest.load(str(manifest_path))

    labels_all = []
    predictions_all = []
    sequence_all = []
    quality_rows = []
    group_rows = []
    with torch.no_grad():
        for batch_index, full_cpu in enumerate(val_loader):
            batch = _apply_manifest(full_cpu, manifest, device)
            full_batch = move_batch_to_device(full_cpu, device)
            evidence = _load_evidence(
                batch,
                retriever,
                args.evidence_mode,
                evidence_dim,
                args.top_k,
                seed=args.seed,
            )
            student_output = model(batch.features, batch.availability, evidence=evidence)
            scores = teacher.score_candidates(
                full_batch.features,
                batch.availability,
                student_output.prompt_candidates,
                student_output.evidence,
                batch.labels,
            )
            utility_gap = scores.utility[:, :, :, 1] - scores.utility[:, :, :, 0]
            target = build_route_target(
                scores.utility[:, :, :, 0],
                scores.utility[:, :, :, 1],
                ~batch.availability,
                torch.tensor(args.route_temperature, dtype=scores.utility.dtype, device=device),
                scores.confidence,
            )
            evidence_available = evidence.mask.any(dim=-1).unsqueeze(-1).expand(-1, -1, 3)
            quality = compute_route_quality(
                student_output.route, target, utility_gap,
                batch.availability, batch.sequence_mask, evidence_available,
            )
            quality_rows.append({"batch": batch_index, **quality})
            for row in route_quality_groups(
                student_output.route, target, utility_gap,
                batch.availability, batch.sequence_mask, evidence_available,
            ):
                group_rows.append({"batch": batch_index, **row})

            override = None
            if cli.route_override == "oracle":
                override = target
            elif cli.route_override == "fixed":
                override = torch.full_like(
                    student_output.route, cli.fixed_source_weight
                )
            elif cli.route_override == "prompt":
                override = torch.zeros_like(student_output.route)
            output = (
                student_output
                if override is None
                else model(batch.features, batch.availability, evidence=evidence, route_override=override)
            )
            loss = masked_task_loss(
                output.logits, batch.labels, batch.sequence_mask,
                "regression" if args.dataset in {"CMUMOSI", "CMUMOSEI"} else "classification",
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite validation loss")
            labels_all.append(batch.labels)
            sequence_all.append(batch.sequence_mask)
            if args.dataset in {"CMUMOSI", "CMUMOSEI"}:
                predictions_all.append(output.logits.squeeze(-1))
            else:
                predictions_all.append(output.logits.argmax(dim=-1))

    labels = torch.cat(labels_all, dim=0)
    predictions = torch.cat(predictions_all, dim=0)
    sequence_mask = torch.cat(sequence_all, dim=0)
    metrics = _metrics(labels, predictions, sequence_mask, args.dataset)
    output_dir = Path(cli.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "teacher_oracle.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["route_override", "deployable", *metrics.keys()])
        writer.writeheader()
        writer.writerow({
            "route_override": cli.route_override,
            "deployable": cli.route_override != "oracle",
            **metrics,
        })
    quality_fields = sorted({key for row in quality_rows for key in row})
    with (output_dir / "teacher_target_quality.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=quality_fields)
        writer.writeheader()
        writer.writerows(quality_rows)
    group_fields = sorted({key for row in group_rows for key in row})
    with (output_dir / "teacher_target_groups.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=group_fields)
        writer.writeheader()
        writer.writerows(group_rows)
    summary = {
        "route_override": cli.route_override,
        "fixed_source_weight": cli.fixed_source_weight,
        "evidence_mode": args.evidence_mode,
        "deployable": cli.route_override != "oracle",
        "uses_validation_labels": cli.route_override == "oracle",
        "split": "validation",
        "student_checkpoint": cli.checkpoint_student,
        "teacher_checkpoint": cli.teacher_checkpoint,
        "mask_manifest": cli.mask_manifest,
        "metrics": metrics,
    }
    (output_dir / "oracle_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
