from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ComP.per.batch import PERDatasetAdapter, move_batch_to_device
from ComP.per.evidence_adapter import load_adapter_checkpoint
from ComP.per.evidence_oracle import isolate_rank_evidence
from ComP.per.evidence_oracle_v2 import (
    GAMMA_VALUES,
    OracleAction,
    action_grid,
    action_histogram,
    action_parameters,
    select_actions,
    validate_oracle_v2_split,
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


def element_task_loss(logits, labels, task):
    if task == "regression":
        return (logits.squeeze(-1) - labels).pow(2)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1).long(), reduction="none"
    ).reshape(labels.shape)


def build_vectors_for_action(action, rank_vectors, prompt_candidates, device):
    vectors = {}
    for modality in MODALITIES:
        if action.use_evidence:
            vectors[modality] = rank_vectors[action.candidate_rank][modality]
        else:
            vectors[modality] = prompt_candidates[modality]
    return vectors


def selected_plan(selected, actions, rank_vectors, prompt_candidates):
    ranks, gammas, use = action_parameters(
        selected, selected.new_ones(selected.shape, dtype=torch.bool), actions
    )
    vectors = {}
    for modality_index, modality in enumerate(MODALITIES):
        stacked = torch.stack([vectors_by_rank[modality] for vectors_by_rank in rank_vectors], dim=0)
        ranks_m = ranks[..., modality_index]
        use_m = use[..., modality_index]
        b, t = ranks_m.shape
        dim = stacked.size(-1)
        safe = ranks_m.clamp_min(0).unsqueeze(0).reshape(1, b, t, 1)
        gather = safe.expand(1, b, t, dim)
        chosen = stacked.gather(0, gather).squeeze(0)
        vectors[modality] = torch.where(use_m.unsqueeze(-1), chosen, prompt_candidates[modality])
    return vectors, gammas.to(dtype=prompt_candidates[MODALITIES[0]].dtype), ranks, use


def main() -> int:
    cli = parser().parse_args()
    validate_oracle_v2_split(cli.split)
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
    actions = action_grid(args.top_k)
    action_lookup = {action: index for index, action in enumerate(actions)}
    skip_index = 0

    teacher_predictions = []
    task_predictions = []
    labels_all = []
    sequence_all = []
    teacher_selected_all = []
    task_selected_all = []
    active_all = []
    teacher_trace = []
    task_trace = []

    with torch.no_grad():
        for batch_index, full_cpu in enumerate(val_loader):
            batch = _apply_manifest(full_cpu, manifest, device)
            full_batch = move_batch_to_device(full_cpu, device)
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
            base_output = model(batch.features, batch.availability, evidence=reference)
            prompt_candidates = base_output.prompt_candidates
            rank_vectors = []
            for rank in range(args.top_k):
                probe = isolate_rank_evidence(reference, rank)
                rank_vectors.append(
                    model.evidence_encoder(probe, batch.availability, ~batch.availability)
                )
            active = (
                (~batch.availability)
                & batch.sequence_mask.unsqueeze(-1)
                & reference.mask.any(dim=-1, keepdim=True)
            )

            B, T = batch.sequence_mask.shape
            A = len(actions)
            teacher_scores = torch.empty(B, T, 3, A, device=device)
            task_scores = torch.empty(B, T, 3, A, device=device)
            for action_index, action in enumerate(actions):
                vectors = build_vectors_for_action(
                    action, rank_vectors, prompt_candidates, device
                )
                route = torch.full(
                    (B, T, 3), action.gamma, dtype=prompt_candidates[MODALITIES[0]].dtype, device=device
                )
                output = model(
                    batch.features,
                    batch.availability,
                    evidence=reference,
                    route_override=route,
                    evidence_vectors_override=vectors,
                )
                element_loss = element_task_loss(output.logits, batch.labels, task)
                task_scores[..., action_index] = (-element_loss).unsqueeze(-1).expand(B, T, 3)
                if action.use_evidence:
                    mixed = {
                        modality: prompt_candidates[modality]
                        + action.gamma * (rank_vectors[action.candidate_rank][modality] - prompt_candidates[modality])
                        for modality in MODALITIES
                    }
                else:
                    mixed = dict(prompt_candidates)
                scores = teacher.score_candidates(
                    full_batch.features,
                    batch.availability,
                    prompt_candidates,
                    mixed,
                    batch.labels,
                )
                teacher_scores[..., action_index] = scores.utility[..., 1]

            teacher_selected = select_actions(teacher_scores, active, A)
            task_selected = select_actions(task_scores, active, A)
            teacher_vectors, teacher_gammas, teacher_ranks, teacher_use = selected_plan(
                teacher_selected, actions, rank_vectors, prompt_candidates
            )
            task_vectors, task_gammas, task_ranks, task_use = selected_plan(
                task_selected, actions, rank_vectors, prompt_candidates
            )
            teacher_output = model(
                batch.features, batch.availability, evidence=reference,
                route_override=teacher_gammas, evidence_vectors_override=teacher_vectors,
            )
            task_output = model(
                batch.features, batch.availability, evidence=reference,
                route_override=task_gammas, evidence_vectors_override=task_vectors,
            )
            labels_all.append(batch.labels.detach().cpu())
            sequence_all.append(batch.sequence_mask.detach().cpu())
            prediction_teacher = teacher_output.logits.squeeze(-1) if task == "regression" else teacher_output.logits.argmax(-1)
            prediction_task = task_output.logits.squeeze(-1) if task == "regression" else task_output.logits.argmax(-1)
            teacher_predictions.append(prediction_teacher.detach().cpu())
            task_predictions.append(prediction_task.detach().cpu())
            teacher_selected_all.append(teacher_selected.detach().cpu())
            task_selected_all.append(task_selected.detach().cpu())
            active_all.append(active.detach().cpu())

            for planner, selected, ranks, gammas, use in (
                ("oracle_teacher_v2", teacher_selected, teacher_ranks, teacher_gammas, teacher_use),
                ("oracle_task_v2", task_selected, task_ranks, task_gammas, task_use),
            ):
                target_rows = teacher_trace if planner.startswith("teacher") else task_trace
                for b, sample_id in enumerate(batch.sample_ids):
                    for t, utterance_id in enumerate(batch.utterance_ids[b]):
                        if not bool(batch.sequence_mask[b, t]):
                            continue
                        for m, modality in enumerate(MODALITIES):
                            if not bool(active[b, t, m]):
                                continue
                            action = actions[int(selected[b, t, m].item())]
                            target_rows.append({
                                "planner": planner,
                                "sample_id": sample_id,
                                "utterance_id": utterance_id,
                                "target_modality": modality,
                                "use_evidence": bool(use[b, t, m]),
                                "selected_rank": int(ranks[b, t, m]),
                                "selected_gamma": float(gammas[b, t, m]),
                                "action_key": (
                                    "skip_evidence" if not action.use_evidence
                                    else f"rank{action.candidate_rank}_gamma{action.gamma:g}"
                                ),
                            })

    labels = torch.cat(labels_all, dim=0)
    seq = torch.cat(sequence_all, dim=0)
    teacher_pred = torch.cat(teacher_predictions, dim=0)
    task_pred = torch.cat(task_predictions, dim=0)
    teacher_selected = torch.cat(teacher_selected_all, dim=0)
    task_selected = torch.cat(task_selected_all, dim=0)
    active = torch.cat(active_all, dim=0)
    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "oracle_v2_selection_trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in teacher_trace + task_trace:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "schema_version": 1,
        "split": cli.split,
        "test_access": "forbidden",
        "evidence_source": cli.evidence_source,
        "deployable": False,
        "uses_validation_labels": True,
        "student_checkpoint_sha256": sha256_file(cli.student_checkpoint),
        "teacher_checkpoint_sha256": sha256_file(cli.teacher_checkpoint),
        "mask_manifest_sha256": sha256_file(cli.mask_manifest),
        "top_k": args.top_k,
        "gamma_values": list(GAMMA_VALUES),
        "oracle_teacher_metrics": _metrics(labels, teacher_pred, seq, args.dataset),
        "oracle_task_metrics": _metrics(labels, task_pred, seq, args.dataset),
        "oracle_teacher_actions": action_histogram(teacher_selected, active, actions),
        "oracle_task_actions": action_histogram(task_selected, active, actions),
        "selection_method": "homogeneous-action-counterfactual",
    }
    (output_dir / "oracle_v2_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
