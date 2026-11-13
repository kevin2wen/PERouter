from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ComP.per.batch import PERDatasetAdapter, move_batch_to_device
from ComP.per.evidence import randomize_evidence, shuffled_evidence, source_only_evidence
from ComP.per.evidence_adapter import load_adapter_checkpoint
from ComP.per.evidence_metrics import grouped_metrics
from ComP.per.evidence_oracle import isolate_rank_evidence
from ComP.per.evidence_reranker import EvidenceReranker
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
    p.add_argument("--adapter-checkpoint")
    p.add_argument("--mask-manifest", required=True)
    p.add_argument("--evidence-source", choices=("feature", "text"), default="feature")
    p.add_argument("--evidence-mode", choices=("empty", "feature", "random", "shuffled", "source_only"), required=True)
    p.add_argument("--method", choices=("raw", "confidence", "random_action", "listwise", "pairwise"), required=True)
    p.add_argument("--reranker-dir")
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    return p


def load_rerankers(directory: Path, device):
    paths = sorted(directory.glob("reranker_*_fold*.pt"))
    if not paths:
        raise FileNotFoundError(f"no fold rerankers in {directory}")
    models = []
    metadata = []
    expected = None
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = EvidenceReranker(
            payload["candidate_dim"], payload["prompt_dim"], payload["top_k"],
            hidden_dim=payload["hidden_dim"],
        ).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        identity = (
            payload["source"], payload["objective"], payload["top_k"],
            payload["candidate_dim"], payload["prompt_dim"],
        )
        if expected is None:
            expected = identity
        elif identity != expected:
            raise ValueError("reranker fold metadata mismatch")
        models.append(model)
        metadata.append({"path": str(path), "sha256": sha256_file(str(path)), **payload["metrics"]})
    return models, {"paths": paths, "folds": metadata, "identity": expected}


def evaluate_outputs(logits, labels, sequence_mask, availability, dataset, task):
    prediction = logits.squeeze(-1) if task == "regression" else logits.argmax(dim=-1)
    metrics = _metrics(labels, prediction, sequence_mask, dataset)
    grouped = grouped_metrics(
        labels, prediction, sequence_mask, availability, dataset, _metrics
    )
    return metrics, grouped


def main() -> int:
    cli = parser().parse_args()
    if cli.method in {"listwise", "pairwise"} and not cli.reranker_dir:
        raise ValueError("learned methods require --reranker-dir")
    manifest = MaskManifest.load(cli.mask_manifest)
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
            top_k=args.top_k, device=torch.device("cpu"),
        )
        retriever.add_loader(train_loader)
    else:
        retriever = _build_retriever(train_loader, adim, tdim, vdim, args.top_k)
    model = _build_student(args, device, adim, tdim, vdim, evidence_dim)
    model.load_state_dict(payload["model"], strict=True)
    if cli.adapter_checkpoint:
        load_adapter_checkpoint(model, cli.adapter_checkpoint, expected_source=cli.evidence_source)
    model.eval()
    rerankers = []
    reranker_metadata = None
    if cli.method in {"listwise", "pairwise"}:
        rerankers, reranker_metadata = load_rerankers(Path(cli.reranker_dir), device)
        if reranker_metadata["identity"][1] != cli.method:
            raise ValueError("reranker objective does not match --method")
    task = "regression" if args.dataset in {"CMUMOSI", "CMUMOSEI"} else "classification"
    predictions = []
    labels_all = []
    sequence_all = []
    availability_all = []
    action_rows = []
    random.seed(cli.seed)

    with torch.no_grad():
        for batch_index, full_cpu in enumerate(val_loader):
            batch = _apply_manifest(full_cpu, manifest, device)
            if cli.evidence_mode == "empty":
                evidence = _load_evidence(batch, None, "empty", evidence_dim, args.top_k, seed=args.seed)
            elif cli.evidence_mode == "feature":
                evidence = (
                    retriever.retrieve(
                        batch.features, batch.availability, batch.sequence_mask,
                        batch.sample_ids, batch.utterance_ids, raw_texts=batch.raw_texts,
                    ) if cli.evidence_source == "text"
                    else _load_evidence(batch, retriever, "feature", evidence_dim, args.top_k, seed=args.seed)
                )
            else:
                reference = (
                    retriever.retrieve(
                        batch.features, batch.availability, batch.sequence_mask,
                        batch.sample_ids, batch.utterance_ids, raw_texts=batch.raw_texts,
                    ) if cli.evidence_source == "text"
                    else _load_evidence(batch, retriever, "feature", evidence_dim, args.top_k, seed=args.seed)
                )
                if cli.evidence_mode == "random":
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
                else:
                    evidence = source_only_evidence(reference)

            base_output = model(batch.features, batch.availability, evidence=evidence)
            prompt = base_output.prompt_candidates
            rank_vectors = []
            for rank in range(args.top_k):
                probe = isolate_rank_evidence(evidence, rank)
                rank_vectors.append(model.evidence_encoder(probe, batch.availability, ~batch.availability))
            active = (
                (~batch.availability) & batch.sequence_mask.unsqueeze(-1)
                & evidence.mask.any(dim=-1, keepdim=True)
            )
            B, T = batch.sequence_mask.shape
            selected_action = torch.full((B, T, 3), -1, dtype=torch.long, device=device)

            if cli.method == "raw":
                output = model(batch.features, batch.availability, evidence=evidence,
                               route_override=torch.full((B, T, 3), 0.5, device=device))
            else:
                positions = []
                candidate_vectors = []
                prompt_vectors = []
                source_confidence = []
                target_modalities = []
                missing_patterns = []
                candidate_mask = []
                for b in range(B):
                    for t in range(T):
                        if not bool(batch.sequence_mask[b, t]):
                            continue
                        pattern = batch.availability[b, t]
                        for m, modality in enumerate(MODALITIES):
                            if not bool(active[b, t, m]):
                                continue
                            positions.append((b, t, m))
                            candidate_vectors.append(torch.stack([
                                rank_vectors[rank][modality][b, t] for rank in range(args.top_k)
                            ], dim=0))
                            prompt_vectors.append(prompt[modality][b, t])
                            source_confidence.append(evidence.source_confidence[b, t])
                            target_modalities.append(m)
                            missing_patterns.append(pattern.long())
                            candidate_mask.append(evidence.mask[b, t])
                if not positions:
                    vectors = {modality: prompt[modality] for modality in MODALITIES}
                    route = torch.zeros((B, T, 3), device=device)
                    output = model(
                        batch.features, batch.availability, evidence=evidence,
                        route_override=route, evidence_vectors_override=vectors,
                    )
                else:
                    candidate_mask_t = torch.stack(candidate_mask)
                    if cli.method in {"listwise", "pairwise"}:
                        batch_logits = []
                        cv = torch.stack(candidate_vectors).to(device)
                        pv = torch.stack(prompt_vectors).to(device)
                        sc = torch.stack(source_confidence).to(device)
                        tm = torch.tensor(target_modalities, dtype=torch.long, device=device)
                        mp = torch.stack(missing_patterns).to(device).long()
                        cm = candidate_mask_t.to(device)
                        for reranker in rerankers:
                            batch_logits.append(reranker(cv, pv, sc, tm, mp, cm))
                        action = torch.stack(batch_logits).mean(0).argmax(dim=-1)
                    elif cli.method == "confidence":
                        cm = candidate_mask_t.to(device)
                        confidence_t = torch.stack(source_confidence).to(device)
                        ranks = confidence_t.masked_fill(~cm, -float("inf")).argmax(dim=-1) + 1
                        rows = torch.arange(cm.size(0), device=device)
                        action = ranks[rows]
                    else:
                        values = []
                        for mask in candidate_mask_t.tolist():
                            legal = [0] + [rank + 1 for rank, valid in enumerate(mask) if valid]
                            values.append(random.choice(legal))
                        action = torch.tensor(values, dtype=torch.long, device=device)

                    for row_index, (b, t, m) in enumerate(positions):
                        action_index = int(action[row_index].item())
                        selected_action[b, t, m] = action_index
                        action_rows.append({
                            "batch": batch_index,
                            "split": "validation",
                            "sample_id": batch.sample_ids[b],
                            "utterance_id": batch.utterance_ids[b][t],
                            "target_modality": MODALITIES[m],
                            "method": cli.method,
                            "evidence_mode": cli.evidence_mode,
                            "candidate_rank": action_index - 1 if action_index > 0 else -1,
                            "use_evidence": action_index > 0,
                            "gamma": 0.5 if action_index > 0 else 0.0,
                        })
                    vectors = {}
                    route = torch.zeros((B, T, 3), device=device)
                    for m, modality in enumerate(MODALITIES):
                        rank = selected_action[:, :, m] - 1
                        use = selected_action[:, :, m] > 0
                        stacked = torch.stack([rv[modality] for rv in rank_vectors], dim=0)
                        safe = rank.clamp_min(0).unsqueeze(0)
                        gather = safe.reshape(1, B, T, 1).expand(1, B, T, stacked.size(-1))
                        chosen = stacked.gather(0, gather).squeeze(0)
                        vectors[modality] = torch.where(use.unsqueeze(-1), chosen, prompt[modality])
                        route[:, :, m] = use.to(route.dtype) * 0.5
                    output = model(
                        batch.features, batch.availability, evidence=evidence,
                        route_override=route, evidence_vectors_override=vectors,
                    )
            loss = masked_task_loss(output.logits, batch.labels, batch.sequence_mask, task)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite validation loss")
            prediction = output.logits.squeeze(-1) if task == "regression" else output.logits.argmax(dim=-1)
            predictions.append(prediction.detach().cpu())
            labels_all.append(batch.labels.detach().cpu())
            sequence_all.append(batch.sequence_mask.detach().cpu())
            availability_all.append(batch.availability.detach().cpu())

    labels = torch.cat(labels_all, dim=0)
    sequence = torch.cat(sequence_all, dim=0)
    availability = torch.cat(availability_all, dim=0)
    prediction = torch.cat(predictions, dim=0)
    metrics = _metrics(labels, prediction, sequence, args.dataset)
    grouped = grouped_metrics(labels, prediction, sequence, availability, args.dataset, _metrics)
    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "reranker_actions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in action_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(action_rows)
    summary = {
        "schema_version": 1,
        "split": "validation",
        "test_access": "forbidden",
        "source": cli.evidence_source,
        "method": cli.method,
        "evidence_mode": cli.evidence_mode,
        "student_checkpoint_sha256": sha256_file(cli.student_checkpoint),
        "adapter_checkpoint_sha256": sha256_file(cli.adapter_checkpoint) if cli.adapter_checkpoint else None,
        "mask_manifest_sha256": sha256_file(cli.mask_manifest),
        "reranker": (
            {"directory": cli.reranker_dir, **reranker_metadata}
            if reranker_metadata is not None else None
        ),
        "metrics": metrics,
        "metrics_by_missing_pattern_and_target": grouped,
        "use_evidence_rate": (
            sum(row["use_evidence"] for row in action_rows) / len(action_rows)
            if action_rows else 0.0
        ),
        "skip_rate": (
            sum(not row["use_evidence"] for row in action_rows) / len(action_rows)
            if action_rows else 0.0
        ),
        "action_count": len(action_rows),
    }
    (output_dir / "result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
