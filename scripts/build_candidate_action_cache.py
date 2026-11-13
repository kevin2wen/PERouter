from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ComP.per.batch import PERDatasetAdapter, move_batch_to_device
from ComP.per.candidate_actions import (
    make_candidate_cache_key,
    select_best_actions,
    validate_train_only_split,
)
from ComP.per.evidence_adapter import load_adapter_checkpoint
from ComP.per.evidence import FeatureNeighborRetriever
from ComP.per.evidence_oracle import isolate_rank_evidence
from ComP.per.missing import MaskManifest, MissingPatternSampler
from ComP.per.text_evidence import TextNeighborRetriever
from ComP.per.types import MODALITIES
from ComP.train_per import (
    _build_retriever,
    _build_student,
    _check_feature_paths,
    _dataset_and_splits,
    _load_evidence,
    _make_loader,
)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_hash(state: dict) -> str:
    buffer = io.BytesIO()
    torch.save({key: value.detach().cpu() for key, value in state.items()}, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def element_task_loss(logits, labels, task):
    if task == "regression":
        return (logits.squeeze(-1) - labels).pow(2)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1).long(), reduction="none"
    ).reshape(labels.shape)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--student-checkpoint", required=True)
    p.add_argument("--evidence-source", choices=("feature", "text"), required=True)
    p.add_argument("--adapter-checkpoint")
    p.add_argument("--min-helpful-gain", type=float, default=0.0)
    p.add_argument("--exclude-same-video", action="store_true")
    p.add_argument("--exclude-same-dialogue", action="store_true")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    return p


def main() -> int:
    cli = parser().parse_args()
    validate_train_only_split("train")
    if cli.min_helpful_gain < 0:
        raise ValueError("min-helpful-gain must be non-negative")
    payload = torch.load(cli.student_checkpoint, map_location="cpu", weights_only=False)
    saved = dict(payload["args"])
    device = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    saved.update({"no_cuda": device.type == "cpu", "gpu": cli.gpu, "device": device})
    args = Namespace(**saved)
    _check_feature_paths(args)
    dataset, splits, _ = _dataset_and_splits(args)
    split = next(item for item in splits if int(item["fold"]) == 0)
    adim, tdim, vdim = dataset.get_featDim()
    train_adapter = PERDatasetAdapter(dataset, split["train"])
    train_loader = _make_loader(train_adapter, args, training=False)
    evidence_dim = adim + tdim + vdim
    exclusion_policy = (
        'utterance+video+dialogue'
        if cli.exclude_same_video or cli.exclude_same_dialogue
        else 'utterance-only'
    )
    if cli.evidence_source == "text":
        retriever = TextNeighborRetriever(
            {"audio": adim, "text": tdim, "video": vdim},
            top_k=args.top_k,
            device=torch.device("cpu"),
            exclude_same_video=cli.exclude_same_video,
            exclude_same_dialogue=cli.exclude_same_dialogue,
        )
        retriever.add_loader(train_loader)
        evidence_source_hash = retriever.source_fingerprint()
    else:
        if cli.exclude_same_video or cli.exclude_same_dialogue:
            retriever = FeatureNeighborRetriever(
                {"audio": adim, "text": tdim, "video": vdim},
                top_k=args.top_k,
                device=torch.device("cpu"),
                exclude_same_utterance=True,
                exclude_same_video=cli.exclude_same_video,
                exclude_same_dialogue=cli.exclude_same_dialogue,
            )
            for train_batch in train_loader:
                for batch_index, sample_id in enumerate(train_batch.sample_ids):
                    retriever.add(
                        sample_id,
                        train_batch.utterance_ids[batch_index],
                        {name: train_batch.features[name][batch_index] for name in MODALITIES},
                        train_batch.sequence_mask[batch_index],
                    )
        else:
            retriever = _build_retriever(train_loader, adim, tdim, vdim, args.top_k)
        evidence_source_hash = hashlib.sha256(
            f"feature_neighbor_v1|top_k={args.top_k}|exclude={exclusion_policy}".encode("utf-8")
        ).hexdigest()

    model = _build_student(args, device, adim, tdim, vdim, evidence_dim)
    model.load_state_dict(payload["model"], strict=True)
    if cli.adapter_checkpoint:
        load_adapter_checkpoint(model, cli.adapter_checkpoint, expected_source=cli.evidence_source)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    sampler = MissingPatternSampler(
        seed=args.seed,
        missing_rate=args.mask_rate,
        protocol=args.missing_protocol,
    )
    task = "regression" if args.dataset in {"CMUMOSI", "CMUMOSEI"} else "classification"
    records = []
    candidate_vectors_all = []
    prompt_vectors_all = []
    confidence_all = []
    candidate_mask_all = []
    utility_all = []
    skip_utility_all = []
    target_labels_all = []
    target_modalities_all = []
    missing_patterns_all = []
    mask_sample_ids = []
    mask_utterance_ids = []
    mask_sequence = []
    mask_availability = []

    with torch.no_grad():
        for batch_index, full_cpu in enumerate(train_loader):
            batch = move_batch_to_device(full_cpu, device)
            availability = sampler.sample(batch.sample_ids, batch.sequence_mask, epoch=0).to(device)
            batch.availability = availability
            mask_sample_ids.extend(batch.sample_ids)
            mask_utterance_ids.extend(batch.utterance_ids)
            mask_sequence.append(batch.sequence_mask.detach().cpu())
            mask_availability.append(availability.detach().cpu())

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
            prompt_vectors = base_output.prompt_candidates
            rank_vectors = []
            for rank in range(args.top_k):
                probe = isolate_rank_evidence(reference, rank)
                rank_vectors.append(model.evidence_encoder(probe, batch.availability, ~batch.availability))

            batch_size, seq_len = batch.sequence_mask.shape
            prompt_route = torch.zeros(batch_size, seq_len, 3, device=device)
            skip_output = model(
                batch.features,
                batch.availability,
                evidence=reference,
                route_override=prompt_route,
                evidence_vectors_override=prompt_vectors,
            )
            skip_loss = element_task_loss(skip_output.logits, batch.labels, task)
            utility = torch.empty(batch_size, seq_len, 3, args.top_k, device=device)
            for modality_index in range(3):
                for rank in range(args.top_k):
                    vectors = dict(prompt_vectors)
                    vectors[MODALITIES[modality_index]] = rank_vectors[rank][MODALITIES[modality_index]]
                    route = torch.zeros(batch_size, seq_len, 3, device=device)
                    active_target = (~batch.availability[:, :, modality_index]) & batch.sequence_mask
                    route[:, :, modality_index] = active_target.to(route.dtype) * 0.5
                    output = model(
                        batch.features,
                        batch.availability,
                        evidence=reference,
                        route_override=route,
                        evidence_vectors_override=vectors,
                    )
                    loss = element_task_loss(output.logits, batch.labels, task)
                    utility[:, :, modality_index, rank] = -loss

            active = (
                (~batch.availability)
                & batch.sequence_mask.unsqueeze(-1)
                & reference.mask.any(dim=-1, keepdim=True)
            )
            skip_utility = (-skip_loss).unsqueeze(-1).expand(-1, -1, 3)
            selected = select_best_actions(
                utility, skip_utility, active, cli.min_helpful_gain
            )
            for b, sample_id in enumerate(batch.sample_ids):
                for t, utterance_id in enumerate(batch.utterance_ids[b]):
                    if not bool(batch.sequence_mask[b, t]):
                        continue
                    for modality_index, modality in enumerate(MODALITIES):
                        if not bool(active[b, t, modality_index]):
                            continue
                        rank_cos = [
                            float(F.cosine_similarity(
                                rank_vectors[rank][modality][b, t],
                                prompt_vectors[modality][b, t],
                                dim=0,
                            ).item())
                            for rank in range(args.top_k)
                        ]
                        candidate_vectors_all.append(
                            torch.stack([
                                rank_vectors[rank][modality][b, t].detach().cpu()
                                for rank in range(args.top_k)
                            ], dim=0)
                        )
                        prompt_vectors_all.append(prompt_vectors[modality][b, t].detach().cpu())
                        confidence_all.append(reference.source_confidence[b, t].detach().cpu())
                        candidate_mask_all.append(reference.mask[b, t].detach().cpu())
                        utility_all.append(utility[b, t, modality_index].detach().cpu())
                        skip_utility_all.append((-skip_loss[b, t]).detach().cpu())
                        target_labels_all.append(batch.labels[b, t].detach().cpu())
                        target_modalities_all.append(modality_index)
                        missing_patterns_all.append(
                            "".join("1" if bool(value) else "0" for value in batch.availability[b, t].detach().cpu())
                        )
                        records.append({
                            "batch": batch_index,
                            "split": "train",
                            "sample_id": sample_id,
                            "dialogue_id": sample_id,
                            "utterance_id": utterance_id,
                            "availability": [bool(v) for v in batch.availability[b, t].detach().cpu()],
                            "missing_pattern": missing_patterns_all[-1],
                            "target_modality": modality,
                            "target_modality_index": modality_index,
                            "evidence_mode": cli.evidence_source,
                            "candidate_ids": list(reference.ids[b][t]),
                            "candidate_ranks": list(range(args.top_k)),
                            "candidate_mask": reference.mask[b, t].detach().cpu().tolist(),
                            "source_confidence": reference.source_confidence[b, t].detach().cpu().tolist(),
                            "candidate_prompt_cosine": rank_cos,
                            "utility": utility[b, t, modality_index].detach().cpu().tolist(),
                            "skip_utility": float((-skip_loss[b, t]).item()),
                            "delta_utility": selected["delta_utility"][b, t, modality_index].detach().cpu().tolist()
                                if selected["delta_utility"][b, t, modality_index].ndim == 0
                                else float(selected["delta_utility"][b, t, modality_index].item()),
                            "best_rank": int(selected["best_rank"][b, t, modality_index].item()),
                            "use_evidence": bool(selected["use_evidence"][b, t, modality_index].item()),
                            "selected_rank": int(selected["selected_rank"][b, t, modality_index].item()),
                            "min_helpful_gain": cli.min_helpful_gain,
        "exclusion_policy": exclusion_policy,
                            "gamma": 0.5,
                        })

    output = Path(cli.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mask_path = output / "train_mask_manifest.csv"
    sequence_mask = torch.cat(mask_sequence, dim=0)
    availability = torch.cat(mask_availability, dim=0)
    MaskManifest.save(
        str(mask_path),
        mask_sample_ids,
        mask_utterance_ids,
        availability,
        sequence_mask,
        seed=args.seed,
        mask_rate=args.mask_rate,
        sampler_version=args.missing_protocol,
    )
    train_split_hash = hashlib.sha256(
        json.dumps(sorted(set(mask_sample_ids))).encode("utf-8")
    ).hexdigest()
    encoder_hash = tensor_state_hash(model.evidence_encoder.state_dict())
    key = make_candidate_cache_key(
        dataset_version=f"{args.dataset}:CMUMOSI_features_raw_2way",
        train_split_hash=train_split_hash,
        student_checkpoint_hash=sha256_file(cli.student_checkpoint),
        adapter_checkpoint_hash=(sha256_file(cli.adapter_checkpoint) if cli.adapter_checkpoint else None),
        evidence_source=cli.evidence_source,
        evidence_source_hash=evidence_source_hash,
        evidence_encoder_hash=encoder_hash,
        mask_manifest_hash=sha256_file(str(mask_path)),
        top_k=args.top_k,
        gamma=0.5,
        evidence_mode=cli.evidence_source,
        sampler_version=args.missing_protocol,
        min_helpful_gain=cli.min_helpful_gain,
        exclusion_policy=exclusion_policy,
    )
    cache = {
        "schema_version": 1,
        "cache_key": key,
        "split": "train",
        "test_used": False,
        "validation_used": False,
        "teacher_used": False,
        "student_checkpoint": cli.student_checkpoint,
        "adapter_checkpoint": cli.adapter_checkpoint,
        "evidence_source": cli.evidence_source,
        "min_helpful_gain": cli.min_helpful_gain,
        "exclusion_policy": exclusion_policy,
        "candidate_vectors": torch.stack(candidate_vectors_all),
        "prompt_vectors": torch.stack(prompt_vectors_all),
        "source_confidence": torch.stack(confidence_all),
        "candidate_mask": torch.stack(candidate_mask_all),
        "utility": torch.stack(utility_all),
        "skip_utility": torch.stack(skip_utility_all),
        "target_labels": torch.stack(target_labels_all),
        "target_modalities": torch.tensor(target_modalities_all, dtype=torch.long),
        "missing_patterns": missing_patterns_all,
        "records": records,
    }
    cache_path = output / "candidate_action_cache.pt"
    torch.save(cache, cache_path)
    with (output / "candidate_action_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    metadata = {
        "cache_key": key,
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(str(cache_path)),
        "mask_manifest": str(mask_path),
        "mask_manifest_sha256": sha256_file(str(mask_path)),
        "train_split_hash": train_split_hash,
        "evidence_source_hash": evidence_source_hash,
        "evidence_encoder_hash": encoder_hash,
        "student_checkpoint_sha256": sha256_file(cli.student_checkpoint),
        "adapter_checkpoint_sha256": sha256_file(cli.adapter_checkpoint) if cli.adapter_checkpoint else None,
        "top_k": args.top_k,
        "gamma": 0.5,
        "min_helpful_gain": cli.min_helpful_gain,
        "records": len(records),
        "use_evidence_rate": sum(row["use_evidence"] for row in records) / max(1, len(records)),
        "skip_rate": sum(not row["use_evidence"] for row in records) / max(1, len(records)),
    }
    (output / "cache_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
