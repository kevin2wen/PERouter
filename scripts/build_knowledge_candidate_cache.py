from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.knowledge.audit import stable_hash
from ComP.knowledge.filter_features import (
    FEATURE_DIM,
    card_feature_vector,
    card_index,
)
from ComP.knowledge.pipeline import (
    KnowledgeBatchResult,
    knowledge_forward,
)
from ComP.knowledge.retriever import KnowledgeContextRetriever
from ComP.per.batch import PERDatasetAdapter
from ComP.per.missing import MaskManifest
from ComP.per.types import MODALITIES
from ComP.train_knowledge import _build_knowledge_student, build_pipeline
from ComP.train_per import (
    _apply_manifest,
    _build_student,
    _check_feature_paths,
    _dataset_and_splits,
    _make_loader,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--student-checkpoint", required=True)
    p.add_argument("--card-adapter-checkpoint", required=True)
    p.add_argument("--knowledge-index")
    p.add_argument("--mask-manifest", required=True)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--min-helpful-gain", type=float, default=0.0)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--structured-features", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", required=True)
    return p


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def isolate_rank(result, target, rank):
    evidence = {}
    for name, value in result.evidence_by_target.items():
        mask = torch.zeros_like(value.mask)
        if name == target:
            mask[:, :, rank] = value.mask[:, :, rank]
        evidence[name] = type(value)(
            ids=[[list(row) for row in sample] for sample in value.ids],
            features=value.features.clone(),
            mask=mask,
            source_confidence=value.source_confidence
            * mask.to(value.source_confidence.dtype),
            conflict_score=(
                value.conflict_score * mask.to(value.conflict_score.dtype)
                if value.conflict_score is not None
                else None
            ),
        )
    return KnowledgeBatchResult(
        result.queries, result.contexts, result.cards, result.candidates, evidence
    )


def element_loss(output, labels, task):
    if task == "regression":
        return (output.logits.squeeze(-1) - labels).pow(2)
    return F.cross_entropy(
        output.logits.reshape(-1, output.logits.size(-1)),
        labels.reshape(-1).long(),
        reduction="none",
    ).reshape(labels.shape)


def main() -> int:
    cli = parser().parse_args()
    student = torch.load(
        cli.student_checkpoint, map_location="cpu", weights_only=False
    )
    saved = dict(student["args"])
    device = torch.device(
        f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu"
    )
    saved.update(
        {"no_cuda": device.type == "cpu", "gpu": cli.gpu, "device": device}
    )
    args = Namespace(**saved)
    _check_feature_paths(args)
    dataset, splits, _ = _dataset_and_splits(args)
    split = next(item for item in splits if int(item["fold"]) == 0)
    adim, tdim, vdim = dataset.get_featDim()
    train_loader = _make_loader(
        PERDatasetAdapter(dataset, split["train"]), args, training=False
    )
    evidence_dim = adim + tdim + vdim
    bank = (
        KnowledgeContextRetriever.load(cli.knowledge_index)
        if cli.knowledge_index and Path(cli.knowledge_index).exists()
        else None
    )
    pipeline, projector, bank = build_pipeline(
        train_loader,
        adim=adim,
        tdim=tdim,
        vdim=vdim,
        evidence_dim=evidence_dim,
        top_k=args.top_k,
        seed=args.seed,
        bank=bank,
    )
    adapter = torch.load(
        cli.card_adapter_checkpoint, map_location="cpu", weights_only=False
    )
    projector.load_state_dict(adapter["projector"], strict=True)
    projector.to(device)
    projector.eval()
    model = _build_knowledge_student(
        args, device, adim, tdim, vdim, evidence_dim, student
    )
    encoder_state = adapter.get("evidence_encoder")
    if encoder_state is None:
        raise ValueError(
            "card adapter checkpoint has no evidence_encoder state"
        )
    model.evidence_encoder.load_state_dict(encoder_state, strict=True)
    model.eval()
    model.requires_grad_(False)
    manifest = MaskManifest.load(cli.mask_manifest)
    task = "regression" if args.dataset in {"CMUMOSI", "CMUMOSEI"} else "classification"
    records = []
    candidate_vectors = []
    prompt_vectors = []
    source_confidence = []
    candidate_mask_rows = []
    utility_rows = []
    skip_utility = []
    target_labels = []
    target_modalities = []
    missing_patterns = []
    candidate_ids = []
    seen = 0
    with torch.no_grad():
        for full_cpu in train_loader:
            batch = _apply_manifest(full_cpu, manifest, device)
            result = pipeline.build(batch)
            card_map = card_index(result.cards)
            empty_vectors = {
                name: torch.zeros(
                    *batch.sequence_mask.shape,
                    model.dim,
                    dtype=batch.features[MODALITIES[0]].dtype,
                    device=device,
                )
                for name in MODALITIES
            }
            prompt_output = model(
                batch.features,
                batch.availability,
                evidence=None,
                return_intermediates=True,
                route_override=torch.zeros_like(batch.availability).to(
                    batch.features[MODALITIES[0]].dtype
                ),
                evidence_vectors_override=empty_vectors,
            )
            skip_loss = element_loss(prompt_output, batch.labels, task)
            utilities = {
                name: torch.full(
                    (
                        *batch.sequence_mask.shape,
                        result.evidence_by_target[name].mask.size(-1),
                    ),
                    float("-inf"),
                    device=device,
                )
                for name in MODALITIES
            }
            projected_by_target = {}
            for target_index, target in enumerate(MODALITIES):
                evidence = result.evidence_by_target[target]
                projected = model.evidence_encoder.projections[target](
                    evidence.features
                )
                projected_by_target[target] = projected
                for rank in range(args.top_k):
                    isolated = isolate_rank(result, target, rank)
                    output = knowledge_forward(
                        model, batch, isolated, gamma=cli.gamma
                    )
                    rank_gain = (
                        skip_loss - element_loss(output, batch.labels, task)
                    )
                    utilities[target][:, :, rank] = torch.where(
                        evidence.mask[:, :, rank],
                        rank_gain,
                        torch.full_like(rank_gain, float("-inf")),
                    )
                for batch_index in range(batch.sequence_mask.size(0)):
                    for time_index in range(batch.sequence_mask.size(1)):
                        if not bool(batch.sequence_mask[batch_index, time_index]):
                            continue
                        if bool(batch.availability[batch_index, time_index, target_index]):
                            continue
                        mask = evidence.mask[batch_index, time_index]
                        if not bool(mask.any()):
                            continue
                        records.append(
                            {
                                "sample_id": batch.sample_ids[batch_index],
                                "utterance_id": batch.utterance_ids[batch_index][
                                    time_index
                                ],
                                "target_modality": target_index,
                                "missing_pattern": "".join(
                                    "0" if bool(value) else "1"
                                    for value in batch.availability[
                                        batch_index, time_index
                                    ].tolist()
                                ),
                            }
                        )
                        if not cli.structured_features:
                            candidate_vectors.append(
                                projected_by_target[target][
                                    batch_index, time_index
                                ].detach().cpu()
                            )
                        else:
                            cards = card_map
                            rows = []
                            for rank in range(args.top_k):
                                card = cards.get(
                                    str(
                                        evidence.ids[batch_index][
                                            time_index
                                        ][rank]
                                    )
                                )
                                if card is None:
                                    rows.append(
                                        [0.0] * FEATURE_DIM
                                    )
                                    continue
                                rows.append(
                                    card_feature_vector(
                                        card,
                                        target=target,
                                        missing_pattern=(
                                            batch.availability[
                                                batch_index, time_index
                                            ].tolist()
                                        ),
                                        rank=rank,
                                        top_k=args.top_k,
                                        conflict_score=(
                                            float(
                                                evidence.conflict_score[
                                                    batch_index,
                                                    time_index,
                                                    rank,
                                                ]
                                            )
                                            if evidence.conflict_score
                                            is not None
                                            else 0.0
                                        ),
                                    )
                                )
                            candidate_vectors.append(
                                torch.tensor(
                                    rows, dtype=torch.float32
                                )
                            )
                        prompt_vectors.append(
                            prompt_output.prompt_candidates[target][
                                batch_index, time_index
                            ].detach().cpu()
                        )
                        source_confidence.append(
                            evidence.source_confidence[
                                batch_index, time_index
                            ].detach().cpu()
                        )
                        candidate_mask_rows.append(mask.detach().cpu())
                        utility_rows.append(
                            utilities[target][
                                batch_index, time_index
                            ].detach().cpu()
                        )
                        skip_utility.append(
                            float(-skip_loss[batch_index, time_index].item())
                        )
                        target_labels.append(
                            float(batch.labels[batch_index, time_index].item())
                        )
                        target_modalities.append(target_index)
                        missing_patterns.append(
                            "".join(
                                "0" if bool(value) else "1"
                                for value in batch.availability[
                                    batch_index, time_index
                                ].tolist()
                            )
                        )
                        candidate_ids.append(
                            list(evidence.ids[batch_index][time_index])
                        )
            if cli.max_batches and seen + 1 >= cli.max_batches:
                break
            seen += 1
    if not records:
        raise RuntimeError("no active knowledge candidates were generated")
    candidate_tensor = torch.stack(candidate_vectors)
    prompt_tensor = torch.stack(prompt_vectors)
    confidence_tensor = torch.stack(source_confidence)
    candidate_mask = torch.stack(candidate_mask_rows)
    utility_tensor = torch.stack(utility_rows)
    skip_utility_tensor = torch.tensor(skip_utility, dtype=torch.float32)
    delta = utility_tensor - skip_utility_tensor.unsqueeze(-1)
    valid_delta = delta.masked_fill(
        ~candidate_mask, torch.finfo(delta.dtype).min
    )
    best_delta, best_rank = valid_delta.max(dim=-1)
    use = best_delta >= float(cli.min_helpful_gain)
    target_action = torch.where(
        use, best_rank + 1, torch.zeros_like(best_rank)
    )
    key_payload = {
        "dataset": args.dataset,
        "student_checkpoint_sha256": sha256_file(cli.student_checkpoint),
        "card_adapter_sha256": sha256_file(cli.card_adapter_checkpoint),
        "mask_sha256": sha256_file(cli.mask_manifest),
        "source_fingerprint": bank.source_fingerprint(),
        "top_k": args.top_k,
        "gamma": cli.gamma,
        "exclusion_policy": "utterance+video+dialogue",
        "schema_version": "knowledge-card-v1",
        "min_helpful_gain": cli.min_helpful_gain,
        "structured_features": bool(cli.structured_features),
    }
    cache_key = stable_hash(key_payload)
    cache = {
        "format_version": 1,
        "split": "train",
        "cache_key": cache_key,
        "key_payload": key_payload,
        "records": records,
        "records_sample_ids": [
            record["sample_id"] for record in records
        ],
        "candidate_vectors": candidate_tensor,
        "prompt_vectors": prompt_tensor,
        "source_confidence": confidence_tensor,
        "candidate_mask": candidate_mask,
        "utility": utility_tensor,
        "skip_utility": skip_utility_tensor,
        "target_labels": torch.tensor(target_labels, dtype=torch.float32),
        "target_modalities": torch.tensor(target_modalities, dtype=torch.long),
        "missing_patterns": list(missing_patterns),
        "candidate_ids": candidate_ids,
        "target_action": target_action,
        "min_helpful_gain": cli.min_helpful_gain,
    }
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(cache, temporary)
    temporary.replace(output)
    manifest_data = {
        "cache_key": cache_key,
        "record_count": len(records),
        "candidate_dim": int(candidate_tensor.size(-1)),
        "prompt_dim": int(prompt_tensor.size(-1)),
        "use_evidence_rate": float(use.float().mean().item()),
        "cache_sha256": sha256_file(output),
        "source_split": "train",
        "test_accessed": False,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest_data, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
