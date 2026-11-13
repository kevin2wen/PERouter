from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.knowledge.encoder import RuleKnowledgeCardEncoder
from ComP.knowledge.pipeline import (
    KnowledgeBatchResult,
    knowledge_forward,
)
from ComP.per.batch import PERDatasetAdapter
from ComP.per.evidence import FeatureNeighborRetriever
from ComP.per.evidence_reranker import EvidenceReranker
from ComP.per.evidence_metrics import grouped_metrics
from ComP.per.losses import masked_task_loss
from ComP.per.missing import MaskManifest
from ComP.per.types import MODALITIES, EvidenceBatch
from ComP.knowledge.retriever import KnowledgeContextRetriever
from ComP.knowledge.utility import oracle_utility
from ComP.train_knowledge import _build_knowledge_student, build_pipeline
from ComP.train_per import (
    _apply_manifest,
    _build_student,
    _check_feature_paths,
    _dataset_and_splits,
    _make_loader,
    _metrics,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--student-checkpoint", required=True)
    p.add_argument("--card-adapter-checkpoint")
    p.add_argument("--mask-manifest", required=True)
    p.add_argument("--mode", choices=(
        "empty",
        "raw_feature",
        "card",
        "random_card",
        "shuffled_card",
        "source_only",
        "oracle_card",
        "confidence",
        "random_action",
        "listwise_filter",
        "pairwise_filter",
    ), required=True)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--knowledge-index")
    p.add_argument("--reranker-dir")
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--oracle-objective",
        choices=("task", "mae", "acc", "composite"),
        default="task",
        help="Objective used by --mode oracle_card",
    )
    p.add_argument("--composite-lambda-cls", type=float, default=0.5)
    p.add_argument("--composite-lambda-flip", type=float, default=0.5)
    return p


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_feature_retriever(train_loader, adim, tdim, vdim, top_k):
    retriever = FeatureNeighborRetriever(
        {"audio": adim, "text": tdim, "video": vdim},
        top_k=top_k,
        exclude_same_utterance=True,
        exclude_same_video=True,
        exclude_same_dialogue=True,
    )
    for batch in train_loader:
        for batch_index, sample_id in enumerate(batch.sample_ids):
            retriever.add(
                sample_id,
                batch.utterance_ids[batch_index],
                {
                    name: batch.features[name][batch_index]
                    for name in MODALITIES
                },
                batch.sequence_mask[batch_index],
            )
    return retriever


def load_rerankers(directory: Path, device):
    paths = sorted(directory.glob("reranker_*_fold*.pt"))
    if not paths:
        raise FileNotFoundError(f"no fold rerankers in {directory}")
    models = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = EvidenceReranker(
            payload["candidate_dim"],
            payload["prompt_dim"],
            payload["top_k"],
            hidden_dim=payload["hidden_dim"],
        ).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        models.append(model)
    return models


def clone_evidence(evidence: EvidenceBatch) -> EvidenceBatch:
    return EvidenceBatch(
        ids=[[list(rank_ids) for rank_ids in sample] for sample in evidence.ids],
        features=evidence.features.clone(),
        mask=evidence.mask.clone(),
        source_confidence=evidence.source_confidence.clone(),
        conflict_score=(
            evidence.conflict_score.clone()
            if evidence.conflict_score is not None
            else None
        ),
    )


def source_only(result: KnowledgeBatchResult) -> KnowledgeBatchResult:
    evidence = {}
    for target, value in result.evidence_by_target.items():
        clone = clone_evidence(value)
        clone.features.zero_()
        evidence[target] = clone
    return KnowledgeBatchResult(
        result.queries, result.contexts, result.cards, result.candidates, evidence
    )


def shuffled(result: KnowledgeBatchResult, seed: int) -> KnowledgeBatchResult:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    evidence = {}
    for target, value in result.evidence_by_target.items():
        clone = clone_evidence(value)
        batch_size, seq_len, top_k = clone.mask.shape
        permutation = torch.randperm(batch_size * seq_len, generator=generator)
        flat = clone.features.reshape(batch_size * seq_len, top_k, -1)
        clone.features = flat[permutation].reshape(
            batch_size, seq_len, top_k, -1
        )
        evidence[target] = clone
    return KnowledgeBatchResult(
        result.queries, result.contexts, result.cards, result.candidates, evidence
    )


def randomized(
    result: KnowledgeBatchResult,
    seed: int,
) -> KnowledgeBatchResult:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    pools = []
    for value in result.evidence_by_target.values():
        if value.mask.any():
            pools.append(value.features[value.mask])
    pool = torch.cat(pools, dim=0) if pools else None
    evidence = {}
    for target, value in result.evidence_by_target.items():
        clone = clone_evidence(value)
        if pool is not None:
            indices = torch.randint(
                0, pool.size(0), (int(clone.mask.sum()),), generator=generator
            )
            clone.features[clone.mask] = pool[indices].to(clone.features)
        evidence[target] = clone
    return KnowledgeBatchResult(
        result.queries, result.contexts, result.cards, result.candidates, evidence
    )


def isolate_rank(result: KnowledgeBatchResult, target: str, rank: int):
    evidence = {name: clone_evidence(value) for name, value in result.evidence_by_target.items()}
    value = evidence[target]
    mask = torch.zeros_like(value.mask)
    mask[:, :, rank] = value.mask[:, :, rank]
    value.mask = mask
    value.source_confidence = value.source_confidence * mask.to(
        value.source_confidence.dtype
    )
    if value.conflict_score is not None:
        value.conflict_score = value.conflict_score * mask.to(
            value.conflict_score.dtype
        )
    return KnowledgeBatchResult(
        result.queries, result.contexts, result.cards, result.candidates, evidence
    )


def element_loss(output, labels, task):
    if task == "regression":
        return (output.logits.squeeze(-1) - labels).pow(2)
    return torch.nn.functional.cross_entropy(
        output.logits.reshape(-1, output.logits.size(-1)),
        labels.reshape(-1).long(),
        reduction="none",
    ).reshape(labels.shape)


def select_result(
    model,
    pipeline,
    batch,
    result,
    *,
    method: str,
    rerankers,
    seed: int,
):
    empty_vectors = {
        name: torch.zeros(
            *batch.sequence_mask.shape,
            model.dim,
            dtype=batch.features[MODALITIES[0]].dtype,
            device=batch.features[MODALITIES[0]].device,
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
    candidate_vectors = []
    prompt_vectors = []
    source_confidence = []
    target_modalities = []
    missing_patterns = []
    candidate_masks = []
    positions = []
    for target_index, target in enumerate(MODALITIES):
        evidence = result.evidence_by_target[target]
        projected = model.evidence_encoder.projections[target](
            evidence.features
        )
        for batch_index in range(batch.sequence_mask.size(0)):
            for time_index in range(batch.sequence_mask.size(1)):
                mask = evidence.mask[batch_index, time_index]
                if not bool(mask.any()):
                    continue
                positions.append((batch_index, time_index, target_index, target))
                candidate_vectors.append(
                    projected[batch_index, time_index]
                )
                prompt_vectors.append(
                    prompt_output.prompt_candidates[target][batch_index, time_index]
                )
                source_confidence.append(
                    evidence.source_confidence[batch_index, time_index]
                )
                target_modalities.append(target_index)
                missing_patterns.append(
                    batch.availability[batch_index, time_index].long()
                )
                candidate_masks.append(mask)
    if not positions:
        return source_only(result), []
    cv = torch.stack(candidate_vectors)
    pv = torch.stack(prompt_vectors)
    sc = torch.stack(source_confidence)
    tm = torch.tensor(
        target_modalities, dtype=torch.long, device=cv.device
    )
    mp = torch.stack(missing_patterns).to(cv.device)
    cm = torch.stack(candidate_masks).to(cv.device)
    if method in {"listwise_filter", "pairwise_filter"}:
        if not rerankers:
            raise ValueError("learned filter requires rerankers")
        with torch.no_grad():
            logits = torch.stack(
                [
                    reranker(cv, pv, sc, tm, mp, cm)
                    for reranker in rerankers
                ],
                dim=0,
            ).mean(dim=0)
        action = logits.argmax(dim=-1)
    elif method == "confidence":
        action = sc.masked_fill(~cm, float("-inf")).argmax(dim=-1) + 1
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        values = []
        for mask in cm.detach().cpu().tolist():
            legal = [0] + [
                rank + 1 for rank, valid in enumerate(mask) if valid
            ]
            values.append(legal[int(torch.randint(0, len(legal), (1,), generator=generator))])
        action = torch.tensor(values, dtype=torch.long, device=cv.device)
    evidence = {
        name: clone_evidence(value)
        for name, value in result.evidence_by_target.items()
    }
    for name in evidence:
        evidence[name].mask.zero_()
        evidence[name].source_confidence.zero_()
        if evidence[name].conflict_score is not None:
            evidence[name].conflict_score.zero_()
    action_rows = []
    for row_index, (batch_index, time_index, target_index, target) in enumerate(positions):
        action_index = int(action[row_index].item())
        if action_index > 0:
            rank = action_index - 1
            source = result.evidence_by_target[target]
            evidence[target].mask[batch_index, time_index, rank] = True
            evidence[target].source_confidence[
                batch_index, time_index, rank
            ] = source.source_confidence[batch_index, time_index, rank]
            if evidence[target].conflict_score is not None:
                evidence[target].conflict_score[
                    batch_index, time_index, rank
                ] = result.evidence_by_target[target].conflict_score[
                    batch_index, time_index, rank
                ]
        action_rows.append(
            {
                "sample_id": batch.sample_ids[batch_index],
                "utterance_id": batch.utterance_ids[batch_index][time_index],
                "target_modality": target,
                "action": action_index,
                "candidate_rank": action_index - 1 if action_index else -1,
                "use_evidence": action_index > 0,
            }
        )
    return KnowledgeBatchResult(
        result.queries, result.contexts, result.cards, result.candidates, evidence
    ), action_rows


def oracle_card(
    model,
    batch,
    result,
    *,
    gamma: float,
    task: str,
    objective: str = "task",
    lambda_cls: float = 0.5,
    lambda_flip: float = 0.5,
) -> tuple[
    KnowledgeBatchResult,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    with torch.no_grad():
        skip_output = model(
            batch.features,
            batch.availability,
            evidence=None,
            return_intermediates=True,
            route_override=torch.zeros_like(batch.availability).to(
                batch.features[MODALITIES[0]].dtype
            ),
            evidence_vectors_override={
                name: torch.zeros(
                    *batch.sequence_mask.shape,
                    model.dim,
                    dtype=batch.features[MODALITIES[0]].dtype,
                    device=batch.features[MODALITIES[0]].device,
                )
                for name in MODALITIES
            },
        )
        skip_prediction = (
            skip_output.logits.squeeze(-1)
            if task == "regression"
            else skip_output.logits.argmax(dim=-1)
        )

        def utility(output):
            if objective == "task":
                return -element_loss(output, batch.labels, task)
            prediction = (
                output.logits.squeeze(-1)
                if task == "regression"
                else output.logits.argmax(dim=-1)
            )
            return oracle_utility(
                prediction,
                batch.labels,
                skip_prediction,
                task=task,
                objective=objective,
                lambda_cls=lambda_cls,
                lambda_flip=lambda_flip,
            )

        skip_utility = utility(skip_output)
        selected = {}
        best_rank = {}
        for target in MODALITIES:
            best_gain = torch.zeros_like(skip_utility)
            best = torch.full_like(batch.labels, -1, dtype=torch.long)
            for rank in range(result.evidence_by_target[target].mask.size(-1)):
                isolated = isolate_rank(result, target, rank)
                output = knowledge_forward(
                    model, batch, isolated, gamma=gamma
                )
                gain = utility(output) - skip_utility
                eligible = (
                    isolated.evidence_by_target[target].mask[:, :, rank]
                    & (~batch.availability[:, :, MODALITIES.index(target)])
                )
                better = eligible & (gain > best_gain)
                best_gain = torch.where(better, gain, best_gain)
                best = torch.where(better, torch.full_like(best, rank), best)
            selected[target] = best
            best_rank[target] = best_gain
        chosen = {}
        for target, value in result.evidence_by_target.items():
            clone = clone_evidence(value)
            chosen_mask = torch.zeros_like(clone.mask)
            rank_values = selected[target]
            for rank in range(clone.mask.size(-1)):
                selected_here = (rank_values == rank) & clone.mask[:, :, rank]
                chosen_mask[:, :, rank] = selected_here
            clone.mask = chosen_mask
            clone.source_confidence = clone.source_confidence * chosen_mask.to(
                clone.source_confidence.dtype
            )
            if clone.conflict_score is not None:
                clone.conflict_score = clone.conflict_score * chosen_mask.to(
                    clone.conflict_score.dtype
                )
            chosen[target] = clone
        return (
            KnowledgeBatchResult(
                result.queries,
                result.contexts,
                result.cards,
                result.candidates,
                chosen,
            ),
            best_rank,
            selected,
        )


def main() -> int:
    cli = parser().parse_args()
    if cli.split == "test" and not cli.allow_test:
        raise PermissionError("test requires --allow-test after configs are frozen")
    if cli.mode == "oracle_card" and cli.split != "validation":
        raise PermissionError("oracle_card is validation-only")
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
    if cli.top_k is not None:
        args.top_k = int(cli.top_k)
    _check_feature_paths(args)
    dataset, splits, _ = _dataset_and_splits(args)
    split = next(item for item in splits if int(item["fold"]) == 0)
    split_key = "val" if cli.split == "validation" else "test"
    adim, tdim, vdim = dataset.get_featDim()
    train_loader = _make_loader(
        PERDatasetAdapter(dataset, split["train"]), args, training=False
    )
    target_loader = _make_loader(
        PERDatasetAdapter(dataset, split[split_key]), args, training=False
    )
    evidence_dim = adim + tdim + vdim
    model = _build_knowledge_student(
        args, device, adim, tdim, vdim, evidence_dim, student
    )
    model.eval()
    pipeline = None
    retriever = None
    adapter_payload = None
    if cli.card_adapter_checkpoint and Path(cli.card_adapter_checkpoint).exists():
        adapter_payload = torch.load(
            cli.card_adapter_checkpoint, map_location="cpu", weights_only=False
        )
        encoder_state = adapter_payload.get("evidence_encoder")
        if encoder_state is not None:
            model.evidence_encoder.load_state_dict(encoder_state, strict=True)
    card_modes = {
        "card",
        "random_card",
        "shuffled_card",
        "source_only",
        "oracle_card",
        "confidence",
        "random_action",
        "listwise_filter",
        "pairwise_filter",
    }
    rerankers = []
    if cli.mode in card_modes:
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
        if not cli.card_adapter_checkpoint:
            raise ValueError("card modes require --card-adapter-checkpoint")
        if adapter_payload is None:
            adapter_payload = torch.load(
                cli.card_adapter_checkpoint, map_location="cpu", weights_only=False
            )
        projector.load_state_dict(adapter_payload["projector"], strict=True)
        projector.to(device)
        projector.eval()
        if cli.mode in {"listwise_filter", "pairwise_filter"}:
            if not cli.reranker_dir:
                raise ValueError("learned filter modes require --reranker-dir")
            rerankers = load_rerankers(Path(cli.reranker_dir), device)
    elif cli.mode == "raw_feature":
        retriever = clean_feature_retriever(
            train_loader, adim, tdim, vdim, args.top_k
        )
    manifest = MaskManifest.load(cli.mask_manifest)
    task = "regression" if args.dataset in {"CMUMOSI", "CMUMOSEI"} else "classification"
    labels_all = []
    predictions_all = []
    sequence_all = []
    availability_all = []
    losses = []
    oracle_rows = []
    action_rows = []
    regression = task == "regression"
    per_sample_rows = []
    with torch.no_grad():
        for batch_index, full_cpu in enumerate(target_loader):
            batch = _apply_manifest(full_cpu, manifest, device)
            if cli.mode == "empty":
                output = model(
                    batch.features,
                    batch.availability,
                    evidence=None,
                    return_intermediates=True,
                    route_override=torch.zeros_like(batch.availability).to(
                        batch.features[MODALITIES[0]].dtype
                    ),
                )
            elif cli.mode == "raw_feature":
                evidence = retriever.retrieve(
                    batch.features,
                    batch.availability,
                    batch.sequence_mask,
                    batch.sample_ids,
                    batch.utterance_ids,
                )
                output = model(
                    batch.features,
                    batch.availability,
                    evidence=evidence,
                    return_intermediates=True,
                    route_override=torch.full_like(
                        batch.availability,
                        cli.gamma,
                        dtype=batch.features[MODALITIES[0]].dtype,
                    ),
                )
            else:
                result = pipeline.build(batch)
                if cli.mode == "random_card":
                    result = randomized(result, cli.seed + batch_index)
                elif cli.mode == "shuffled_card":
                    result = shuffled(result, cli.seed + batch_index)
                elif cli.mode == "source_only":
                    result = source_only(result)
                elif cli.mode == "oracle_card":
                    result, gains, _ = oracle_card(
                        model,
                        batch,
                        result,
                        gamma=cli.gamma,
                        task=task,
                        objective=cli.oracle_objective,
                        lambda_cls=cli.composite_lambda_cls,
                        lambda_flip=cli.composite_lambda_flip,
                    )
                    for target, gain in gains.items():
                        active = (~batch.availability[:, :, MODALITIES.index(target)])
                        oracle_rows.append(
                            {
                                "selected_gain_mean": float(gain[active].mean())
                                if active.any()
                                else 0.0,
                                "target_modality": target,
                            }
                        )
                elif cli.mode in {
                    "confidence",
                    "random_action",
                    "listwise_filter",
                    "pairwise_filter",
                }:
                    result, selected_rows = select_result(
                        model,
                        pipeline,
                        batch,
                        result,
                        method=cli.mode,
                        rerankers=rerankers,
                        seed=cli.seed + batch_index,
                    )
                    action_rows.extend(selected_rows)
                output = knowledge_forward(
                    model,
                    batch,
                    result,
                    gamma=cli.gamma,
                    zero_content=cli.mode == "source_only",
                )
            loss = masked_task_loss(
                output.logits, batch.labels, batch.sequence_mask, task
            )
            losses.append(float(loss.item()))
            labels_all.append(batch.labels.cpu())
            sequence_all.append(batch.sequence_mask.cpu())
            availability_all.append(batch.availability.cpu())
            batch_prediction = (
                output.logits.squeeze(-1)
                if regression
                else output.logits.argmax(dim=-1)
            ).detach().cpu()
            batch_labels = batch.labels.detach().cpu()
            batch_mask = batch.sequence_mask.detach().cpu()
            batch_availability = batch.availability.detach().cpu()
            for batch_row in range(batch_labels.size(0)):
                for time_index in range(batch_labels.size(1)):
                    if not bool(batch_mask[batch_row, time_index]):
                        continue
                    label_value = float(batch_labels[batch_row, time_index])
                    if regression and label_value == 0.0:
                        continue
                    prediction = float(batch_prediction[batch_row, time_index])
                    available = [
                        bool(value)
                        for value in batch_availability[
                            batch_row, time_index
                        ].tolist()
                    ]
                    sample_id = batch.sample_ids[batch_row]
                    per_sample_rows.append(
                        {
                            "mode": cli.mode,
                            "gamma": cli.gamma,
                            "split": cli.split,
                            "sample_id": sample_id,
                            "dialogue_id": sample_id,
                            "utterance_id": batch.utterance_ids[batch_row][
                                time_index
                            ],
                            "availability": "".join(
                                "1" if value else "0" for value in available
                            ),
                            "missing_count": 3 - sum(available),
                            "label": label_value,
                            "prediction": prediction,
                            "abs_error": abs(prediction - label_value),
                            "correct": int(
                                (prediction > 0) == (label_value > 0)
                            )
                            if regression
                            else int(prediction == label_value),
                        }
                    )
            predictions_all.append(
                batch_prediction
            )
            if cli.max_batches and batch_index + 1 >= cli.max_batches:
                break
    labels = torch.cat(labels_all)
    predictions = torch.cat(predictions_all)
    sequence_mask = torch.cat(sequence_all)
    availability = torch.cat(availability_all)
    metrics = _metrics(labels, predictions, sequence_mask, args.dataset)
    metrics["loss"] = sum(losses) / max(1, len(losses))
    grouped = grouped_metrics(
        labels,
        predictions,
        sequence_mask,
        availability,
        args.dataset,
        _metrics,
    )
    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_sample.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = (
            "mode",
            "gamma",
            "split",
            "sample_id",
            "dialogue_id",
            "utterance_id",
            "availability",
            "missing_count",
            "label",
            "prediction",
            "abs_error",
            "correct",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_sample_rows)
    result = {
        "split": cli.split,
        "mode": cli.mode,
        "gamma": cli.gamma,
        "oracle_objective": cli.oracle_objective,
        "metrics": metrics,
        "grouped_metrics": grouped,
        "student_checkpoint": cli.student_checkpoint,
        "student_checkpoint_sha256": sha256_file(cli.student_checkpoint),
        "card_adapter_checkpoint": cli.card_adapter_checkpoint,
        "card_adapter_sha256": (
            sha256_file(cli.card_adapter_checkpoint)
            if cli.card_adapter_checkpoint
            else None
        ),
        "mask_manifest": cli.mask_manifest,
        "mask_sha256": sha256_file(cli.mask_manifest),
        "teacher_used_at_inference": False,
        "source_split": "train",
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if oracle_rows:
        with (output_dir / "oracle_actions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("target_modality", "selected_gain_mean")
            )
            writer.writeheader()
            writer.writerows(oracle_rows)
    if action_rows:
        with (output_dir / "action_rows.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "sample_id",
                    "utterance_id",
                    "target_modality",
                    "action",
                    "candidate_rank",
                    "use_evidence",
                ),
            )
            writer.writeheader()
            writer.writerows(action_rows)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
