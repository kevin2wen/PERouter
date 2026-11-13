from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader, Sampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from ComP.dataloader_cmumosi import CMUMOSIDataset
from ComP.dataloader_iemocap import IEMOCAPDataset
from ComP.loss import DisentangledLoss
from ComP.model import ComP
from ComP.per.batch import (
    PERDatasetAdapter,
    apply_availability,
    collate_per,
    move_batch_to_device,
)
from ComP.per.evidence import (
    FeatureNeighborRetriever,
    empty_evidence,
    perturb_evidence,
    randomize_evidence,
    shuffled_evidence,
    source_only_evidence,
)
from ComP.per.diagnostics import RouteDiagnosticsAccumulator
from ComP.per.evidence_audit import EvidenceCoverageAccumulator
from ComP.per.evaluation_audit import EvaluationAuditAccumulator
from ComP.per.fusion_diagnostics import FusionDiagnosticsAccumulator
from ComP.per.gradient_diagnostics import (
    component_gradient_norms,
    parameter_grad_norm,
)
from ComP.per.teacher_diagnostics import pearson, spearman
from ComP.per.losses import (
    masked_reconstruction_loss,
    build_route_target,
    masked_task_loss,
    per_loss,
)
from ComP.per.missing import MaskManifest, MissingPatternSampler
from ComP.per.manifest_utils import resolve_manifest_path, sha256_file
from ComP.per.model import PERStudent
from ComP.per.teacher import CompleteViewTeacher
from ComP.per.types import MODALITIES, EvidenceBatch, PERBatch, PEROutput


LEGACY_PROTOCOL = "legacy-compat"
FAIR_PROTOCOL = "fair"
PER_MODES = ("off", "generated", "fixed", "task_router", "teacher_router")


class EpochRandomSampler(Sampler[int]):
    def __init__(self, length: int, seed: int):
        self.length = int(length)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + 104729 * self.epoch)
        return iter(torch.randperm(self.length, generator=generator).tolist())

    def __len__(self) -> int:
        return self.length


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _task_name(dataset: str) -> str:
    return "regression" if dataset in {"CMUMOSI", "CMUMOSEI"} else "classification"


def _dataset_and_splits(args) -> Tuple[object, List[Dict[str, object]], int]:
    if args.dataset in {"CMUMOSI", "CMUMOSEI"}:
        dataset = CMUMOSIDataset(
            label_path=config.PATH_TO_LABEL[args.dataset],
            audio_root=os.path.join(config.PATH_TO_FEATURES[args.dataset], args.audio_feature),
            text_root=os.path.join(config.PATH_TO_FEATURES[args.dataset], args.text_feature),
            video_root=os.path.join(config.PATH_TO_FEATURES[args.dataset], args.video_feature),
        )
        train_n = len(dataset.trainVids)
        val_n = len(dataset.valVids)
        test_n = len(dataset.testVids)
        train_indices = list(range(train_n))
        val_indices = list(range(train_n, train_n + val_n))
        test_indices = list(range(train_n + val_n, train_n + val_n + test_n))
        if not val_indices:
            raise ValueError("CMU validation split is empty")
        return dataset, [
            {
                "fold": 0,
                "train": train_indices,
                "val": val_indices,
                "test": test_indices,
            }
        ], 1

    if args.dataset in {"IEMOCAPFour", "IEMOCAPSix"}:
        dataset = IEMOCAPDataset(
            label_path=config.PATH_TO_LABEL[args.dataset],
            audio_root=os.path.join(config.PATH_TO_FEATURES[args.dataset], args.audio_feature),
            text_root=os.path.join(config.PATH_TO_FEATURES[args.dataset], args.text_feature),
            video_root=os.path.join(config.PATH_TO_FEATURES[args.dataset], args.video_feature),
        )
        by_session: Dict[int, List[int]] = {}
        for index, video_id in enumerate(dataset.vids):
            session = int(video_id[4]) - 1
            by_session.setdefault(session, []).append(index)
        if len(by_session) != 5:
            raise ValueError(f"expected five IEMOCAP sessions, got {len(by_session)}")
        splits = []
        for test_session in range(5):
            test_indices = by_session[test_session]
            remaining = [
                index
                for session in range(5)
                if session != test_session
                for index in by_session[session]
            ]
            val_indices, train_indices = _dialogue_validation_split(
                dataset, remaining, args.val_ratio, args.seed + test_session
            )
            splits.append(
                {
                    "fold": test_session,
                    "train": train_indices,
                    "val": val_indices,
                    "test": test_indices,
                }
            )
        return dataset, splits, 5
    raise ValueError(f"unsupported dataset: {args.dataset}")


def _dialogue_validation_split(
    dataset,
    indices: Sequence[int],
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be in (0,1)")
    groups: Dict[str, List[int]] = {}
    for index in indices:
        groups.setdefault(str(dataset.vids[index]), []).append(index)
    if len(groups) < 2:
        raise ValueError("need at least two training dialogues for validation")
    ordered = sorted(
        groups,
        key=lambda vid: hashlib.sha256(f"{seed}:{vid}".encode("utf-8")).hexdigest(),
    )
    target = min(len(ordered) - 1, max(1, int(round(len(ordered) * val_ratio))))
    val_groups = set(ordered[:target])
    val_indices = [
        index for group in val_groups for index in groups[group]
    ]
    train_indices = [
        index for group, values in groups.items() if group not in val_groups for index in values
    ]
    return sorted(val_indices), sorted(train_indices)


def _check_feature_paths(args) -> None:
    roots = [
        os.path.join(config.PATH_TO_FEATURES[args.dataset], value)
        for value in (args.audio_feature, args.text_feature, args.video_feature)
    ]
    if not all(os.path.exists(root) for root in roots):
        raise FileNotFoundError(
            "ComP feature directories do not exist; update config.py paths or "
            f"feature names: {roots}"
        )
    if not os.path.exists(config.PATH_TO_LABEL[args.dataset]):
        raise FileNotFoundError(config.PATH_TO_LABEL[args.dataset])


def _make_loader(
    adapter: PERDatasetAdapter,
    args,
    *,
    training: bool,
    sampler: Optional[Sampler[int]] = None,
) -> DataLoader:
    if sampler is None:
        sampler = (
            EpochRandomSampler(len(adapter), args.seed)
            if training
            else torch.utils.data.SequentialSampler(adapter)
        )
    return DataLoader(
        adapter,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=lambda batch: collate_per(
            batch,
            {"sequence": args.seq_len},
            args.top_k,
            training=training,
        ),
        num_workers=args.num_workers,
        pin_memory=False,
    )


def _collect_split_masks(
    loader: DataLoader,
    sampler: MissingPatternSampler,
    epoch: int,
    output_path: Optional[str],
    *,
    fixed_pattern: Optional[Sequence[int]] = None,
    fixed_manifest: Optional[
        Dict[Tuple[str, int], Tuple[bool, bool, bool]]
    ] = None,
) -> Dict[Tuple[str, int], Tuple[bool, bool, bool]]:
    if fixed_manifest is not None:
        return fixed_manifest
    sample_ids: List[str] = []
    utterance_ids: List[List[str]] = []
    sequence_masks = []
    availability_rows = []
    for batch in loader:
        if fixed_pattern is None:
            current = sampler.sample(batch.sample_ids, batch.sequence_mask, epoch)
        else:
            current = sampler.fixed(
                batch.sample_ids, tuple(fixed_pattern), batch.sequence_mask
            )
        sample_ids.extend(batch.sample_ids)
        utterance_ids.extend(batch.utterance_ids)
        sequence_masks.append(batch.sequence_mask)
        availability_rows.append(current)
    sequence_mask = torch.cat(sequence_masks, dim=0)
    availability = torch.cat(availability_rows, dim=0)
    if output_path is not None:
        MaskManifest.save(
            output_path,
            sample_ids,
            utterance_ids,
            availability,
            sequence_mask,
            seed=sampler.seed,
            mask_rate=sampler.missing_rate,
            sampler_version=(
                sampler.protocol
                if fixed_pattern is None
                else "fixed-" + "".join(str(value) for value in fixed_pattern)
            ),
        )
    return MaskManifest.load(output_path) if output_path is not None else {
        (sample_id, time_index): tuple(
            bool(value) for value in availability[batch_index, time_index].tolist()
        )
        for batch_index, sample_id in enumerate(sample_ids)
        for time_index in range(sequence_mask.shape[1])
        if bool(sequence_mask[batch_index, time_index])
    }


def _apply_manifest(
    batch: PERBatch,
    manifest: Dict[Tuple[str, int], Tuple[bool, bool, bool]],
    device: torch.device,
) -> PERBatch:
    availability = MaskManifest.to_availability(
        manifest, batch.sample_ids, batch.sequence_mask
    ).to(device)
    return apply_availability(move_batch_to_device(batch, device), availability)


def _load_evidence(
    batch: PERBatch,
    retriever: Optional[FeatureNeighborRetriever],
    evidence_mode: str,
    evidence_dim: int,
    top_k: int,
    *,
    seed: int = 0,
) -> EvidenceBatch:
    if evidence_mode == "empty":
        return empty_evidence(
            batch.sequence_mask.size(0),
            batch.sequence_mask.size(1),
            top_k,
            max(1, evidence_dim),
            device=batch.sequence_mask.device,
            dtype=batch.features[MODALITIES[0]].dtype,
        )
    if retriever is None:
        raise ValueError(f"evidence mode {evidence_mode} requires a retriever")
    reference = retriever.retrieve(
        batch.features,
        batch.availability,
        batch.sequence_mask,
        batch.sample_ids,
        batch.utterance_ids,
    )
    if evidence_mode == "feature":
        return reference
    if evidence_mode == "random":
        return randomize_evidence(
            reference,
            retriever,
            seed=seed,
            query_sample_ids=batch.sample_ids,
        )
    if evidence_mode == "shuffled":
        return shuffled_evidence(
            reference,
            seed=seed,
            query_sample_ids=batch.sample_ids,
        )
    if evidence_mode == "source_only":
        return source_only_evidence(reference)
    raise ValueError(f"unsupported evidence mode: {evidence_mode}")


def _build_retriever(
    train_loader: DataLoader,
    adim: int,
    tdim: int,
    vdim: int,
    top_k: int,
) -> FeatureNeighborRetriever:
    retriever = FeatureNeighborRetriever(
        {"audio": adim, "text": tdim, "video": vdim}, top_k=top_k
    )
    with torch.no_grad():
        for batch in train_loader:
            for batch_index, sample_id in enumerate(batch.sample_ids):
                retriever.add(
                    sample_id,
                    batch.utterance_ids[batch_index],
                    {
                        modality: batch.features[modality][batch_index]
                        for modality in MODALITIES
                    },
                    batch.sequence_mask[batch_index],
                )
    return retriever


def _split_logits(outputs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    private = outputs[10]
    n_classes = outputs[2].size(-1)
    return (
        private[:, :, :n_classes],
        private[:, :, n_classes:2 * n_classes],
        private[:, :, 2 * n_classes:],
    )


def _add_loss(registry: Dict[str, float], name: str, value: torch.Tensor) -> None:
    numeric = float(value.detach().item())
    if not math.isfinite(numeric):
        raise FloatingPointError(f"non-finite loss {name}: {numeric}")
    registry[name] = registry.get(name, 0.0) + numeric


def _stage1_loss(
    outputs,
    batch: PERBatch,
    task: str,
    args,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    registry: Dict[str, float] = {}
    losses = []
    modality_logits = {
        "audio": outputs[2],
        "text": outputs[3],
        "video": outputs[4],
    }
    for modality in MODALITIES:
        value = masked_task_loss(
            modality_logits[modality], batch.labels, batch.sequence_mask, task
        )
        _add_loss(registry, f"stage1_{modality}", value)
        losses.append(value)

    private_a, private_t, private_v = _split_logits(outputs)
    if task == "regression":
        noise = torch.randn_like(batch.labels).clamp(min=-3.0, max=3.0)
        for name, private in (
            ("audio", private_a),
            ("text", private_t),
            ("video", private_v),
        ):
            value = masked_task_loss(
                private, noise, batch.sequence_mask, task
            ) * args.private_noise_weight
            _add_loss(registry, f"stage1_private_noise_{name}", value)
            losses.append(value)
    else:
        all_ones = torch.ones_like(private_a)
        for name, private in (
            ("audio", private_a),
            ("text", private_t),
            ("video", private_v),
        ):
            value = masked_task_loss(
                all_ones - private, batch.labels, batch.sequence_mask, task
            ) * args.private_antilabel_weight
            _add_loss(registry, f"stage1_private_anti_{name}", value)
            losses.append(value)

    total = torch.stack(losses).sum()
    _add_loss(registry, "stage1_total", total)
    return total, registry


def _stage2_loss(
    output: PEROutput,
    batch: PERBatch,
    task: str,
    args,
    *,
    teacher_logits: Optional[torch.Tensor] = None,
    route_target: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if output.modality_logits is None:
        raise RuntimeError("ComP did not return modality logits")
    if not output.intermediates:
        raise RuntimeError("stage-2 PEROutput requires intermediates")
    evidence_available = None
    if batch.evidence is not None:
        evidence_available = batch.evidence.mask.any(dim=-1).unsqueeze(-1).expand(-1, -1, 3)
    denominators = per_loss(
        output,
        batch.labels,
        task,
        route_target,
        ~batch.availability,
        teacher_logits=teacher_logits,
        lambda_route=args.route_weight,
        lambda_kd=args.kd_weight,
        kd_temperature=args.kd_temperature,
        sequence_mask=batch.sequence_mask,
        route_objective=getattr(args, "route_objective", "bce"),
        evidence_available=evidence_available,
    )
    registry = {}
    total = denominators["task"]
    _add_loss(registry, "task_joint", denominators["task"])
    for modality in MODALITIES:
        value = masked_task_loss(
            output.modality_logits[modality],
            batch.labels,
            batch.sequence_mask,
            task,
        )
        _add_loss(registry, f"task_{modality}", value)
        total = total + value
    reconstruction = masked_reconstruction_loss(
        output.intermediates["proj_atv_rec"],
        output.intermediates["proj_atv"],
        batch.sequence_mask,
        batch.availability,
    )
    total = total + args.reconstruction_weight * reconstruction
    _add_loss(registry, "reconstruction", reconstruction)
    total = total + args.route_weight * denominators["route"]
    _add_loss(registry, "route", denominators["route"])
    total = total + args.kd_weight * denominators["kd"]
    _add_loss(registry, "kd", denominators["kd"])
    if args.disentangle_weight:
        shared = output.intermediates["shared_atv"]
        private = output.intermediates["private_atv"]
        dim = shared.size(-1) // 3
        shared_parts = [
            shared[:, :, index * dim:(index + 1) * dim] for index in range(3)
        ]
        private_parts = [
            private[:, :, index * dim:(index + 1) * dim] for index in range(3)
        ]
        criterion = DisentangledLoss()
        disentangle = torch.stack(
            [
                criterion.foward_2(
                    private_parts[i], private_parts[j], shared_parts[i], shared_parts[j],
                    batch.sequence_mask.to(torch.float32),
                )
                for i in range(3)
                for j in range(i + 1, 3)
            ]
        ).sum()
        total = total + args.disentangle_weight * disentangle
        _add_loss(registry, "disentangle", disentangle)
    _add_loss(registry, "total", total)
    components = {
        "task": denominators["task"],
        "route": denominators["route"],
        "kd": denominators["kd"],
    }
    return total, registry, components


def _build_teacher_model(args, adim: int, tdim: int, vdim: int) -> ComP:
    return ComP(
        args,
        adim,
        tdim,
        vdim,
        args.hidden,
        n_classes=args.n_classes,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=1,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        no_cuda=args.no_cuda,
        lbd=args.lbd,
    )


def _load_teacher(
    args,
    device: torch.device,
    adim: int,
    tdim: int,
    vdim: int,
    fold: int,
):
    if not args.teacher_checkpoint:
        raise ValueError(
            "--per-mode teacher_router requires --teacher-checkpoint; "
            "refusing to substitute the student for the complete-view teacher"
        )
    checkpoint_name = args.teacher_checkpoint.format(fold=fold)
    path = Path(checkpoint_name)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if isinstance(state, dict) and "backbone." in next(iter(state), ""):
        state = {key.replace("backbone.", "", 1): value for key, value in state.items()}
    model = _build_teacher_model(args, adim, tdim, vdim)
    model.load_state_dict(state, strict=True)
    model.to(device)
    teacher = CompleteViewTeacher(
        model,
        task=_task_name(args.dataset),
        dim=args.hidden,
        teacher_ckpt_hash=_sha256_file(str(path)),
    )
    return teacher


def _build_student(args, device: torch.device, adim: int, tdim: int, vdim: int, evidence_dim: int):
    use_router = args.per_mode in {"fixed", "task_router", "teacher_router"}
    model = PERStudent(
        args,
        adim,
        tdim,
        vdim,
        evidence_dim=max(1, evidence_dim),
        top_k=args.top_k,
        task=_task_name(args.dataset),
        use_router=use_router,
        router_hidden=args.router_hidden,
        router_dropout=args.router_dropout,
        fix_private_branch=args.fix_private_branch,
    ).to(device)
    return model


def _teacher_terms(
    teacher: CompleteViewTeacher,
    output: PEROutput,
    student_batch: PERBatch,
    full_batch: PERBatch,
    args,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    evidence = student_batch.evidence
    if evidence is None:
        raise RuntimeError("teacher-guided router requires an EvidenceBatch")
    scores = teacher.score_candidates(
        full_batch.features,
        student_batch.availability,
        output.prompt_candidates,
        output.evidence,
        student_batch.labels,
    )
    evidence_available = evidence.mask.any(dim=-1).unsqueeze(-1).expand(-1, -1, 3)
    target = build_route_target(
        scores.utility[:, :, :, 0],
        scores.utility[:, :, :, 1],
        ~student_batch.availability,
        torch.tensor(
            args.route_temperature,
            dtype=scores.utility.dtype,
            device=scores.utility.device,
        ),
        scores.confidence,
        evidence_available=evidence_available,
    )
    teacher_logits = teacher.full_view_logits(
        full_batch.features, full_batch.sequence_mask
    ) if args.kd_weight > 0 else None
    utility_gap = scores.utility[:, :, :, 1] - scores.utility[:, :, :, 0]
    return target, teacher_logits, utility_gap


def train_epoch(
    model: PERStudent,
    teacher: Optional[CompleteViewTeacher],
    loader: DataLoader,
    mask_manifest,
    evidence_mode: str,
    evidence_dim: int,
    retriever: Optional[FeatureNeighborRetriever],
    optimizer: torch.optim.Optimizer,
    args,
    device: torch.device,
    first_stage: bool,
    epoch: int,
    router_warmup_active: bool = False,
) -> Dict[str, float]:
    model.train()
    registry: Dict[str, float] = {}
    diagnostics = RouteDiagnosticsAccumulator()
    evidence_coverage = EvidenceCoverageAccumulator()
    grad_registry: Dict[str, float] = {}
    grad_diag_batches = 0
    seen_batches = 0
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + epoch * 1009)
    for full_batch_cpu in loader:
        full_batch = move_batch_to_device(full_batch_cpu, device)
        batch = _apply_manifest(full_batch_cpu, mask_manifest, device)
        evidence = _load_evidence(
            batch, retriever, evidence_mode, evidence_dim, args.top_k
        )
        if args.evidence_dropout or args.evidence_shuffle_ratio:
            evidence = perturb_evidence(
                evidence,
                dropout_prob=args.evidence_dropout,
                shuffle_prob=args.evidence_shuffle_ratio,
                generator=generator,
            )
        batch.evidence = evidence
        evidence_coverage.add(evidence, batch.availability, batch.sequence_mask)
        optimizer.zero_grad(set_to_none=True)
        if first_stage:
            outputs = model.stage1_forward(batch.features, batch.availability, batch.labels)
            total, batch_registry = _stage1_loss(
                outputs, batch, _task_name(args.dataset), args
            )
        else:
            output = model(
                batch.features,
                batch.availability,
                evidence=evidence,
                return_intermediates=True,
            )
            route_target = None
            teacher_logits = None
            utility_gap = None
            if args.per_mode == "teacher_router":
                if teacher is None:
                    raise RuntimeError("teacher_router mode has no teacher")
                route_target, teacher_logits, utility_gap = _teacher_terms(
                    teacher, output, batch, full_batch, args
                )
            if model.use_router:
                diagnostics.add(
                    output,
                    batch,
                    route_target=route_target,
                    utility_gap=utility_gap,
                )
            total, batch_registry, loss_components = _stage2_loss(
                output,
                batch,
                _task_name(args.dataset),
                args,
                teacher_logits=teacher_logits,
                route_target=route_target,
            )
        if not torch.isfinite(total):
            raise FloatingPointError(
                f"non-finite total loss at epoch={epoch}, batch={seen_batches}"
            )
        gradient_values = {}
        interval = max(1, int(getattr(args, "route_gradient_interval", 20)))
        if (
            getattr(args, "route_gradient_diagnostics", False)
            and not first_stage
            and model.router is not None
            and seen_batches % interval == 0
        ):
            gradient_values.update(
                component_gradient_norms(
                    args.route_weight * loss_components["route"],
                    loss_components["task"],
                    list(model.router.parameters()),
                )
            )
            valid_route = (~batch.availability) & batch.sequence_mask.unsqueeze(-1)
            if batch.evidence is not None:
                valid_route &= batch.evidence.mask.any(dim=-1).unsqueeze(-1).expand(-1, -1, 3)
            if route_target is not None and int(valid_route.sum()) > 1:
                gradient_values["student_target_pearson"] = pearson(
                    output.route[valid_route], route_target[valid_route]
                )
                gradient_values["student_target_spearman"] = spearman(
                    output.route[valid_route], route_target[valid_route]
                )
            gradient_values["valid_route_positions"] = float(valid_route.sum())
        total.backward()
        if gradient_values:
            gradient_values["router_grad_norm"] = parameter_grad_norm(
                model.router.parameters()
            )
            gradient_values["backbone_grad_norm"] = parameter_grad_norm(
                model.backbone.parameters()
            )
            for name, value in gradient_values.items():
                grad_registry[name] = grad_registry.get(name, 0.0) + float(value)
            grad_diag_batches += 1
        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        for name, value in batch_registry.items():
            registry[name] = registry.get(name, 0.0) + value
        seen_batches += 1
    count = max(1, seen_batches)
    metrics = {name: value / count for name, value in registry.items()}
    metrics.update(diagnostics.summary())
    metrics.update(evidence_coverage.summary())
    if grad_diag_batches:
        metrics.update({
            name: value / grad_diag_batches
            for name, value in grad_registry.items()
        })
    metrics["route_bce"] = metrics.get("route", 0.0)
    metrics["task_loss_total"] = metrics.get("task_joint", 0.0)
    metrics["route_weight"] = float(args.route_weight)
    metrics["router_warmup_active"] = float(router_warmup_active)
    return metrics


def _metrics(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    sequence_mask: torch.BoolTensor,
    dataset: str,
) -> Dict[str, float]:
    labels_np = labels.detach().cpu().numpy()
    preds_np = predictions.detach().cpu().numpy()
    mask_np = sequence_mask.detach().cpu().numpy().astype(bool)
    if dataset in {"CMUMOSI", "CMUMOSEI"}:
        valid = mask_np & (labels_np != 0)
        y = labels_np[valid]
        p = preds_np[valid]
        if y.size == 0:
            return {"acc": 0.0, "f1": 0.0, "mae": 0.0, "corr": 0.0, "count": 0}
        binary_y = y > 0
        binary_p = p > 0
        corr = float(np.corrcoef(y, p)[0, 1]) if y.size > 1 else 0.0
        return {
            "acc": float(accuracy_score(binary_y, binary_p)),
            "f1": float(f1_score(binary_y, binary_p, average="weighted")),
            "mae": float(np.mean(np.abs(y - p))),
            "corr": corr,
            "count": int(y.size),
        }
    valid = mask_np
    y = labels_np[valid].astype(np.int64)
    p = preds_np[valid].astype(np.int64)
    weights = mask_np[valid].astype(np.float32)
    if y.size == 0:
        return {"acc": 0.0, "f1": 0.0, "ua": 0.0, "mae": 0.0, "count": 0}
    return {
        "acc": float(accuracy_score(y, p, sample_weight=weights)),
        "f1": float(f1_score(y, p, average="weighted", sample_weight=weights)),
        "ua": float(
            recall_score(y, p, average="macro", sample_weight=weights, zero_division=0)
        ),
        "mae": 0.0,
        "count": int(y.size),
    }


def evaluate(
    model: PERStudent,
    loader: DataLoader,
    mask_manifest,
    evidence_mode: str,
    evidence_dim: int,
    retriever: Optional[FeatureNeighborRetriever],
    args,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_labels = []
    all_predictions = []
    all_masks = []
    total_loss = 0.0
    batch_count = 0
    diagnostics = RouteDiagnosticsAccumulator()
    evidence_coverage = EvidenceCoverageAccumulator()
    evaluation_audit = EvaluationAuditAccumulator()
    fusion_diagnostics = FusionDiagnosticsAccumulator()
    with torch.no_grad():
        for full_batch_cpu in loader:
            batch = _apply_manifest(full_batch_cpu, mask_manifest, device)
            evidence = _load_evidence(
                batch, retriever, evidence_mode, evidence_dim, args.top_k
            )
            batch.evidence = evidence
            output = model(
                batch.features,
                batch.availability,
                evidence=evidence,
                return_intermediates=True,
            )
            if model.use_router:
                diagnostics.add(output, batch)
            fusion_diagnostics.add(output, batch)
            loss = masked_task_loss(
                output.logits,
                batch.labels,
                batch.sequence_mask,
                _task_name(args.dataset),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite validation/test task loss")
            total_loss += float(loss.item())
            batch_count += 1
            all_labels.append(batch.labels)
            all_masks.append(batch.sequence_mask)
            if _task_name(args.dataset) == "regression":
                batch_predictions = output.logits.squeeze(-1)
            else:
                batch_predictions = output.logits.argmax(dim=-1)
            all_predictions.append(batch_predictions)
            evaluation_audit.add(
                batch.labels,
                batch_predictions,
                batch.availability,
                batch.sequence_mask,
            )
    labels = torch.cat(all_labels, dim=0)
    predictions = torch.cat(all_predictions, dim=0)
    sequence_mask = torch.cat(all_masks, dim=0)
    result = _metrics(labels, predictions, sequence_mask, args.dataset)
    result["loss"] = total_loss / max(1, batch_count)
    result.update(diagnostics.summary())
    result.update(evidence_coverage.summary())
    result.update(evaluation_audit.summary(args.dataset))
    result.update(fusion_diagnostics.summary())
    return result


def _selection_score(metrics: Dict[str, float], dataset: str) -> float:
    if dataset in {"CMUMOSI", "CMUMOSEI"}:
        return -metrics["mae"]
    return metrics["acc"]


def _set_requires_grad(module: Optional[torch.nn.Module], enabled: bool) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _configure_route_training(model, args, epoch: int, first_stage: bool) -> bool:
    """Configure warmup without changing default training behavior."""
    _set_requires_grad(
        model.backbone, epoch >= int(getattr(args, "freeze_backbone_epochs", 0))
    )
    warmup_epochs = int(getattr(args, "router_warmup_epochs", 0))
    warmup = (
        not first_stage
        and warmup_epochs > 0
        and epoch < args.stage_epoch + warmup_epochs
    )
    if warmup and bool(getattr(args, "freeze_candidate_during_warmup", False)):
        _set_requires_grad(model.backbone, False)
        _set_requires_grad(model.evidence_encoder, False)
        _set_requires_grad(model.observed_projections, False)
    else:
        _set_requires_grad(model.evidence_encoder, True)
        _set_requires_grad(model.observed_projections, True)
    _set_requires_grad(model.router, True)
    return warmup


def _save_checkpoint(
    path: Path,
    model: PERStudent,
    optimizer: torch.optim.Optimizer,
    args,
    epoch: int,
    stage: str,
    metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_args = {
        key: str(value) if isinstance(value, torch.device) else value
        for key, value in vars(args).items()
    }
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "stage": stage,
        "metrics": metrics,
        "args": serializable_args,
        "protocol": FAIR_PROTOCOL,
        "format_version": 1,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_fair(args, device: torch.device) -> Dict[str, object]:
    _check_feature_paths(args)
    seed_everything(args.seed)
    dataset, splits, expected_folds = _dataset_and_splits(args)
    adim, tdim, vdim = dataset.get_featDim()
    run_dir = Path(args.output_dir) / args.dataset / (
        f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        f"_seed{args.seed}_mr{args.mask_rate}_{args.per_mode}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for split in splits:
        fold = int(split["fold"])
        train_adapter = PERDatasetAdapter(dataset, split["train"])
        val_adapter = PERDatasetAdapter(dataset, split["val"])
        test_adapter = PERDatasetAdapter(dataset, split["test"])
        train_sampler = EpochRandomSampler(len(train_adapter), args.seed + fold)
        train_loader = _make_loader(
            train_adapter, args, training=True, sampler=train_sampler
        )
        sequential_train_loader = _make_loader(train_adapter, args, training=False)
        val_loader = _make_loader(val_adapter, args, training=False)
        test_loader = _make_loader(test_adapter, args, training=False)

        retriever = None
        evidence_dim = 1
        if args.evidence_mode == "feature":
            retriever = _build_retriever(
                sequential_train_loader, adim, tdim, vdim, args.top_k
            )
            evidence_dim = retriever.evidence_dim
        fold_dir = run_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        val_sampler = MissingPatternSampler(
            seed=args.seed + 7919 + fold,
            missing_rate=args.mask_rate,
            protocol=args.missing_protocol,
        )
        test_sampler = MissingPatternSampler(
            seed=args.seed,
            missing_rate=args.mask_rate,
            protocol=args.missing_protocol,
        )
        train_mask_sampler = MissingPatternSampler(
            seed=args.seed + fold,
            missing_rate=args.mask_rate,
            protocol=args.missing_protocol,
        )
        val_manifest_path = str(resolve_manifest_path(
            getattr(args, "val_mask_manifest", None),
            fold_dir / "val_mask_manifest.csv",
        ))
        test_manifest_path = str(resolve_manifest_path(
            getattr(args, "test_mask_manifest", None),
            fold_dir / "test_mask_manifest.csv",
        ))
        if not Path(val_manifest_path).exists():
            if getattr(args, "val_mask_manifest", None):
                raise FileNotFoundError(val_manifest_path)
            _collect_split_masks(
                val_loader,
                val_sampler,
                epoch=0,
                output_path=val_manifest_path,
                fixed_pattern=args.fixed_pattern,
            )
        if not Path(test_manifest_path).exists():
            if getattr(args, "test_mask_manifest", None):
                raise FileNotFoundError(test_manifest_path)
            _collect_split_masks(
                test_loader,
                test_sampler,
                epoch=0,
                output_path=test_manifest_path,
                fixed_pattern=args.fixed_pattern,
            )
        val_manifest = MaskManifest.load(val_manifest_path)
        test_manifest = MaskManifest.load(test_manifest_path)
        val_mask_sha256 = sha256_file(val_manifest_path)
        test_mask_sha256 = sha256_file(test_manifest_path)

        model = _build_student(args, device, adim, tdim, vdim, evidence_dim)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        teacher = (
            _load_teacher(args, device, adim, tdim, vdim, fold)
            if args.per_mode == "teacher_router"
            else None
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.l2
        )
        best_score = -float("inf")
        best_epoch = -1
        checkpoint_path = fold_dir / "best_val.pt"
        history = []
        for epoch in range(args.epochs):
            first_stage = epoch < args.stage_epoch
            router_warmup_active = _configure_route_training(
                model, args, epoch, first_stage
            )
            train_sampler.set_epoch(epoch)
            train_manifest_path = (
                str(fold_dir / "masks" / f"train_epoch_{epoch:04d}.csv")
                if args.save_train_masks
                else None
            )
            train_manifest = _collect_split_masks(
                train_loader,
                train_mask_sampler,
                epoch=epoch,
                output_path=train_manifest_path,
                fixed_pattern=args.fixed_pattern,
            )
            train_metrics = train_epoch(
                model,
                teacher,
                train_loader,
                train_manifest,
                args.evidence_mode,
                evidence_dim,
                retriever,
                optimizer,
                args,
                device,
                first_stage,
                epoch,
                router_warmup_active=router_warmup_active,
            )
            val_metrics = evaluate(
                model,
                val_loader,
                val_manifest,
                args.evidence_mode,
                evidence_dim,
                retriever,
                args,
                device,
            )
            score = _selection_score(val_metrics, args.dataset)
            if not first_stage and score > best_score:
                best_score = score
                best_epoch = epoch
                _save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    args,
                    epoch,
                    "stage2",
                    val_metrics,
                )
            history.append(
                {
                    "epoch": epoch,
                    "stage": "stage1" if first_stage else "stage2",
                    "train": train_metrics,
                    "val": val_metrics,
                }
            )
            print(
                f"fold={fold} epoch={epoch} stage={history[-1]['stage']} "
                f"val={json.dumps(val_metrics, sort_keys=True)}",
                flush=True,
            )
        if not checkpoint_path.exists():
            raise RuntimeError("fair training completed without a stage-2 checkpoint")
        payload = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(payload["model"], strict=True)
        test_metrics = evaluate(
            model,
            test_loader,
            test_manifest,
            args.evidence_mode,
            evidence_dim,
            retriever,
            args,
            device,
        )
        fold_result = {
            "protocol": FAIR_PROTOCOL,
            "fold": fold,
            "best_epoch": best_epoch,
            "validation": payload["metrics"],
            "test": test_metrics,
            "history": history,
            "checkpoint": str(checkpoint_path),
            "val_mask_manifest": val_manifest_path,
            "test_mask_manifest": test_manifest_path,
            "val_mask_sha256": val_mask_sha256,
            "test_mask_sha256": test_mask_sha256,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
        }
        _write_json(fold_dir / "result.json", fold_result)
        all_results.append(fold_result)
    summary = {
        "protocol": FAIR_PROTOCOL,
        "dataset": args.dataset,
        "seed": args.seed,
        "mask_rate": args.mask_rate,
        "missing_protocol": args.missing_protocol,
        "per_mode": args.per_mode,
        "coordinator_mode": args.coordinator_mode,
        "fixed_source_weight": args.fixed_source_weight,
        "evidence_seed": args.seed,
        "route_objective": args.route_objective,
        "router_warmup_epochs": args.router_warmup_epochs,
        "freeze_candidate_during_warmup": args.freeze_candidate_during_warmup,
        "evidence_mode": args.evidence_mode,
        "evidence_description": {
            "feature": "training-only feature-neighbor evidence; not semantic KB evidence",
            "empty": "empty evidence / prompt-only fallback",
            "random": "deterministic random training-bank evidence",
            "shuffled": "deterministically shuffled evidence",
            "source_only": "zero-content evidence with source availability retained",
        }[args.evidence_mode],
        "dataset_version": args.dataset_version,
        "teacher_checkpoint_template": args.teacher_checkpoint,
        "val_mask_manifest_override": getattr(args, "val_mask_manifest", None),
        "test_mask_manifest_override": getattr(args, "test_mask_manifest", None),
        "folds": all_results,
    }
    _write_json(run_dir / "summary.json", summary)
    return summary


def delegate_legacy(args) -> int:
    if args.per_mode != "off":
        raise ValueError("legacy-compat supports only --per-mode off")
    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "ComP" / "train_mr.py"),
        f"--dataset={args.dataset}",
        f"--audio-feature={args.audio_feature}",
        f"--text-feature={args.text_feature}",
        f"--video-feature={args.video_feature}",
        f"--seed={args.seed}",
        f"--batch-size={args.batch_size}",
        f"--epochs={args.epochs}",
        f"--stage_epoch={args.stage_epoch}",
        f"--lr={args.lr}",
        f"--hidden={args.hidden}",
        f"--depth={args.depth}",
        f"--num_heads={args.num_heads}",
        f"--drop_rate={args.drop_rate}",
        f"--attn_drop_rate={args.attn_drop_rate}",
        f"--gpu={args.gpu}",
        f"--mask_rate={args.mask_rate}",
        f"--lbd={args.lbd}",
    ]
    if args.no_cuda:
        command.append("--no-cuda")
    print(
        "PROTOCOL=legacy-compat (official ComP entry; not comparable to fair results)",
        flush=True,
    )
    return subprocess.call(command, cwd=str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-feature", type=str, required=True)
    parser.add_argument("--text-feature", type=str, required=True)
    parser.add_argument("--video-feature", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="CMUMOSI")
    parser.add_argument("--protocol", choices=(LEGACY_PROTOCOL, FAIR_PROTOCOL), default=FAIR_PROTOCOL)
    parser.add_argument("--per-mode", choices=PER_MODES, default="task_router")
    parser.add_argument("--missing-protocol", choices=("categorical", "bernoulli"), default="categorical")
    parser.add_argument("--fixed-pattern", type=int, nargs=3, default=None)
    parser.add_argument("--evidence-mode", choices=("empty", "feature", "random", "shuffled", "source_only"), default="feature")
    parser.add_argument("--coordinator-mode", choices=("on", "fixed", "off"), default="on")
    parser.add_argument("--coordinator-fixed-weights", type=float, nargs=3, default=None)
    parser.add_argument("--fix-private-branch", action="store_true", default=True)
    parser.add_argument("--legacy-private-branch", dest="fix_private_branch", action="store_false")

    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--drop_rate", type=float, default=0.0)
    parser.add_argument("--attn_drop_rate", type=float, default=0.0)
    parser.add_argument("--hidden", type=int, default=100)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--n_speakers", type=int, default=2)
    parser.add_argument("--lbd", type=float, default=0.5)
    parser.add_argument("--seq_len", type=int, default=65)

    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stage_epoch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--mask_rate", type=float, default=0.7)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--save-train-masks", action="store_true", default=True)
    parser.add_argument("--no-save-train-masks", dest="save_train_masks", action="store_false")

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--val-mask-manifest", type=str, default=None)
    parser.add_argument("--test-mask-manifest", type=str, default=None)
    parser.add_argument("--router-hidden", type=int, default=None)
    parser.add_argument("--router-dropout", type=float, default=0.0)
    parser.add_argument("--route-weight", type=float, default=0.5)
    parser.add_argument("--route-temperature", type=float, default=1.0)
    parser.add_argument("--route-objective", choices=("bce", "smooth_l1"), default="bce")
    parser.add_argument("--route-gradient-diagnostics", action="store_true", default=False)
    parser.add_argument("--route-gradient-interval", type=int, default=20)
    parser.add_argument("--router-warmup-epochs", type=int, default=0)
    parser.add_argument("--freeze-candidate-during-warmup", action="store_true", default=False)
    parser.add_argument("--fixed-source-weight", type=float, default=0.5)
    parser.add_argument("--kd-weight", type=float, default=0.0)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--evidence-dropout", type=float, default=0.0)
    parser.add_argument("--evidence-shuffle-ratio", type=float, default=0.0)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--disentangle-weight", type=float, default=0.0)
    parser.add_argument("--private-noise-weight", type=float, default=1.0)
    parser.add_argument("--private-antilabel-weight", type=float, default=1.0)
    parser.add_argument("--gradient-clip", type=float, default=0.0)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--teacher-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "runs" / "per"))
    parser.add_argument("--dataset-version", type=str, default="ComP-28192d3")
    return parser


def _configure_dataset_defaults(args) -> None:
    if args.dataset == "CMUMOSI":
        args.num_folder, args.n_classes, args.n_speakers, args.seq_len = 1, 1, 1, 65
    elif args.dataset == "CMUMOSEI":
        args.num_folder, args.n_classes, args.n_speakers, args.seq_len = 1, 1, 1, 150
    elif args.dataset == "IEMOCAPFour":
        args.num_folder, args.n_classes, args.n_speakers, args.seq_len = 5, 4, 2, 150
    elif args.dataset == "IEMOCAPSix":
        args.num_folder, args.n_classes, args.n_speakers, args.seq_len = 5, 6, 2, 150
    else:
        raise ValueError(args.dataset)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_dataset_defaults(args)
    args.device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    if args.protocol == LEGACY_PROTOCOL:
        return delegate_legacy(args)
    if args.fixed_pattern is not None:
        if any(value not in (0, 1) for value in args.fixed_pattern) or sum(
            args.fixed_pattern
        ) == 0:
            parser.error("--fixed-pattern must be a non-empty binary triple")
    if args.per_mode in {"off", "generated"} and args.evidence_mode != "empty":
        print(
            f"INFO: --per-mode {args.per_mode} forces evidence_mode=empty",
            file=sys.stderr,
        )
        args.evidence_mode = "empty"
    if args.per_mode == "teacher_router" and not args.teacher_checkpoint:
        parser.error("--per-mode teacher_router requires --teacher-checkpoint")
    if not 0.0 <= args.fixed_source_weight <= 1.0:
        parser.error("--fixed-source-weight must be in [0,1]")
    if args.router_warmup_epochs < 0:
        parser.error("--router-warmup-epochs must be non-negative")
    if args.route_gradient_interval <= 0:
        parser.error("--route-gradient-interval must be positive")
    if args.per_mode == "generated" and args.evidence_mode != "empty":
        parser.error("--per-mode generated does not use external evidence")
    if args.per_mode in {"fixed", "task_router", "teacher_router"} and args.evidence_mode == "empty":
        print(
            "WARNING: task/teacher router has no evidence; empty evidence forces "
            "fallback to prompt candidates.",
            file=sys.stderr,
        )
    result = run_fair(args, args.device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
