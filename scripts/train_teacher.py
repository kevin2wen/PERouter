from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.per.batch import PERDatasetAdapter
from ComP.per.missing import MaskManifest, MissingPatternSampler
from ComP.train_per import (
    EpochRandomSampler,
    _build_student,
    _check_feature_paths,
    _collect_split_masks,
    _configure_dataset_defaults,
    _dataset_and_splits,
    _make_loader,
    _set_requires_grad,
    _selection_score,
    seed_everything,
    evaluate,
    train_epoch,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a frozen complete-view ComP teacher without touching test data."
    )
    parser.add_argument("--audio-feature", required=True)
    parser.add_argument("--text-feature", required=True)
    parser.add_argument("--video-feature", required=True)
    parser.add_argument("--dataset", default="CMUMOSI")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--drop_rate", type=float, default=0.0)
    parser.add_argument("--attn_drop_rate", type=float, default=0.0)
    parser.add_argument("--hidden", type=int, default=100)
    parser.add_argument("--lbd", type=float, default=0.5)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stage_epoch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--private-noise-weight", type=float, default=1.0)
    parser.add_argument("--private-antilabel-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--disentangle-weight", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=0.0)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--output", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_dataset_defaults(args)
    args.protocol = "fair"
    args.per_mode = "off"
    args.missing_protocol = "categorical"
    args.evidence_mode = "empty"
    args.coordinator_mode = "on"
    args.coordinator_fixed_weights = None
    args.fix_private_branch = True
    args.mask_rate = 0.0
    args.top_k = 1
    args.router_hidden = None
    args.router_dropout = 0.0
    args.route_weight = 0.0
    args.route_temperature = 1.0
    args.kd_weight = 0.0
    args.kd_temperature = 2.0
    args.evidence_dropout = 0.0
    args.evidence_shuffle_ratio = 0.0
    args.fixed_pattern = None
    args.save_train_masks = False
    args.output_dir = str(REPO_ROOT / "runs" / "per")
    args.dataset_version = "ComP-28192d3"
    args.teacher_checkpoint = None
    args.device = torch.device(
        f"cuda:{args.gpu}"
        if torch.cuda.is_available() and not args.no_cuda
        else "cpu"
    )
    _check_feature_paths(args)
    seed_everything(args.seed)
    dataset, splits, _ = _dataset_and_splits(args)
    split = next((item for item in splits if int(item["fold"]) == args.fold), None)
    if split is None:
        raise ValueError(f"split fold {args.fold} does not exist")
    adim, tdim, vdim = dataset.get_featDim()
    train_adapter = PERDatasetAdapter(dataset, split["train"])
    val_adapter = PERDatasetAdapter(dataset, split["val"])
    train_sampler = EpochRandomSampler(len(train_adapter), args.seed + args.fold)
    train_loader = _make_loader(
        train_adapter, args, training=True, sampler=train_sampler
    )
    val_loader = _make_loader(val_adapter, args, training=False)
    sequential_loader = _make_loader(train_adapter, args, training=False)

    output = (
        Path(args.output)
        if args.output
        else REPO_ROOT
        / "runs"
        / "per"
        / "teachers"
        / args.dataset
        / f"seed{args.seed}_fold{args.fold}"
        / "teacher.pt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.parent / "full_view_manifest.csv"
    sampler = MissingPatternSampler(
        seed=args.seed, missing_rate=0.0, protocol="categorical"
    )
    _collect_split_masks(
        sequential_loader,
        sampler,
        epoch=0,
        output_path=str(manifest_path),
        fixed_pattern=(1, 1, 1),
    )
    manifest = MaskManifest.load(str(manifest_path))
    val_manifest_path = output.parent / "full_view_val_manifest.csv"
    _collect_split_masks(
        val_loader,
        sampler,
        epoch=0,
        output_path=str(val_manifest_path),
        fixed_pattern=(1, 1, 1),
    )
    val_manifest = MaskManifest.load(str(val_manifest_path))

    student = _build_student(
        args, args.device, adim, tdim, vdim, evidence_dim=1
    )
    optimizer = torch.optim.Adam(
        student.parameters(), lr=args.lr, weight_decay=args.l2
    )
    history = []
    best_score = -float("inf")
    best_epoch = -1
    best_state = None
    for epoch in range(args.epochs):
        first_stage = epoch < args.stage_epoch
        _set_requires_grad(
            student.backbone, epoch >= args.freeze_backbone_epochs
        )
        train_sampler.set_epoch(epoch)
        metrics = train_epoch(
            student,
            None,
            train_loader,
            manifest,
            "empty",
            1,
            None,
            optimizer,
            args,
            args.device,
            first_stage,
            epoch,
        )
        val_metrics = evaluate(
            student,
            val_loader,
            val_manifest,
            "empty",
            1,
            None,
            args,
            args.device,
        )
        if not first_stage:
            score = _selection_score(val_metrics, args.dataset)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in student.backbone.state_dict().items()
                }
        history.append(
            {
                "epoch": epoch,
                "stage": "stage1" if first_stage else "stage2",
                "train_loss": metrics,
                "val": val_metrics,
            }
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    if best_state is None:
        raise RuntimeError("teacher training produced no stage-2 checkpoint")
    payload = {
        "model": best_state,
        "args": {
            key: str(value) if isinstance(value, torch.device) else value
            for key, value in vars(args).items()
        },
        "teacher_kind": "complete-view-ComP",
        "full_view": True,
        "split_fold": args.fold,
        "best_epoch": best_epoch,
        "validation": history[best_epoch]["val"],
        "history": history,
        "protocol": "fair-teacher-only",
    }
    temporary = output.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    print(f"saved complete-view teacher to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
