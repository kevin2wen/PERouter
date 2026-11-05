from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Optional, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.per.batch import PERDatasetAdapter
from ComP.per.missing import MaskManifest, MissingPatternSampler
from ComP.train_per import (
    _build_retriever,
    _build_student,
    _check_feature_paths,
    _collect_split_masks,
    _dataset_and_splits,
    _make_loader,
    evaluate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-fold", type=int, default=0)
    parser.add_argument("--mask-manifest", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    cli = build_parser().parse_args(argv)
    checkpoint_path = Path(cli.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if payload.get("protocol") != "fair":
        raise ValueError("eval_per.py accepts fair checkpoints only")
    saved = payload["args"]
    saved.update(
        {
            "no_cuda": cli.no_cuda,
            "gpu": cli.gpu,
            "device": torch.device(
                f"cuda:{cli.gpu}"
                if torch.cuda.is_available() and not cli.no_cuda
                else "cpu"
            ),
        }
    )
    args = Namespace(**saved)
    _check_feature_paths(args)
    dataset, splits, _num_folds = _dataset_and_splits(args)
    split = next((item for item in splits if int(item["fold"]) == cli.split_fold), None)
    if split is None:
        raise ValueError(f"split fold {cli.split_fold} does not exist")
    adim, tdim, vdim = dataset.get_featDim()
    train_adapter = PERDatasetAdapter(dataset, split["train"])
    test_adapter = PERDatasetAdapter(dataset, split["test"])
    sequential_train_loader = _make_loader(train_adapter, args, training=False)
    test_loader = _make_loader(test_adapter, args, training=False)

    retriever = None
    evidence_dim = 1
    if args.evidence_mode == "feature":
        retriever = _build_retriever(
            sequential_train_loader, adim, tdim, vdim, args.top_k
        )
        evidence_dim = retriever.evidence_dim
    model = _build_student(
        args, args.device, adim, tdim, vdim, evidence_dim
    )
    model.load_state_dict(payload["model"], strict=True)

    if cli.mask_manifest is None:
        manifest_path = checkpoint_path.parent / "test_mask_manifest.csv"
    else:
        manifest_path = Path(cli.mask_manifest)
    if not manifest_path.exists():
        sampler = MissingPatternSampler(
            seed=args.seed,
            missing_rate=args.mask_rate,
            protocol=args.missing_protocol,
        )
        _collect_split_masks(
            test_loader,
            sampler,
            epoch=0,
            output_path=str(manifest_path),
            fixed_pattern=args.fixed_pattern,
        )
    manifest = MaskManifest.load(str(manifest_path))
    metrics = evaluate(
        model,
        test_loader,
        manifest,
        args.evidence_mode,
        evidence_dim,
        retriever,
        args,
        args.device,
    )
    result = {
        "protocol": "fair",
        "checkpoint": str(checkpoint_path),
        "mask_manifest": str(manifest_path),
        "split_fold": cli.split_fold,
        "metrics": metrics,
        "teacher_used_at_inference": False,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if cli.output:
        output = Path(cli.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
