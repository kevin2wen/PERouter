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

from ComP.per.batch import PERDatasetAdapter
from ComP.per.evidence_audit import EvidenceCoverageAccumulator, evidence_trace_rows
from ComP.per.missing import MaskManifest
from ComP.train_per import (
    _apply_manifest,
    _build_retriever,
    _build_student,
    _check_feature_paths,
    _dataset_and_splits,
    _load_evidence,
    _make_loader,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--mask-manifest", required=True)
    parser.add_argument("--evidence-mode", choices=("empty", "feature", "random", "shuffled", "source_only"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    cli = parser.parse_args()
    mask_path = Path(cli.mask_manifest)
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    payload = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    saved = dict(payload["args"])
    device = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    saved.update({"no_cuda": device.type == "cpu", "gpu": cli.gpu, "device": device})
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
    evidence_dim = 1
    if cli.evidence_mode != "empty":
        retriever = _build_retriever(train_loader, adim, tdim, vdim, args.top_k)
        evidence_dim = retriever.evidence_dim
    model = _build_student(args, device, adim, tdim, vdim, evidence_dim)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    manifest = MaskManifest.load(str(mask_path))
    coverage = EvidenceCoverageAccumulator()
    trace_rows = []
    with torch.no_grad():
        for full_cpu in target_loader:
            batch = _apply_manifest(full_cpu, manifest, device)
            evidence = _load_evidence(
                batch,
                retriever,
                cli.evidence_mode,
                evidence_dim,
                args.top_k,
                seed=args.seed,
            )
            coverage.add(evidence, batch.availability, batch.sequence_mask)
            trace_rows.extend(
                evidence_trace_rows(
                    evidence,
                    batch.availability,
                    batch.sequence_mask,
                    split=cli.split,
                    sample_ids=batch.sample_ids,
                    utterance_ids=batch.utterance_ids,
                    evidence_mode=cli.evidence_mode,
                    mask_rate=args.mask_rate,
                    seed=args.seed,
                    sampler_version=args.missing_protocol,
                )
            )
    output = Path(cli.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evidence_coverage.json").write_text(
        json.dumps(
            {
                "split": cli.split,
                "evidence_mode": cli.evidence_mode,
                "checkpoint": cli.checkpoint,
                "mask_manifest": cli.mask_manifest,
                "coverage": coverage.summary(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    fields = list(trace_rows[0]) if trace_rows else [
        "split", "sample_id", "utterance_id", "availability",
        "target_modality", "evidence_rank", "neighbor_sample_id",
        "neighbor_utterance_id", "source_confidence", "evidence_available",
        "evidence_mode", "mask_rate", "seed", "sampler_version",
        "shared_across_target_modalities",
    ]
    with (output / f"evidence_trace_{cli.split}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trace_rows)
    print(json.dumps(coverage.summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
