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

from ComP.per.batch import PERDatasetAdapter
from ComP.per.missing import MaskManifest
from ComP.train_knowledge import build_pipeline
from ComP.train_per import (
    _apply_manifest,
    _check_feature_paths,
    _dataset_and_splits,
    _make_loader,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--student-checkpoint", required=True)
    p.add_argument("--mask-manifest", required=True)
    p.add_argument("--query-split", choices=("train", "validation"), default="validation")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    return p


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def main() -> int:
    cli = parser().parse_args()
    payload = torch.load(
        cli.student_checkpoint, map_location="cpu", weights_only=False
    )
    saved = dict(payload["args"])
    device = torch.device("cpu")
    saved.update(
        {"no_cuda": True, "device": device, "gpu": 0}
    )
    args = Namespace(**saved)
    if cli.top_k is not None:
        args.top_k = int(cli.top_k)
    _check_feature_paths(args)
    dataset, splits, _ = _dataset_and_splits(args)
    split = next(item for item in splits if int(item["fold"]) == 0)
    adim, tdim, vdim = dataset.get_featDim()
    train_loader = _make_loader(
        PERDatasetAdapter(dataset, split["train"]), args, training=False
    )
    split_key = "train" if cli.query_split == "train" else "val"
    query_loader = _make_loader(
        PERDatasetAdapter(dataset, split[split_key]), args, training=False
    )
    pipeline, projector, bank = build_pipeline(
        train_loader,
        adim=adim,
        tdim=tdim,
        vdim=vdim,
        evidence_dim=adim + tdim + vdim,
        top_k=args.top_k,
        seed=args.seed,
    )
    manifest = MaskManifest.load(cli.mask_manifest)
    output = Path(cli.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cards_path = output / "knowledge_cards.jsonl"
    contexts_path = output / "ragpt_retrieval_trace.jsonl"
    candidates_path = output / "candidate_rows.csv"
    card_rows_by_id = {}
    context_rows = []
    candidate_rows = []
    with torch.no_grad():
        for batch_index, full_cpu in enumerate(query_loader):
            batch = _apply_manifest(full_cpu, manifest, device)
            result = pipeline.build(batch)
            for card in result.cards:
                card_rows_by_id[card.knowledge_id] = card.__dict__.copy()
            for context in result.contexts:
                context_rows.append(
                    {
                        "query_id": context.query_id,
                        "sample_id": context.sample_id,
                        "dialogue_id": context.dialogue_id,
                        "utterance_id": context.utterance_id,
                        "target_modality": context.target_modality,
                        "rank": context.rank,
                        "text": context.text,
                        "retrieval_score": context.retrieval_score,
                        "source_confidence": context.source_confidence,
                        "source_modalities": list(context.source_modalities),
                        "source_id": context.source_id,
                        "source_split": "train",
                        "audio_feature_sha256": (
                            tensor_hash(context.audio_feature)
                            if context.audio_feature is not None
                            else None
                        ),
                        "video_feature_sha256": (
                            tensor_hash(context.video_feature)
                            if context.video_feature is not None
                            else None
                        ),
                        "same_video": None,
                    }
                )
            for target, candidates in result.candidates.items():
                for candidate in candidates:
                    candidate_rows.append(
                        {
                            "query_id": candidate.query_id,
                            "target_modality": target,
                            "rank": candidate.rank,
                            "knowledge_id": candidate.card.knowledge_id,
                            "source_id": candidate.card.source_id,
                            "applicability_score": candidate.applicability_score,
                            "retrieval_score": candidate.retrieval_score,
                            "source_confidence": candidate.source_confidence,
                            "conflict_score": candidate.conflict_score,
                            "encoded_feature_sha256": tensor_hash(
                                candidate.encoded_feature
                            ),
                        }
                    )
            if cli.max_batches and batch_index + 1 >= cli.max_batches:
                break
    card_rows = [
        card_rows_by_id[knowledge_id]
        for knowledge_id in sorted(card_rows_by_id)
    ]
    with cards_path.open("w", encoding="utf-8") as handle:
        for row in card_rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    with contexts_path.open("w", encoding="utf-8") as handle:
        for row in context_rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    with candidates_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "query_id",
            "target_modality",
            "rank",
            "knowledge_id",
            "source_id",
            "applicability_score",
            "retrieval_score",
            "source_confidence",
            "conflict_score",
            "encoded_feature_sha256",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidate_rows)
    manifest_data = {
        "query_split": cli.query_split,
        "source_split": "train",
        "student_checkpoint": cli.student_checkpoint,
        "student_checkpoint_sha256": sha256_file(cli.student_checkpoint),
        "mask_manifest": cli.mask_manifest,
        "mask_sha256": sha256_file(cli.mask_manifest),
        "source_fingerprint": bank.source_fingerprint(),
        "schema_version": "knowledge-card-v1",
        "card_count": len(card_rows),
        "context_count": len(context_rows),
        "candidate_count": len(candidate_rows),
        "card_file_sha256": sha256_file(cards_path),
        "context_file_sha256": sha256_file(contexts_path),
        "candidate_file_sha256": sha256_file(candidates_path),
    }
    (output / "source_manifest.json").write_text(
        json.dumps(manifest_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest_data, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
