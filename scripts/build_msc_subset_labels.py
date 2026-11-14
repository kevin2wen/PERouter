from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.eval_knowledge import element_loss
from ComP.knowledge.msc.cache import build_state_cache, load_state_cache, query_id
from ComP.knowledge.msc.forward import apply_subset
from ComP.knowledge.msc.subsets import valid_subset_masks
from ComP.knowledge.pipeline import knowledge_forward
from ComP.knowledge.retriever import KnowledgeContextRetriever
from ComP.per.batch import PERDatasetAdapter
from ComP.per.missing import MaskManifest
from ComP.per.types import MODALITIES
from ComP.train_knowledge import _build_knowledge_student, build_pipeline
from ComP.train_per import (
    _apply_manifest,
    _check_feature_paths,
    _dataset_and_splits,
    _make_loader,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_single_card(args) -> int:
    cache = load_state_cache(args.state_cache)
    utility = cache["utility"]
    valid = cache["candidate_mask"].bool()
    positive = (utility > 0) & valid
    skip_label = (~positive.any(dim=-1)).to(torch.float32)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "msc_format_version": 1,
        "split": cache["split"],
        "query_ids": cache["query_ids"],
        "records": cache.get("records", []),
        "observed_mask": cache["observed_mask"],
        "target_modality": cache["target_modality"],
        "missing_pattern": cache["missing_pattern"],
        "missing_rate": cache["missing_rate"],
        "candidate_ids": cache["candidate_ids"],
        "candidate_mask": cache["candidate_mask"],
        "candidate_vectors": cache["candidate_vectors"],
        "prompt_vectors": cache["prompt_vectors"],
        "source_confidence": cache["source_confidence"],
        "conflict_score": cache.get("conflict_score"),
        "single_utility": utility,
        "skip_label": skip_label,
        "skip_utility": cache["skip_utility"],
    }
    torch.save(payload, output / "msc_single_cache.pt")
    (output / "msc_single_manifest.json").write_text(
        json.dumps(
            {
                "query_count": len(cache["query_ids"]),
                "skip_rate": float(skip_label.mean()),
                "positive_candidate_rate": float(positive.sum() / max(1, int(valid.sum()))),
                "source_state_cache_sha256": args.state_cache_sha256 or None,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


def build_exact_subsets(args) -> int:
    payload = torch.load(args.student_checkpoint, map_location="cpu", weights_only=False)
    saved = dict(payload["args"])
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    saved.update({"no_cuda": device.type == "cpu", "gpu": args.gpu, "device": device})
    cli = Namespace(**saved)
    _check_feature_paths(cli)
    dataset, splits, _ = _dataset_and_splits(cli)
    split = splits[0]
    key = "train" if args.split == "train" else "val"
    loader = _make_loader(
        PERDatasetAdapter(dataset, split[key]), cli, training=False
    )
    adapter = torch.load(
        args.card_adapter_checkpoint, map_location="cpu", weights_only=False
    )
    adim, tdim, vdim = dataset.get_featDim()
    evidence_dim = adim + tdim + vdim
    model = _build_knowledge_student(
        cli, device, adim, tdim, vdim, evidence_dim, payload
    )
    model.evidence_encoder.load_state_dict(adapter["evidence_encoder"], strict=True)
    model.eval()
    model.requires_grad_(False)
    bank = (
        KnowledgeContextRetriever.load(args.knowledge_index)
        if args.knowledge_index and Path(args.knowledge_index).exists()
        else None
    )
    pipeline, projector, _ = build_pipeline(
        loader,
        adim=adim,
        tdim=tdim,
        vdim=vdim,
        evidence_dim=evidence_dim,
        top_k=cli.top_k,
        seed=cli.seed,
        bank=bank,
    )
    projector.load_state_dict(adapter["projector"], strict=True)
    projector.to(device)
    projector.eval()
    manifest = MaskManifest.load(args.mask_manifest)
    task = "regression" if cli.dataset in {"CMUMOSI", "CMUMOSEI"} else "classification"
    query_rows = []
    count = 0
    for full_cpu in loader:
        if count >= args.query_limit:
            break
        batch = _apply_manifest(full_cpu, manifest, device)
        result = pipeline.build(batch)
        skip_output = model(
            batch.features,
            batch.availability,
            evidence=None,
            return_intermediates=True,
            route_override=torch.zeros_like(batch.availability).to(
                batch.features[MODALITIES[0]].dtype
            ),
        )
        skip_loss = element_loss(skip_output, batch.labels, task)
        for batch_index in range(batch.sequence_mask.size(0)):
            if count >= args.query_limit:
                break
            for time_index in range(batch.sequence_mask.size(1)):
                if count >= args.query_limit:
                    break
                if not bool(batch.sequence_mask[batch_index, time_index]):
                    continue
                for target_index, target in enumerate(MODALITIES):
                    if count >= args.query_limit:
                        break
                    if bool(batch.availability[batch_index, time_index, target_index]):
                        continue
                    evidence = result.evidence_by_target[target]
                    candidate_valid = evidence.mask[batch_index, time_index]
                    if not bool(candidate_valid.any()):
                        continue
                    subsets = valid_subset_masks(candidate_valid, evidence.mask.size(-1))
                    qid = query_id(
                        str(batch.sample_ids[batch_index]),
                        str(batch.utterance_ids[batch_index][time_index]),
                        str(target_index),
                    )
                    base = float(skip_loss[batch_index, time_index])
                    for slot in range(subsets.size(0)):
                        subset = apply_subset(
                            result, target, batch_index, time_index, subsets[slot]
                        )
                        output = knowledge_forward(model, batch, subset, gamma=args.gamma)
                        loss_value = float(
                            element_loss(output, batch.labels, task)[
                                batch_index, time_index
                            ]
                        )
                        query_rows.append(
                            {
                                "query_id": qid,
                                "sample_id": str(batch.sample_ids[batch_index]),
                                "utterance_id": str(
                                    batch.utterance_ids[batch_index][time_index]
                                ),
                                "target_modality": target,
                                "subset_mask": int(
                                    sum(
                                        int(subsets[slot, rank])
                                        << rank
                                        for rank in range(subsets.size(1))
                                    )
                                ),
                                "subset_size": int(subsets[slot].sum()),
                                "task_loss": loss_value,
                                "utility": base - loss_value,
                            }
                        )
                    count += 1
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    subset_count = len(query_rows)
    query_count = len({row["query_id"] for row in query_rows})
    torch.save(
        {
            "msc_format_version": 1,
            "split": args.split,
            "query_rows": query_rows,
            "student_checkpoint_sha256": sha256_file(args.student_checkpoint),
            "card_adapter_sha256": sha256_file(args.card_adapter_checkpoint),
            "mask_manifest_sha256": sha256_file(args.mask_manifest),
            "gamma": args.gamma,
        },
        output / "msc_subset_cache.pt",
    )
    (output / "msc_subset_manifest.json").write_text(
        json.dumps(
            {
                "query_count": query_count,
                "subset_count": subset_count,
                "split": args.split,
                "gamma": args.gamma,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {"query_count": query_count, "subset_count": subset_count},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    single = sub.add_parser("single_card")
    single.add_argument("--state-cache", required=True)
    single.add_argument("--output-dir", required=True)
    single.add_argument("--state-cache-sha256")

    exact = sub.add_parser("exact_subsets")
    exact.add_argument("--student-checkpoint", required=True)
    exact.add_argument("--card-adapter-checkpoint", required=True)
    exact.add_argument("--knowledge-index", required=True)
    exact.add_argument("--mask-manifest", required=True)
    exact.add_argument("--split", choices=("train", "validation"), required=True)
    exact.add_argument("--query-limit", type=int, default=200)
    exact.add_argument("--gamma", type=float, default=0.5)
    exact.add_argument("--gpu", type=int, default=0)
    exact.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    if args.mode == "single_card":
        return build_single_card(args)
    return build_exact_subsets(args)


if __name__ == "__main__":
    raise SystemExit(main())
