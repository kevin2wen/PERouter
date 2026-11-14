from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from argparse import Namespace
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.knowledge.adapters.knowledge_bridger_adapter import KnowledgeBridgerAdapter
from ComP.knowledge.adapters.ragpt_adapter import RAGPTAdapter
from ComP.knowledge.card_builder_v2 import RuleBasedKnowledgeExtractorV2
from ComP.knowledge.cards import KnowledgeCandidateBuilder
from ComP.knowledge.encoder import RuleKnowledgeCardEncoder
from ComP.knowledge.pipeline import KnowledgePipeline
from ComP.knowledge.retriever import KnowledgeContextRetriever
from ComP.per.batch import PERDatasetAdapter
from ComP.per.missing import MaskManifest
from ComP.train_per import _apply_manifest, _check_feature_paths, _dataset_and_splits, _make_loader


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--knowledge-index", required=True)
    parser.add_argument("--mask-manifest", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    payload = torch.load(args.student_checkpoint, map_location="cpu", weights_only=False)
    cli = Namespace(**payload["args"])
    cli.no_cuda = True
    _check_feature_paths(cli)
    dataset, splits, _ = _dataset_and_splits(cli)
    split = splits[0]
    key = "train" if args.split == "train" else "val"
    loader = _make_loader(PERDatasetAdapter(dataset, split[key]), cli, training=False)
    bank = KnowledgeContextRetriever.load(args.knowledge_index)
    retrieval = RAGPTAdapter(bank.retrieve, top_k=args.top_k, bank_split="train")
    bridger = KnowledgeBridgerAdapter(
        RuleBasedKnowledgeExtractorV2(),
        source_type="train_retrieved_rule_v2",
        source_version="knowledge-rule-v2",
        source_license="research-only",
    )
    adim, tdim, vdim = dataset.get_featDim()
    encoder = RuleKnowledgeCardEncoder(adim, vdim)
    pipeline = KnowledgePipeline(retrieval, bridger, encoder, top_k=args.top_k)
    manifest = MaskManifest.load(args.mask_manifest)
    cards = {}
    for full_cpu in loader:
        batch = _apply_manifest(full_cpu, manifest, torch.device("cpu"))
        result = pipeline.build(batch)
        for card in result.cards:
            cards[card.knowledge_id] = card
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "knowledge_cards_v2.jsonl").open("w", encoding="utf-8") as handle:
        for card in cards.values():
            handle.write(json.dumps(dataclasses.asdict(card), sort_keys=True) + "\n")
    report = {
        "card_version": "v2",
        "cards": len(cards),
        "split": args.split,
        "top_k": args.top_k,
        "affected_distribution": dict(
            Counter(card.affect_prior for card in cards.values())
        ),
        "relation_distribution": dict(
            Counter(card.relation_type for card in cards.values())
        ),
        "trigger_distribution": dict(
            Counter(card.trigger_type for card in cards.values())
        ),
        "knowledge_index_sha256": sha256_file(args.knowledge_index),
        "mask_manifest_sha256": sha256_file(args.mask_manifest),
        "student_checkpoint_sha256": sha256_file(args.student_checkpoint),
        "cards_sha256": sha256_file(output / "knowledge_cards_v2.jsonl"),
        "test_accessed": False,
    }
    (output / "card_bank_manifest_v2.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
