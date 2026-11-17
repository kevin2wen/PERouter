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

from ComP.knowledge.adapters.knowledge_bridger_adapter import KnowledgeBridgerAdapter
from ComP.knowledge.adapters.ragpt_adapter import RAGPTAdapter
from ComP.knowledge.cards import KnowledgeCandidateBuilder
from ComP.knowledge.encoder import RuleKnowledgeCardEncoder, TargetSpecificCardProjector
from ComP.knowledge.kb_v3 import VaderKnowledgeExtractor
from ComP.knowledge.pipeline import KnowledgeBatchResult, KnowledgePipeline
from ComP.knowledge.retriever import KnowledgeContextRetriever
from ComP.per.batch import PERDatasetAdapter
from ComP.per.missing import MaskManifest
from ComP.per.types import MODALITIES, EvidenceBatch
from ComP.train_knowledge import _build_knowledge_student
from ComP.train_per import (
    _apply_manifest, _check_feature_paths, _dataset_and_splits, _make_loader,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def isolate(evidence: EvidenceBatch, b: int, t: int, rank: int) -> EvidenceBatch:
    mask = torch.zeros_like(evidence.mask)
    mask[b, t, rank] = evidence.mask[b, t, rank]
    confidence = evidence.source_confidence * mask.to(evidence.source_confidence.dtype)
    conflict = (
        evidence.conflict_score * mask.to(evidence.conflict_score.dtype)
        if evidence.conflict_score is not None else None
    )
    return EvidenceBatch(
        ids=evidence.ids, features=evidence.features.clone(), mask=mask,
        source_confidence=confidence, conflict_score=conflict,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--card-adapter-checkpoint", required=True)
    parser.add_argument("--knowledge-index", required=True)
    parser.add_argument("--mask-manifest", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    student = torch.load(args.student_checkpoint, map_location="cpu", weights_only=False)
    cli = Namespace(**student["args"])
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    cli.no_cuda = device.type == "cpu"
    cli.gpu = args.gpu
    cli.device = device
    _check_feature_paths(cli)
    dataset, splits, _ = _dataset_and_splits(cli)
    split = splits[0]
    key = "train" if args.split == "train" else "val"
    loader = _make_loader(PERDatasetAdapter(dataset, split[key]), cli, training=False)
    adim, tdim, vdim = dataset.get_featDim()
    evidence_dim = adim + tdim + vdim
    model = _build_knowledge_student(cli, device, adim, tdim, vdim, evidence_dim, student)
    adapter = torch.load(args.card_adapter_checkpoint, map_location="cpu", weights_only=False)
    model.evidence_encoder.load_state_dict(adapter["evidence_encoder"], strict=True)
    model.eval()
    bank = KnowledgeContextRetriever.load(args.knowledge_index)
    card_encoder = RuleKnowledgeCardEncoder(adim, vdim)
    projector = TargetSpecificCardProjector(
        card_encoder.output_dim, evidence_dim, zero_init=True
    )
    projector.load_state_dict(adapter["projector"], strict=True)
    projector.to(device).eval()
    pipeline = KnowledgePipeline(
        RAGPTAdapter(bank.retrieve, top_k=cli.top_k, bank_split="train"),
        KnowledgeBridgerAdapter(VaderKnowledgeExtractor(), source_type="vader_lexicon",
                                source_version="kb_v3", source_license="MIT"),
        card_encoder,
        top_k=cli.top_k,
        projector=projector,
    )
    manifest = MaskManifest.load(args.mask_manifest)
    task = "regression" if cli.dataset in {"CMUMOSI", "CMUMOSEI"} else "classification"
    records = []
    with torch.no_grad():
        for batch_index, full_cpu in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            batch = _apply_manifest(full_cpu, manifest, device)
            result = pipeline.build(batch)
            card_text_by_id = {
                str(card.knowledge_id): card.evidence_text
                for card in result.cards
            }
            prompt = model(batch.features, batch.availability, evidence=None,
                           return_intermediates=True,
                           route_override=torch.zeros_like(batch.availability).to(
                               batch.features[MODALITIES[0]].dtype))
            empty_logits = prompt.logits  # [B,T,C]
            labels = batch.labels
            for b in range(batch.sequence_mask.size(0)):
                for t in range(batch.sequence_mask.size(1)):
                    if not bool(batch.sequence_mask[b, t]):
                        continue
                    for target_index, target in enumerate(MODALITIES):
                        if args.max_queries and len(records) >= args.max_queries:
                            break
                        if bool(batch.availability[b, t, target_index]):
                            continue
                        evidence = result.evidence_by_target[target]
                        if not bool(evidence.mask[b, t].any()):
                            continue
                        ranks = evidence.mask[b, t].nonzero(as_tuple=False).flatten().tolist()
                        card_logits = []
                        for rank in ranks:
                            isolated = isolate(evidence, b, t, rank)
                            out = model(batch.features, batch.availability,
                                        evidence=isolated, return_intermediates=True,
                                        route_override=torch.full_like(
                                            batch.availability, args.gamma,
                                            dtype=batch.features[MODALITIES[0]].dtype))
                            card_logits.append(out.logits[b, t])
                        empty = empty_logits[b, t]
                        y = labels[b, t]
                        if task == "classification":
                            ce_empty = F.cross_entropy(empty.reshape(1, -1),
                                                       y.reshape(1).long()).item()
                            ce_cards = [
                                F.cross_entropy(x.reshape(1, -1), y.reshape(1).long()).item()
                                for x in card_logits
                            ]
                            corr_empty = int(empty.argmax().item() == int(y.item()))
                            corr_cards = [
                                int(x.argmax().item() == int(y.item())) for x in card_logits
                            ]
                            u_ce = [ce_empty - value for value in ce_cards]
                            u_corr = [c - corr_empty for c in corr_cards]
                            u_mae = [0.0] * len(ranks)
                        else:
                            ce_empty = F.smooth_l1_loss(empty.float(), y.float().reshape(1)).item()
                            ce_cards = [
                                F.smooth_l1_loss(x.float(), y.float().reshape(1)).item()
                                for x in card_logits
                            ]
                            u_ce = [ce_empty - value for value in ce_cards]
                            u_corr = [0.0] * len(ranks)
                            u_mae = list(u_ce)
                        records.append({
                            "sample_id": str(batch.sample_ids[b]),
                            "dialogue_id": str(batch.sample_ids[b]),
                            "utterance_id": str(batch.utterance_ids[b][t]),
                            "target_modality": target_index,
                            "availability": batch.availability[b, t].tolist(),
                            "missing_pattern": int("".join(
                                str(int(v)) for v in
                                (~batch.availability[b, t]).to(torch.int).tolist()), 2),
                            "missing_rate": float((~batch.availability[b, t]).float().mean()),
                            "label": float(y),
                            "candidate_ids": [str(evidence.ids[b][t][r]) for r in ranks],
                            "candidate_texts": [
                                card_text_by_id.get(
                                    str(evidence.ids[b][t][r]), ""
                                )
                                for r in ranks
                            ],
                            "candidate_mask": [True] * len(ranks),
                            "empty_logits": empty.cpu(),
                            "empty_regression": (
                                float(empty.float().reshape(-1)[0])
                                if task == "regression" else None
                            ),
                            "candidate_regression": (
                                [float(x.float().reshape(-1)[0]) for x in card_logits]
                                if task == "regression" else None
                            ),
                            "u_ce": u_ce,
                            "u_corr": u_corr,
                            "u_mae": u_mae,
                        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "split": args.split,
            "task": task,
            "gamma": args.gamma,
            "records": records,
            "student_checkpoint_sha256": sha256_file(args.student_checkpoint),
            "card_adapter_sha256": sha256_file(args.card_adapter_checkpoint),
            "mask_manifest_sha256": sha256_file(args.mask_manifest),
            "test_accessed": False,
        },
        output,
    )
    print(json.dumps({"records": len(records), "split": args.split}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
