from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from itertools import product
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.eval_knowledge import element_loss
from ComP.knowledge.msc.joint_action import (
    joint_action_space,
    joint_interaction_error,
    local_sum,
)
from ComP.knowledge.pipeline import KnowledgeBatchResult, knowledge_forward
from ComP.knowledge.retriever import KnowledgeContextRetriever
from ComP.per.batch import PERDatasetAdapter
from ComP.per.missing import MaskManifest
from ComP.per.types import MODALITIES, EvidenceBatch
from ComP.train_knowledge import _build_knowledge_student, build_pipeline
from ComP.train_per import _apply_manifest, _dataset_and_splits, _make_loader


def zeroed(result):
    out = {}
    for name in MODALITIES:
        value = result.evidence_by_target[name]
        out[name] = EvidenceBatch(
            ids=value.ids, features=value.features.clone(),
            mask=torch.zeros_like(value.mask),
            source_confidence=torch.zeros_like(value.source_confidence),
            conflict_score=(torch.zeros_like(value.conflict_score)
                            if value.conflict_score is not None else None),
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--card-adapter-checkpoint", required=True)
    parser.add_argument("--knowledge-index", required=True)
    parser.add_argument("--mask-manifest", required=True)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--utterance-limit", type=int, default=6)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    student = torch.load(args.student_checkpoint, map_location="cpu", weights_only=False)
    saved = dict(student["args"])
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    saved.update({"no_cuda": device.type == "cpu", "gpu": args.gpu, "device": device})
    cli = Namespace(**saved)
    dataset, splits, _ = _dataset_and_splits(cli)
    split = splits[0]
    train_loader = _make_loader(PERDatasetAdapter(dataset, split["train"]), cli, training=False)
    val_loader = _make_loader(PERDatasetAdapter(dataset, split["val"]), cli, training=False)
    adim, tdim, vdim = dataset.get_featDim()
    evidence_dim = adim + tdim + vdim
    model = _build_knowledge_student(cli, device, adim, tdim, vdim, evidence_dim, student)
    model.eval()
    adapter = torch.load(args.card_adapter_checkpoint, map_location="cpu", weights_only=False)
    model.evidence_encoder.load_state_dict(adapter["evidence_encoder"], strict=True)
    bank = KnowledgeContextRetriever.load(args.knowledge_index)
    pipeline, projector, _ = build_pipeline(
        train_loader, adim=adim, tdim=tdim, vdim=vdim, evidence_dim=evidence_dim,
        top_k=cli.top_k, seed=cli.seed, bank=bank,
    )
    projector.load_state_dict(adapter["projector"], strict=True)
    projector.to(device).eval()
    manifest = MaskManifest.load(args.mask_manifest)
    task = "regression" if cli.dataset in {"CMUMOSI", "CMUMOSEI"} else "classification"
    records = []
    with torch.no_grad():
        for full_cpu in val_loader:
            if len(records) >= args.utterance_limit:
                break
            batch = _apply_manifest(full_cpu, manifest, device)
            result = pipeline.build(batch)
            skip_output = model(
                batch.features, batch.availability, evidence=None, return_intermediates=True,
                route_override=torch.zeros_like(batch.availability).to(
                    batch.features[MODALITIES[0]].dtype),
            )
            skip_loss = element_loss(skip_output, batch.labels, task)
            for b in range(batch.sequence_mask.size(0)):
                if len(records) >= args.utterance_limit:
                    break
                for t in range(batch.sequence_mask.size(1)):
                    if len(records) >= args.utterance_limit:
                        break
                    if not bool(batch.sequence_mask[b, t]):
                        continue
                    targets = [
                        (index, name) for index, name in enumerate(MODALITIES)
                        if not bool(batch.availability[b, t, index])
                        and bool(result.evidence_by_target[name].mask[b, t].any())
                    ]
                    if len(targets) < 2:
                        continue
                    options = []
                    for index, name in targets:
                        ranks = result.evidence_by_target[name].mask[b, t].nonzero(
                            as_tuple=False).flatten().tolist()
                        options.append([None] + ranks)
                    best = None
                    combos = list(product(*options))
                    for combo in combos:
                        zero = zeroed(result)
                        for (index, name), rank in zip(targets, combo):
                            if rank is None:
                                continue
                            value = result.evidence_by_target[name]
                            zero[name].mask[b, t, rank] = True
                            zero[name].source_confidence[b, t, rank] = value.source_confidence[b, t, rank]
                            if zero[name].conflict_score is not None:
                                zero[name].conflict_score[b, t, rank] = value.conflict_score[b, t, rank]
                        fused = KnowledgeBatchResult(
                            result.queries, result.contexts, result.cards, result.candidates, zero)
                        output = knowledge_forward(model, batch, fused, gamma=args.gamma)
                        loss_value = float(element_loss(output, batch.labels, task)[b, t])
                        utility = float(skip_loss[b, t]) - loss_value
                        if best is None or utility > best["utility"]:
                            best = {
                                "utility": utility,
                                "combo": combo,
                                "used": sum(1 for rank in combo if rank is not None),
                            }
                    # independent local: best rank per target alone
                    local_best = 0.0
                    for (index, name), target_options in zip(targets, options):
                        single_best = -1e9
                        for rank in target_options:
                            zero = zeroed(result)
                            if rank is not None:
                                value = result.evidence_by_target[name]
                                zero[name].mask[b, t, rank] = True
                                zero[name].source_confidence[b, t, rank] = value.source_confidence[b, t, rank]
                            fused = KnowledgeBatchResult(
                                result.queries, result.contexts, result.cards, result.candidates, zero)
                            output = knowledge_forward(model, batch, fused, gamma=args.gamma)
                            utility = float(skip_loss[b, t]) - float(
                                element_loss(output, batch.labels, task)[b, t])
                            single_best = max(single_best, utility)
                        local_best += max(0.0, single_best)
                    records.append({
                        "sample_id": batch.sample_ids[b],
                        "utterance_id": batch.utterance_ids[b][t],
                        "targets": "+".join(name for _, name in targets),
                        "joint_oracle_utility": best["utility"],
                        "local_sum_utility": local_best,
                        "joint_interaction_error": joint_interaction_error(best["utility"], local_best),
                        "best_combo": str(best["combo"]),
                        "best_used_cards": best["used"],
                        "combo_count": len(combos),
                    })
    import statistics as st
    report = {
        "utterances": len(records),
        "mean_joint_oracle_utility": st.mean(r["joint_oracle_utility"] for r in records),
        "mean_local_sum_utility": st.mean(r["local_sum_utility"] for r in records),
        "mean_joint_interaction_error": st.mean(r["joint_interaction_error"] for r in records),
        "positive_oracle": sum(1 for r in records if r["joint_oracle_utility"] > 0),
        "local_positive_joint_negative": sum(
            1 for r in records if r["local_sum_utility"] > 0 and r["joint_oracle_utility"] < 0),
        "records": records,
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
