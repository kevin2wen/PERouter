from __future__ import annotations

import argparse
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.knowledge.msc.calibration import load_calibration
from ComP.knowledge.msc.single_eu import MSCSingleEU, SingleEUSelector
from ComP.knowledge.pipeline import KnowledgeBatchResult, knowledge_forward
from ComP.knowledge.retriever import KnowledgeContextRetriever
from ComP.per.batch import PERDatasetAdapter
from ComP.per.missing import MaskManifest
from ComP.per.types import MODALITIES, EvidenceBatch
from ComP.train_knowledge import _build_knowledge_student, build_pipeline
from ComP.train_per import _apply_manifest, _dataset_and_splits, _make_loader
from ComP.eval_knowledge import element_loss


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--card-adapter-checkpoint", required=True)
    parser.add_argument("--knowledge-index", required=True)
    parser.add_argument("--mask-manifest", required=True)
    parser.add_argument("--scorer-checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    ckpt = torch.load(args.scorer_checkpoint, map_location="cpu", weights_only=False)
    calibration = load_calibration(args.calibration, expected_scorer_hash=ckpt["utility_scorer_hash"])
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
    scorer = MSCSingleEU(
        ckpt["candidate_dim"], ckpt["prompt_dim"], state_dim=ckpt["state_dim"],
        hidden_dim=ckpt["hidden_dim"], use_semantic=ckpt["use_semantic"],
        use_cross_attention=ckpt["use_cross_attention"], use_skip_head=ckpt["use_skip_head"],
    )
    scorer.load_state_dict(ckpt["state_dict"], strict=True)
    scorer.to(device).eval()
    selector = SingleEUSelector(tau_skip=float(calibration["tau_skip"]))
    temperature = float(calibration.get("temperature", 1.0))
    manifest = MaskManifest.load(args.mask_manifest)
    task = "regression" if cli.dataset in {"CMUMOSI", "CMUMOSEI"} else "classification"
    rows = []
    with torch.no_grad():
        for full_cpu in val_loader:
            batch = _apply_manifest(full_cpu, manifest, device)
            result = pipeline.build(batch)
            prompt_output = model(
                batch.features, batch.availability, evidence=None, return_intermediates=True,
                route_override=torch.zeros_like(batch.availability).to(
                    batch.features[MODALITIES[0]].dtype),
            )
            skip_output = model(
                batch.features, batch.availability, evidence=None, return_intermediates=True,
                route_override=torch.zeros_like(batch.availability).to(
                    batch.features[MODALITIES[0]].dtype),
            )
            skip_loss = element_loss(skip_output, batch.labels, task)
            local = torch.zeros(batch.sequence_mask.shape, device=device)
            chosen = {}
            for target_index, target in enumerate(MODALITIES):
                evidence = result.evidence_by_target[target]
                projected = model.evidence_encoder.projections[target](evidence.features)
                obs, tgt, pat, rate, prompts, cv, cm, conf, conflict = [], [], [], [], [], [], [], [], []
                positions = []
                for b in range(batch.sequence_mask.size(0)):
                    for t in range(batch.sequence_mask.size(1)):
                        if not bool(batch.sequence_mask[b, t]):
                            continue
                        if bool(batch.availability[b, t, target_index]):
                            continue
                        valid = evidence.mask[b, t]
                        if not bool(valid.any()):
                            continue
                        positions.append((b, t))
                        observed = batch.availability[b, t].float().cpu()
                        missing = 1.0 - observed
                        obs.append(observed)
                        tgt.append(torch.nn.functional.one_hot(
                            torch.tensor(target_index), num_classes=3).float())
                        pat.append(int("".join(str(int(v)) for v in missing.tolist()), 2))
                        rate.append(float(missing.mean()))
                        prompts.append(prompt_output.prompt_candidates[target][b, t].cpu())
                        cv.append(projected[b, t].cpu())
                        cm.append(valid.cpu())
                        conf.append(evidence.source_confidence[b, t].cpu())
                        conflict.append(
                            evidence.conflict_score[b, t].cpu()
                            if evidence.conflict_score is not None
                            else torch.zeros_like(valid, dtype=torch.float32)
                        )
                if not positions:
                    continue
                out = scorer(
                    torch.stack(obs).to(device), torch.stack(tgt).to(device),
                    torch.tensor(pat, device=device), torch.tensor(rate, device=device),
                    torch.stack(prompts).to(device), torch.stack(cv).to(device),
                    torch.stack(cm).to(device), torch.stack(conf).to(device),
                    torch.stack(conflict).to(device),
                )
                decision = selector.select(
                    out["utility"], torch.stack(cm).to(device),
                    out["skip_logit"] / temperature,
                )
                for row, (b, t) in enumerate(positions):
                    if bool(decision["skip"][row]):
                        continue
                    rank = int(decision["selected_index"][row])
                    chosen[(target, b, t)] = rank
                    local[b, t] += float(decision["utility"][row])
            if not chosen:
                continue
            zeroed = {}
            for name in MODALITIES:
                value = result.evidence_by_target[name]
                zeroed[name] = EvidenceBatch(
                    ids=value.ids, features=value.features.clone(),
                    mask=torch.zeros_like(value.mask),
                    source_confidence=torch.zeros_like(value.source_confidence),
                    conflict_score=(torch.zeros_like(value.conflict_score)
                                    if value.conflict_score is not None else None),
                )
            for (target, b, t), rank in chosen.items():
                value = result.evidence_by_target[target]
                zeroed[target].mask[b, t, rank] = True
                zeroed[target].source_confidence[b, t, rank] = value.source_confidence[b, t, rank]
                if zeroed[target].conflict_score is not None:
                    zeroed[target].conflict_score[b, t, rank] = value.conflict_score[b, t, rank]
            fused = KnowledgeBatchResult(
                result.queries, result.contexts, result.cards, result.candidates, zeroed
            )
            output = knowledge_forward(model, batch, fused, gamma=args.gamma)
            joint = skip_loss - element_loss(output, batch.labels, task)
            for b in range(batch.sequence_mask.size(0)):
                for t in range(batch.sequence_mask.size(1)):
                    targets = [name for name in MODALITIES if (name, b, t) in chosen]
                    if len(targets) < 2:
                        continue
                    rows.append(
                        {
                            "sample_id": batch.sample_ids[b],
                            "utterance_id": batch.utterance_ids[b][t],
                            "targets": "+".join(targets),
                            "sum_local_utility": float(local[b, t]),
                            "realized_joint_utility": float(joint[b, t]),
                            "joint_interaction_error": float(joint[b, t] - local[b, t]),
                        }
                    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output_dir / "joint_actions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    errors = [row["joint_interaction_error"] for row in rows]
    summary = {
        "utterances": len(rows),
        "mean_sum_local_utility": float(sum(r["sum_local_utility"] for r in rows) / max(1, len(rows))),
        "mean_realized_joint_utility": float(sum(r["realized_joint_utility"] for r in rows) / max(1, len(rows))),
        "mean_joint_interaction_error": float(sum(errors) / max(1, len(errors))),
        "positive_local_negative_joint": int(
            sum(r["sum_local_utility"] > 0 and r["realized_joint_utility"] < 0 for r in rows)
        ),
    }
    (output_dir / "joint_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
