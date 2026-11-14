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

from ComP.knowledge.msc.calibration import load_calibration
from ComP.knowledge.msc.single_eu import MSCSingleEU, SingleEUSelector
from ComP.knowledge.pipeline import KnowledgeBatchResult, knowledge_forward
from ComP.knowledge.retriever import KnowledgeContextRetriever
from ComP.per.batch import PERDatasetAdapter
from ComP.per.missing import MaskManifest
from ComP.per.types import MODALITIES, EvidenceBatch
from ComP.train_knowledge import _build_knowledge_student, build_pipeline
from ComP.train_per import (
    _apply_manifest,
    _check_feature_paths,
    _dataset_and_splits,
    _make_loader,
    _metrics,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--card-adapter-checkpoint", required=True)
    parser.add_argument("--knowledge-index", required=True)
    parser.add_argument("--mask-manifest", required=True)
    parser.add_argument("--scorer-checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("test split requires --allow-test")
    ckpt = torch.load(args.scorer_checkpoint, map_location="cpu", weights_only=False)
    calibration = load_calibration(
        args.calibration, expected_scorer_hash=ckpt["utility_scorer_hash"]
    )
    student = torch.load(args.student_checkpoint, map_location="cpu", weights_only=False)
    saved = dict(student["args"])
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    saved.update({"no_cuda": device.type == "cpu", "gpu": args.gpu, "device": device})
    cli = Namespace(**saved)
    _check_feature_paths(cli)
    dataset, splits, _ = _dataset_and_splits(cli)
    split = splits[0]
    split_key = "val" if args.split == "validation" else "test"
    train_loader = _make_loader(
        PERDatasetAdapter(dataset, split["train"]), cli, training=False
    )
    target_loader = _make_loader(
        PERDatasetAdapter(dataset, split[split_key]), cli, training=False
    )
    adim, tdim, vdim = dataset.get_featDim()
    evidence_dim = adim + tdim + vdim
    model = _build_knowledge_student(cli, device, adim, tdim, vdim, evidence_dim, student)
    model.eval()
    adapter = torch.load(args.card_adapter_checkpoint, map_location="cpu", weights_only=False)
    model.evidence_encoder.load_state_dict(adapter["evidence_encoder"], strict=True)
    bank = KnowledgeContextRetriever.load(args.knowledge_index)
    pipeline, projector, _ = build_pipeline(
        train_loader, adim=adim, tdim=tdim, vdim=vdim,
        evidence_dim=evidence_dim, top_k=cli.top_k, seed=cli.seed, bank=bank,
    )
    projector.load_state_dict(adapter["projector"], strict=True)
    projector.to(device).eval()
    scorer = MSCSingleEU(
        ckpt["candidate_dim"], ckpt["prompt_dim"],
        state_dim=ckpt["state_dim"], hidden_dim=ckpt["hidden_dim"],
        use_semantic=ckpt["use_semantic"],
        use_cross_attention=ckpt["use_cross_attention"],
        use_skip_head=ckpt["use_skip_head"],
    )
    scorer.load_state_dict(ckpt["state_dict"], strict=True)
    scorer.to(device).eval()
    semantic_encoder = None
    if ckpt.get("candidate_feature_kind") == "semantic":
        from ComP.knowledge.msc.card_encoder import SentenceCardEncoder

        semantic_encoder = SentenceCardEncoder(
            ckpt["semantic_model_path"], device="cpu"
        )
    selector = SingleEUSelector(
        tau_skip=float(calibration["tau_skip"]),
        utility_threshold=0.0,
    )
    temperature = float(calibration.get("temperature", 1.0))
    manifest = MaskManifest.load(args.mask_manifest)
    task = "regression" if cli.dataset in {"CMUMOSI", "CMUMOSEI"} else "classification"
    labels_all, predictions_all, masks_all = [], [], []
    per_sample, actions = [], []
    with torch.no_grad():
        for batch_index, full_cpu in enumerate(target_loader):
            batch = _apply_manifest(full_cpu, manifest, device)
            result = pipeline.build(batch)
            semantic_vectors = {}
            if semantic_encoder is not None:
                card_by_id = {
                    str(card.knowledge_id): card for card in result.cards
                }
                unique_ids = sorted(card_by_id)
                if unique_ids:
                    embeddings = semantic_encoder.encode(
                        [
                            semantic_encoder.card_text(
                                {
                                    "evidence_text": card_by_id[card_id].evidence_text,
                                    "trigger_text": card_by_id[card_id].trigger_text,
                                    "relation_type": card_by_id[card_id].relation_type,
                                }
                            )
                            for card_id in unique_ids
                        ]
                    )
                    semantic_vectors = {
                        card_id: torch.from_numpy(embeddings[index])
                        for index, card_id in enumerate(unique_ids)
                    }
            prompt_output = model(
                batch.features, batch.availability, evidence=None,
                return_intermediates=True,
                route_override=torch.zeros_like(batch.availability).to(
                    batch.features[MODALITIES[0]].dtype
                ),
            )
            zeroed = {}
            for name in MODALITIES:
                value = result.evidence_by_target[name]
                zeroed[name] = EvidenceBatch(
                    ids=value.ids,
                    features=value.features.clone(),
                    mask=torch.zeros_like(value.mask),
                    source_confidence=torch.zeros_like(value.source_confidence),
                    conflict_score=(
                        torch.zeros_like(value.conflict_score)
                        if value.conflict_score is not None else None
                    ),
                )
            for target_index, target in enumerate(MODALITIES):
                evidence = result.evidence_by_target[target]
                projected = (
                    model.evidence_encoder.projections[target](evidence.features)
                    if semantic_encoder is None
                    else None
                )
                rows, obs, tgt, pat, rate, prompts, cv, cm, conf, conflict = (
                    [], [], [], [], [], [], [], [], [], []
                )
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
                        if semantic_encoder is None:
                            cv.append(projected[b, t].cpu())
                        else:
                            cv.append(
                                torch.stack(
                                    [
                                        semantic_vectors.get(
                                            str(evidence.ids[b][t][rank]),
                                            torch.zeros(
                                                ckpt["candidate_dim"]
                                            ),
                                        )
                                        for rank in range(evidence.mask.size(-1))
                                    ]
                                )
                            )
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
                chosen = selector.select(
                    out["utility"], torch.stack(cm).to(device),
                    out["skip_logit"] / temperature,
                )
                for row, (b, t) in enumerate(positions):
                    actions.append({
                        "sample_id": batch.sample_ids[b],
                        "utterance_id": batch.utterance_ids[b][t],
                        "target_modality": target,
                        "skip": bool(chosen["skip"][row]),
                        "utility": float(chosen["utility"][row]),
                        "skip_probability": float(chosen["skip_probability"][row]),
                        "selected_index": int(chosen["selected_index"][row]),
                    })
                    if bool(chosen["skip"][row]):
                        continue
                    rank = int(chosen["selected_index"][row])
                    value = result.evidence_by_target[target]
                    zeroed[target].mask[b, t, rank] = True
                    zeroed[target].source_confidence[b, t, rank] = value.source_confidence[b, t, rank]
                    if zeroed[target].conflict_score is not None:
                        zeroed[target].conflict_score[b, t, rank] = value.conflict_score[b, t, rank]
            fused = KnowledgeBatchResult(
                result.queries, result.contexts, result.cards, result.candidates, zeroed
            )
            output = knowledge_forward(model, batch, fused, gamma=args.gamma)
            prediction = (
                output.logits.squeeze(-1) if task == "regression"
                else output.logits.argmax(dim=-1)
            ).detach().cpu()
            labels_all.append(batch.labels.cpu())
            predictions_all.append(prediction)
            masks_all.append(batch.sequence_mask.cpu())
            for b in range(batch.labels.size(0)):
                for t in range(batch.labels.size(1)):
                    if not bool(batch.sequence_mask[b, t]):
                        continue
                    label = float(batch.labels[b, t])
                    if task == "regression" and label == 0.0:
                        continue
                    pred = float(prediction[b, t])
                    per_sample.append({
                        "mode": "singleeu", "gamma": args.gamma, "split": args.split,
                        "sample_id": batch.sample_ids[b], "dialogue_id": batch.sample_ids[b],
                        "utterance_id": batch.utterance_ids[b][t],
                        "availability": "".join("1" if bool(v) else "0"
                                                 for v in batch.availability[b, t].tolist()),
                        "missing_count": int((~batch.availability[b, t]).sum()),
                        "label": label, "prediction": pred,
                        "abs_error": abs(pred - label),
                        "correct": int((pred > 0) == (label > 0)) if task == "regression"
                        else int(pred == label),
                    })
            if args.max_batches and batch_index + 1 >= args.max_batches:
                break
    metrics = _metrics(
        torch.cat(labels_all), torch.cat(predictions_all), torch.cat(masks_all), cli.dataset
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "split": args.split, "mode": "singleeu", "gamma": args.gamma,
        "metrics": metrics,
        "use_rate": float(sum(not a["skip"] for a in actions) / max(1, len(actions))),
        "skip_rate": float(sum(a["skip"] for a in actions) / max(1, len(actions))),
        "scorer_checkpoint_sha256": sha256_file(args.scorer_checkpoint),
        "calibration_sha256": sha256_file(args.calibration),
        "test_used": args.split == "test",
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output_dir / "per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_sample[0].keys()))
        writer.writeheader(); writer.writerows(per_sample)
    with (output_dir / "action_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(actions[0].keys()))
        writer.writeheader(); writer.writerows(actions)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
