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

from ComP.knowledge.msc.cache import subset_masks_batch
from ComP.knowledge.msc.policy import MSCActionSelector, MSCPolicy
from ComP.knowledge.msc.state import MSCStateEncoder
from ComP.knowledge.pipeline import knowledge_forward
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
    parser.add_argument("--msc-checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--tau-skip", type=float, default=0.5)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("test split requires --allow-test")

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
    model = _build_knowledge_student(
        cli, device, adim, tdim, vdim, evidence_dim, student
    )
    model.eval()
    adapter = torch.load(args.card_adapter_checkpoint, map_location="cpu", weights_only=False)
    model.evidence_encoder.load_state_dict(adapter["evidence_encoder"], strict=True)
    bank = KnowledgeContextRetriever.load(args.knowledge_index)
    pipeline, projector, _ = build_pipeline(
        train_loader,
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

    ckpt = torch.load(args.msc_checkpoint, map_location="cpu", weights_only=False)
    state_encoder = MSCStateEncoder(modality_count=3, hidden_dim=ckpt["state_dim"])
    policy = MSCPolicy(
        ckpt["candidate_dim"], ckpt["state_dim"], ckpt["top_k"],
        hidden_dim=ckpt.get("hidden_dim", 256),
        use_set_scorer=ckpt["use_set_scorer"], use_skip_head=ckpt["use_skip_head"],
    )
    state_sd = {
        key[len("state_encoder."):]: value
        for key, value in ckpt["state_dict"].items()
        if key.startswith("state_encoder.")
    }
    policy_sd = {
        key[len("policy."):]: value
        for key, value in ckpt["state_dict"].items()
        if key.startswith("policy.")
    }
    state_encoder.load_state_dict(state_sd, strict=True)
    policy.load_state_dict(policy_sd, strict=True)
    state_encoder.to(device)
    policy.to(device)
    state_encoder.eval()
    policy.eval()
    selector = MSCActionSelector(tau_skip=args.tau_skip)

    manifest = MaskManifest.load(args.mask_manifest)
    task = "regression" if cli.dataset in {"CMUMOSI", "CMUMOSEI"} else "classification"
    labels_all, predictions_all, masks_all = [], [], []
    per_sample_rows = []
    action_rows = []
    with torch.no_grad():
        for batch_index, full_cpu in enumerate(target_loader):
            batch = _apply_manifest(full_cpu, manifest, device)
            result = pipeline.build(batch)
            positions = []
            obs_list, tgt_list, pat_list, rate_list = [], [], [], []
            cv_list, cm_list = [], []
            for target_index, target in enumerate(MODALITIES):
                evidence = result.evidence_by_target[target]
                projected = model.evidence_encoder.projections[target](
                    evidence.features
                )
                for b in range(batch.sequence_mask.size(0)):
                    for t in range(batch.sequence_mask.size(1)):
                        if not bool(batch.sequence_mask[b, t]):
                            continue
                        if bool(batch.availability[b, t, target_index]):
                            continue
                        valid = evidence.mask[b, t]
                        if not bool(valid.any()):
                            continue
                        positions.append((b, t, target_index, target))
                        observed = batch.availability[b, t].float().cpu()
                        missing = (1.0 - observed)
                        pattern = int("".join(str(int(v)) for v in missing.tolist()), 2)
                        obs_list.append(observed)
                        tgt_list.append(
                            torch.nn.functional.one_hot(
                                torch.tensor(target_index), num_classes=3
                            ).float()
                        )
                        pat_list.append(pattern)
                        rate_list.append(float(missing.mean()))
                        cv_list.append(projected[b, t].cpu())
                        cm_list.append(valid.cpu())
            if not positions:
                output = model(
                    batch.features, batch.availability, evidence=None,
                    return_intermediates=True,
                    route_override=torch.zeros_like(batch.availability).to(
                        batch.features[MODALITIES[0]].dtype
                    ),
                )
            else:
                cv = torch.stack(cv_list).to(device)
                cm = torch.stack(cm_list).to(device)
                obs = torch.stack(obs_list).to(device)
                tgt = torch.stack(tgt_list).to(device)
                pat = torch.tensor(pat_list, dtype=torch.long, device=device)
                rate = torch.tensor(rate_list, dtype=torch.float32, device=device)
                subset_masks, _ = subset_masks_batch(cm, cm.size(1))
                state_vector = state_encoder(obs, tgt, pat, rate)
                out = policy(cv, cm, state_vector, subset_masks)
                selected = selector.select(
                    out["subset_scores"], subset_masks, out["skip_logit"]
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
                            if value.conflict_score is not None
                            else None
                        ),
                    )
                for row, (b, t, target_index, target) in enumerate(positions):
                    keep = selected["subset_mask"][row]
                    if bool(selected["skip"][row]):
                        action_rows.append(
                            {
                                "sample_id": batch.sample_ids[b],
                                "utterance_id": batch.utterance_ids[b][t],
                                "target_modality": target,
                                "action": 0,
                                "subset_size": 0,
                            }
                        )
                        continue
                    value = result.evidence_by_target[target]
                    zeroed[target].mask[b, t] = keep
                    zeroed[target].source_confidence[b, t] = (
                        value.source_confidence[b, t] * keep.to(value.source_confidence.dtype)
                    )
                    if zeroed[target].conflict_score is not None:
                        zeroed[target].conflict_score[b, t] = (
                            value.conflict_score[b, t] * keep.to(value.conflict_score.dtype)
                        )
                    action_rows.append(
                        {
                            "sample_id": batch.sample_ids[b],
                            "utterance_id": batch.utterance_ids[b][t],
                            "target_modality": target,
                            "action": 1,
                            "subset_size": int(keep.sum()),
                        }
                    )
                from ComP.knowledge.pipeline import KnowledgeBatchResult

                result = KnowledgeBatchResult(
                    result.queries, result.contexts, result.cards,
                    result.candidates, zeroed,
                )
                output = knowledge_forward(model, batch, result, gamma=args.gamma)
            prediction = (
                output.logits.squeeze(-1)
                if task == "regression"
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
                    per_sample_rows.append(
                        {
                            "mode": "msc_filter",
                            "gamma": args.gamma,
                            "split": args.split,
                            "sample_id": batch.sample_ids[b],
                            "dialogue_id": batch.sample_ids[b],
                            "utterance_id": batch.utterance_ids[b][t],
                            "availability": "".join(
                                "1" if bool(v) else "0"
                                for v in batch.availability[b, t].tolist()
                            ),
                            "missing_count": int((~batch.availability[b, t]).sum()),
                            "label": label,
                            "prediction": pred,
                            "abs_error": abs(pred - label),
                            "correct": int(
                                (pred > 0) == (label > 0)
                            )
                            if task == "regression"
                            else int(pred == label),
                        }
                    )
            if args.max_batches and batch_index + 1 >= args.max_batches:
                break
    metrics = _metrics(
        torch.cat(labels_all), torch.cat(predictions_all), torch.cat(masks_all), cli.dataset
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "split": args.split,
        "mode": "msc_filter",
        "gamma": args.gamma,
        "tau_skip": args.tau_skip,
        "metrics": metrics,
        "use_rate": (
            sum(1 for row in action_rows if row["action"] == 1) / max(1, len(action_rows))
        ),
        "skip_rate": (
            sum(1 for row in action_rows if row["action"] == 0) / max(1, len(action_rows))
        ),
        "student_checkpoint_sha256": sha256_file(args.student_checkpoint),
        "card_adapter_sha256": sha256_file(args.card_adapter_checkpoint),
        "mask_sha256": sha256_file(args.mask_manifest),
        "test_used": args.split == "test",
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    with (output / "per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_sample_rows)
    with (output / "action_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(action_rows[0].keys()))
        writer.writeheader()
        writer.writerows(action_rows)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
