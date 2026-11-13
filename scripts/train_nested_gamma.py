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

from ComP.eval_knowledge import oracle_card
from ComP.knowledge.pipeline import knowledge_forward
from ComP.per.batch import PERDatasetAdapter
from ComP.per.losses import masked_task_loss
from ComP.per.missing import MaskManifest, MissingPatternSampler
from ComP.per.types import MODALITIES
from ComP.train_knowledge import (
    _build_knowledge_student,
    build_pipeline,
    ensure_manifest,
    evaluate_card_model,
)
from ComP.train_per import (
    _apply_manifest,
    _check_feature_paths,
    _dataset_and_splits,
    _make_loader,
    _metrics,
)


def evaluate_empty(model, loader, manifest, device, task, dataset):
    labels, predictions, masks = [], [], []
    with torch.no_grad():
        for full_cpu in loader:
            batch = _apply_manifest(full_cpu, manifest, device)
            output = model(
                batch.features,
                batch.availability,
                evidence=None,
                return_intermediates=True,
                route_override=torch.zeros_like(batch.availability).to(
                    batch.features[MODALITIES[0]].dtype
                ),
            )
            prediction = (
                output.logits.squeeze(-1)
                if task == "regression"
                else output.logits.argmax(dim=-1)
            )
            labels.append(batch.labels.cpu())
            predictions.append(prediction.cpu())
            masks.append(batch.sequence_mask.cpu())
    return _metrics(
        torch.cat(labels), torch.cat(predictions), torch.cat(masks), dataset
    )


def evaluate_oracle(model, pipeline, loader, manifest, device, task, dataset, gamma):
    labels, predictions, masks = [], [], []
    with torch.no_grad():
        for full_cpu in loader:
            batch = _apply_manifest(full_cpu, manifest, device)
            result = pipeline.build(batch)
            chosen, _, _ = oracle_card(
                model, batch, result, gamma=gamma, task=task, objective="task"
            )
            output = knowledge_forward(model, batch, chosen, gamma=gamma)
            prediction = (
                output.logits.squeeze(-1)
                if task == "regression"
                else output.logits.argmax(dim=-1)
            )
            labels.append(batch.labels.cpu())
            predictions.append(prediction.cpu())
            masks.append(batch.sequence_mask.cpu())
    return _metrics(
        torch.cat(labels), torch.cat(predictions), torch.cat(masks), dataset
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    args = parser.parse_args()
    manifest_data = json.loads(Path(args.split_manifest).read_text())
    fold = manifest_data["fold_assignment"][f"fold{args.fold}"]
    train_dialogues = set(fold["train"])
    val_dialogues = set(fold["validation"])
    payload = torch.load(
        args.student_checkpoint, map_location="cpu", weights_only=False
    )
    saved = dict(payload["args"])
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    saved.update({"no_cuda": device.type == "cpu", "gpu": args.gpu, "device": device})
    cli = Namespace(**saved)
    _check_feature_paths(cli)
    dataset, splits, _ = _dataset_and_splits(cli)
    split = splits[0]
    train_indices = [
        index for index in split["train"]
        if dataset.vids[index] in train_dialogues
    ]
    val_indices = [
        index for index in split["train"]
        if dataset.vids[index] in val_dialogues
    ]
    if not train_indices or not val_indices:
        raise SystemExit("empty nested fold")
    train_loader = _make_loader(
        PERDatasetAdapter(dataset, train_indices), cli, training=False
    )
    val_loader = _make_loader(
        PERDatasetAdapter(dataset, val_indices), cli, training=False
    )
    adim, tdim, vdim = dataset.get_featDim()
    evidence_dim = adim + tdim + vdim
    pipeline, projector, bank = build_pipeline(
        train_loader,
        adim=adim,
        tdim=tdim,
        vdim=vdim,
        evidence_dim=evidence_dim,
        top_k=cli.top_k,
        seed=cli.seed,
    )
    projector.to(device)
    model = _build_knowledge_student(
        cli, device, adim, tdim, vdim, evidence_dim, payload
    )
    model.eval()
    model.requires_grad_(False)
    model.evidence_encoder.requires_grad_(True)
    optimizer = torch.optim.Adam(
        list(projector.parameters()) + list(model.evidence_encoder.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_manifest_path = ensure_manifest(
        train_loader,
        MissingPatternSampler(
            seed=cli.seed, missing_rate=cli.mask_rate,
            protocol=cli.missing_protocol,
        ),
        str(output / "train_mask_manifest.csv"),
    )
    val_manifest_path = ensure_manifest(
        val_loader,
        MissingPatternSampler(
            seed=cli.seed + 7919, missing_rate=cli.mask_rate,
            protocol=cli.missing_protocol,
        ),
        str(output / "val_mask_manifest.csv"),
    )
    train_manifest = MaskManifest.load(train_manifest_path)
    val_manifest = MaskManifest.load(val_manifest_path)
    task = "regression" if cli.dataset in {"CMUMOSI", "CMUMOSEI"} else "classification"
    history = []
    best = float("inf")
    best_epoch = -1
    stale = 0
    for epoch in range(args.epochs):
        model.eval()
        projector.train()
        total = 0.0
        count = 0
        for full_cpu in train_loader:
            batch = _apply_manifest(full_cpu, train_manifest, device)
            result = pipeline.build(batch)
            output_batch = knowledge_forward(model, batch, result, gamma=args.gamma)
            loss = masked_task_loss(
                output_batch.logits, batch.labels, batch.sequence_mask, task
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite nested loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(projector.parameters())
                + list(model.evidence_encoder.parameters()),
                5.0,
            )
            optimizer.step()
            total += float(loss.detach().item())
            count += 1
            if args.max_train_batches and count >= args.max_train_batches:
                break
        metrics = evaluate_card_model(
            model,
            pipeline,
            val_loader,
            val_manifest,
            device,
            gamma=args.gamma,
            task=task,
            dataset=cli.dataset,
            max_batches=args.max_val_batches,
        )
        history.append(
            {"epoch": epoch, "train_loss": total / max(1, count), "validation": metrics}
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        if metrics["mae"] < best:
            best = metrics["mae"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "projector": projector.state_dict(),
                    "evidence_encoder": model.evidence_encoder.state_dict(),
                    "gamma": args.gamma,
                    "fold": args.fold,
                    "best_epoch": best_epoch,
                    "metrics": metrics,
                },
                output / "best_card_adapter.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    empty = evaluate_empty(model, val_loader, val_manifest, device, task, cli.dataset)
    card = evaluate_card_model(
        model,
        pipeline,
        val_loader,
        val_manifest,
        device,
        gamma=args.gamma,
        task=task,
        dataset=cli.dataset,
    )
    oracle = evaluate_oracle(
        model, pipeline, val_loader, val_manifest, device, task, cli.dataset, args.gamma
    )
    row = {
        "gamma": args.gamma,
        "fold": args.fold,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_dialogues": len(train_dialogues),
        "val_dialogues": len(val_dialogues),
        "empty_mae": empty["mae"],
        "empty_acc": empty["acc"],
        "card_mae": card["mae"],
        "card_acc": card["acc"],
        "card_f1": card["f1"],
        "oracle_mae": oracle["mae"],
        "oracle_acc": oracle["acc"],
        "card_minus_empty_mae": card["mae"] - empty["mae"],
        "card_minus_empty_acc": card["acc"] - empty["acc"],
        "oracle_minus_empty_mae": oracle["mae"] - empty["mae"],
        "oracle_minus_empty_acc": oracle["acc"] - empty["acc"],
    }
    with (output / "fold_result.json").open("w", encoding="utf-8") as handle:
        json.dump({"row": row, "history": history}, handle, indent=2, sort_keys=True)
    print("FOLD_RESULT", json.dumps(row, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
