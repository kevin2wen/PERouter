from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ComP.per.batch import PERDatasetAdapter, move_batch_to_device
from ComP.per.evidence_adapter import (
    adapter_parameter_count,
    adapter_state_dict,
    configure_adapter_training,
    sha256_file,
)
from ComP.per.missing import MissingPatternSampler
from ComP.per.text_evidence import TextNeighborRetriever
from ComP.train_per import (
    _build_retriever,
    _build_student,
    _check_feature_paths,
    _dataset_and_splits,
    _load_evidence,
    _make_loader,
    masked_task_loss,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--student-checkpoint", required=True)
    p.add_argument("--evidence-source", choices=("feature", "text"), required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    return p


def main() -> int:
    cli = parser().parse_args()
    if cli.epochs <= 0:
        raise ValueError("epochs must be positive")
    payload = torch.load(cli.student_checkpoint, map_location="cpu", weights_only=False)
    saved = dict(payload["args"])
    device = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    saved.update({"no_cuda": device.type == "cpu", "gpu": cli.gpu, "device": device})
    args = Namespace(**saved)
    _check_feature_paths(args)
    dataset, splits, _ = _dataset_and_splits(args)
    split = next(item for item in splits if int(item["fold"]) == 0)
    adim, tdim, vdim = dataset.get_featDim()
    train_loader = _make_loader(
        PERDatasetAdapter(dataset, split["train"]), args, training=True
    )
    evidence_dim = adim + tdim + vdim
    if cli.evidence_source == "text":
        retriever = TextNeighborRetriever(
            {"audio": adim, "text": tdim, "video": vdim},
            top_k=args.top_k,
            device=torch.device("cpu"),
        )
        retriever.add_loader(train_loader)
    else:
        retriever = _build_retriever(train_loader, adim, tdim, vdim, args.top_k)

    model = _build_student(args, device, adim, tdim, vdim, evidence_dim)
    model.load_state_dict(payload["model"], strict=True)
    configure_adapter_training(model)
    optimizer = torch.optim.Adam(
        [p for p in model.evidence_encoder.parameters() if p.requires_grad],
        lr=cli.learning_rate,
        weight_decay=cli.weight_decay,
    )
    sampler = MissingPatternSampler(
        seed=args.seed,
        missing_rate=args.mask_rate,
        protocol=args.missing_protocol,
    )
    task = "regression" if args.dataset in {"CMUMOSI", "CMUMOSEI"} else "classification"
    history = []
    started = time.time()
    for epoch in range(cli.epochs):
        model.evidence_encoder.train()
        total_loss = 0.0
        batches = 0
        for full_cpu in train_loader:
            batch = move_batch_to_device(full_cpu, device)
            batch.availability = sampler.sample(
                batch.sample_ids, batch.sequence_mask, epoch
            ).to(device)
            if cli.evidence_source == "text":
                evidence = retriever.retrieve(
                    batch.features,
                    batch.availability,
                    batch.sequence_mask,
                    batch.sample_ids,
                    batch.utterance_ids,
                    raw_texts=batch.raw_texts,
                )
            else:
                evidence = _load_evidence(
                    batch, retriever, "feature", evidence_dim, args.top_k, seed=args.seed
                )
            route = torch.full(
                (batch.availability.size(0), batch.availability.size(1), 3),
                0.5,
                dtype=batch.features["audio"].dtype,
                device=device,
            )
            output = model(
                batch.features,
                batch.availability,
                evidence=evidence,
                route_override=route,
            )
            loss = masked_task_loss(
                output.logits, batch.labels, batch.sequence_mask, task
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite adapter loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.evidence_encoder.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().item())
            batches += 1
        epoch_loss = total_loss / max(1, batches)
        history.append({"epoch": epoch, "train_loss": epoch_loss})
        print(json.dumps(history[-1], sort_keys=True), flush=True)

    output = Path(cli.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1,
        "evidence_source": cli.evidence_source,
        "adapter_state": adapter_state_dict(model),
        "student_checkpoint": cli.student_checkpoint,
        "student_checkpoint_sha256": sha256_file(cli.student_checkpoint),
        "evidence_source_fingerprint": (
            retriever.source_fingerprint() if cli.evidence_source == "text" else None
        ),
        "fixed_source_weight": 0.5,
        "train_split": "train",
        "validation_used": False,
        "test_used": False,
        "teacher_used": False,
        "epochs": cli.epochs,
        "seed": args.seed,
        "top_k": args.top_k,
        "args": vars(args),
        "history": history,
    }
    adapter_path = output / f"evidence_adapter_{cli.evidence_source}.pt"
    torch.save(checkpoint, adapter_path)
    (output / "adapter_training_history.json").write_text(
        json.dumps(checkpoint["history"], indent=2), encoding="utf-8"
    )
    parameter_payload = {
        "adapter_parameter_count": adapter_parameter_count(model),
        "total_student_parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }
    (output / "adapter_parameter_count.json").write_text(
        json.dumps(parameter_payload, indent=2), encoding="utf-8"
    )
    summary = {
        "adapter_path": str(adapter_path),
        "adapter_sha256": sha256_file(str(adapter_path)),
        "evidence_source": cli.evidence_source,
        "epochs": cli.epochs,
        "final_train_loss": history[-1]["train_loss"],
        "elapsed_seconds": time.time() - started,
        **parameter_payload,
    }
    (output / "adapter_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
