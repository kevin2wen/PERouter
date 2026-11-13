from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.knowledge.adapters.knowledge_bridger_adapter import (
    KnowledgeBridgerAdapter,
)
from ComP.knowledge.adapters.ragpt_adapter import RAGPTAdapter
from ComP.knowledge.encoder import (
    RuleKnowledgeCardEncoder,
    TargetSpecificCardProjector,
)
from ComP.knowledge.extractor import RuleBasedKnowledgeExtractor
from ComP.knowledge.pipeline import KnowledgePipeline, knowledge_forward
from ComP.knowledge.retriever import KnowledgeContextRetriever
from ComP.per.batch import PERDatasetAdapter, move_batch_to_device
from ComP.per.losses import masked_task_loss
from ComP.per.missing import MaskManifest, MissingPatternSampler
from ComP.train_per import (
    _apply_manifest,
    _build_student,
    _check_feature_paths,
    _collect_split_masks,
    _dataset_and_splits,
    _make_loader,
    _metrics,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--student-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--train-mask-manifest")
    p.add_argument("--val-mask-manifest")
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--knowledge-index")
    p.add_argument("--gpu", type=int, default=0)
    return p


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_pipeline(
    train_loader,
    *,
    adim: int,
    tdim: int,
    vdim: int,
    evidence_dim: int,
    top_k: int,
    seed: int,
    bank: Optional[KnowledgeContextRetriever] = None,
) -> tuple[KnowledgePipeline, TargetSpecificCardProjector, KnowledgeContextRetriever]:
    if bank is None:
        bank = KnowledgeContextRetriever(
            {"audio": adim, "text": tdim, "video": vdim},
            top_k=top_k,
            min_df=1,
        )
        for batch in train_loader:
            bank.add_batch(batch, split="train")
        bank.fit()
    retrieval = RAGPTAdapter(
        bank.retrieve,
        top_k=top_k,
        bank_split="train",
    )
    bridger = KnowledgeBridgerAdapter(
        RuleBasedKnowledgeExtractor(),
        source_type="train_retrieved_rule",
        source_version="knowledge-rule-v1",
        source_license="research-only",
    )
    card_encoder = RuleKnowledgeCardEncoder(adim, vdim)
    projector = TargetSpecificCardProjector(
        card_encoder.output_dim,
        evidence_dim,
        zero_init=True,
    )
    pipeline = KnowledgePipeline(
        retrieval,
        bridger,
        card_encoder,
        top_k=top_k,
        projector=projector,
    )
    return pipeline, projector, bank


def ensure_manifest(loader, sampler, path: str) -> str:
    manifest = Path(path)
    if not manifest.exists():
        _collect_split_masks(
            loader,
            sampler,
            epoch=0,
            output_path=str(manifest),
        )
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    return str(manifest)


def evaluate_card_model(
    model,
    pipeline,
    loader,
    manifest,
    device,
    *,
    gamma: float,
    task: str,
    dataset: str,
    max_batches: int = 0,
) -> Dict[str, float]:
    model.eval()
    pipeline.projector.eval()
    total_loss = 0.0
    count = 0
    labels = []
    predictions = []
    masks = []
    with torch.no_grad():
        for full_cpu in loader:
            batch = _apply_manifest(full_cpu, manifest, device)
            result = pipeline.build(batch)
            output = knowledge_forward(model, batch, result, gamma=gamma)
            loss = masked_task_loss(
                output.logits, batch.labels, batch.sequence_mask, task
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite validation card loss")
            total_loss += float(loss.item())
            count += 1
            labels.append(batch.labels.cpu())
            masks.append(batch.sequence_mask.cpu())
            prediction = (
                output.logits.squeeze(-1)
                if task == "regression"
                else output.logits.argmax(dim=-1)
            )
            predictions.append(prediction.cpu())
            if max_batches and count >= max_batches:
                break
    metrics = _metrics(
        torch.cat(labels),
        torch.cat(predictions),
        torch.cat(masks),
        dataset,
    )
    metrics["loss"] = total_loss / max(1, count)
    return metrics


def _build_knowledge_student(
    args,
    device,
    adim: int,
    tdim: int,
    vdim: int,
    evidence_dim: int,
    payload: Dict[str, object],
):
    """Attach the PER evidence branch to a frozen fair-B0 student.

    Fair B0 checkpoints are trained with ``per_mode=off`` and therefore carry
    no ``evidence_encoder``. The knowledge route injects target-specific
    evidence vectors, so the branch must exist. Rebuild with ``per_mode=fixed``
    (the deterministic route is overridden per position later) and load the
    checkpoint non-strictly, accepting only the newly introduced branch keys
    as missing.
    """
    from argparse import Namespace

    build_args = Namespace(**vars(args))
    build_args.per_mode = "fixed"
    build_args.fixed_source_weight = float(
        getattr(args, "fixed_source_weight", 0.5)
    )
    model = _build_student(build_args, device, adim, tdim, vdim, evidence_dim)
    incompatible = model.load_state_dict(payload["model"], strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "student checkpoint has unexpected keys: "
            f"{list(incompatible.unexpected_keys)[:8]}"
        )
    allowed_prefixes = ("evidence_encoder.", "observed_projections.", "router.")
    for key in incompatible.missing_keys:
        if not key.startswith(allowed_prefixes):
            raise RuntimeError(
                f"student checkpoint misses non-branch parameter: {key}"
            )
    if getattr(model, "evidence_encoder", None) is None:
        raise RuntimeError("failed to attach PER evidence encoder")
    return model


def main() -> int:
    cli = parser().parse_args()
    if cli.epochs <= 0:
        raise ValueError("epochs must be positive")
    payload = torch.load(
        cli.student_checkpoint, map_location="cpu", weights_only=False
    )
    saved = dict(payload["args"])
    device = torch.device(
        f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu"
    )
    saved.update(
        {"no_cuda": device.type == "cpu", "gpu": cli.gpu, "device": device}
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
    val_loader = _make_loader(
        PERDatasetAdapter(dataset, split["val"]), args, training=False
    )
    evidence_dim = adim + tdim + vdim
    bank = (
        KnowledgeContextRetriever.load(cli.knowledge_index)
        if cli.knowledge_index and Path(cli.knowledge_index).exists()
        else None
    )
    pipeline, projector, bank = build_pipeline(
        train_loader,
        adim=adim,
        tdim=tdim,
        vdim=vdim,
        evidence_dim=evidence_dim,
        top_k=args.top_k,
        seed=args.seed,
        bank=bank,
    )
    if cli.knowledge_index and not Path(cli.knowledge_index).exists():
        bank.save(cli.knowledge_index)
    projector.to(device)
    model = _build_knowledge_student(
        args, device, adim, tdim, vdim, evidence_dim, payload
    )
    model.eval()
    model.requires_grad_(False)
    model.evidence_encoder.requires_grad_(True)
    optimizer = torch.optim.Adam(
        list(projector.parameters())
        + list(model.evidence_encoder.parameters()),
        lr=cli.learning_rate,
        weight_decay=cli.weight_decay,
    )
    output = Path(cli.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_manifest_path = ensure_manifest(
        train_loader,
        MissingPatternSampler(
            seed=args.seed,
            missing_rate=args.mask_rate,
            protocol=args.missing_protocol,
        ),
        cli.train_mask_manifest
        or str(output / "train_mask_manifest.csv"),
    )
    val_manifest_path = ensure_manifest(
        val_loader,
        MissingPatternSampler(
            seed=args.seed + 7919,
            missing_rate=args.mask_rate,
            protocol=args.missing_protocol,
        ),
        cli.val_mask_manifest or str(output / "val_mask_manifest.csv"),
    )
    train_manifest = MaskManifest.load(train_manifest_path)
    val_manifest = MaskManifest.load(val_manifest_path)
    task = "regression" if args.dataset in {"CMUMOSI", "CMUMOSEI"} else "classification"
    history = []
    best_score = -float("inf")
    best_epoch = -1
    stale = 0
    for epoch in range(cli.epochs):
        model.eval()
        projector.train()
        total = 0.0
        count = 0
        for full_cpu in train_loader:
            batch = _apply_manifest(full_cpu, train_manifest, device)
            result = pipeline.build(batch)
            card_output = knowledge_forward(
                model, batch, result, gamma=cli.gamma
            )
            loss = masked_task_loss(
                card_output.logits,
                batch.labels,
                batch.sequence_mask,
                task,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite knowledge adapter loss")
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
            if cli.max_train_batches and count >= cli.max_train_batches:
                break
        train_loss = total / max(1, count)
        val_metrics = evaluate_card_model(
            model,
            pipeline,
            val_loader,
            val_manifest,
            device,
            gamma=cli.gamma,
            task=task,
            dataset=args.dataset,
            max_batches=cli.max_val_batches,
        )
        score = -val_metrics["mae"] if task == "regression" else val_metrics["acc"]
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation": val_metrics}
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            checkpoint = {
                "projector": projector.state_dict(),
                "evidence_encoder": model.evidence_encoder.state_dict(),
                "epoch": epoch,
                "metrics": val_metrics,
                "args": {
                    "top_k": args.top_k,
                    "gamma": cli.gamma,
                    "card_feature_dim": pipeline.encoder.output_dim,
                    "evidence_dim": evidence_dim,
                    "source_version": "knowledge-rule-v1",
                    "train_mask_manifest": train_manifest_path,
                    "val_mask_manifest": val_manifest_path,
                    "train_mask_sha256": sha256_file(train_manifest_path),
                    "val_mask_sha256": sha256_file(val_manifest_path),
                    "student_checkpoint": str(Path(cli.student_checkpoint)),
                    "student_checkpoint_sha256": sha256_file(
                        cli.student_checkpoint
                    ),
                    "source_fingerprint": bank.source_fingerprint(),
                    "build_per_mode": "fixed",
                },
                "format_version": 1,
            }
            torch.save(checkpoint, output / "best_card_adapter.pt")
        else:
            stale += 1
            if stale >= cli.patience:
                break
    result = {
        "student_checkpoint": str(Path(cli.student_checkpoint)),
        "student_checkpoint_sha256": sha256_file(cli.student_checkpoint),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "history": history,
        "projector_parameter_count": sum(
            parameter.numel() for parameter in projector.parameters()
        ),
        "evidence_encoder_parameter_count": sum(
            parameter.numel() for parameter in model.evidence_encoder.parameters()
        ),
        "test_accessed": False,
        "created_at": datetime.datetime.now().isoformat(),
    }
    (output / "training_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
