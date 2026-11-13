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

from ComP.knowledge.retriever import KnowledgeContextRetriever
from ComP.per.batch import PERDatasetAdapter
from ComP.train_per import (
    _check_feature_paths,
    _dataset_and_splits,
    _make_loader,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = torch.load(
        args.student_checkpoint, map_location="cpu", weights_only=False
    )
    saved = dict(payload["args"])
    device = torch.device("cpu")
    saved.update({"no_cuda": True, "device": device, "gpu": 0})
    config = Namespace(**saved)
    if args.top_k is not None:
        config.top_k = int(args.top_k)
    _check_feature_paths(config)
    dataset, splits, _ = _dataset_and_splits(config)
    split = next(item for item in splits if int(item["fold"]) == 0)
    adim, tdim, vdim = dataset.get_featDim()
    train_loader = _make_loader(
        PERDatasetAdapter(dataset, split["train"]),
        config,
        training=False,
    )
    bank = KnowledgeContextRetriever(
        {"audio": adim, "text": tdim, "video": vdim},
        top_k=config.top_k,
        min_df=1,
    )
    for batch in train_loader:
        bank.add_batch(batch, split="train")
    bank.fit()
    output = Path(args.output)
    bank.save(output)
    manifest = {
        "source_split": "train",
        "source_fingerprint": bank.source_fingerprint(),
        "entry_count": len(bank),
        "modality_dims": bank.modality_dims,
        "top_k": bank.top_k,
        "index_sha256": sha256_file(output),
        "student_checkpoint": args.student_checkpoint,
        "student_checkpoint_sha256": sha256_file(
            Path(args.student_checkpoint)
        ),
        "test_accessed": False,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
