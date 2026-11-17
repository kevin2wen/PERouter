from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.knowledge.kb_v4 import CrossModalPrototypeBuilder
from ComP.per.batch import PERDatasetAdapter
from ComP.per.missing import MaskManifest
from ComP.train_per import _apply_manifest, _dataset_and_splits, _make_loader


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--mask-manifest", required=True)
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    payload = torch.load(args.student_checkpoint, map_location="cpu", weights_only=False)
    from argparse import Namespace

    cli = Namespace(**payload["args"])
    cli.no_cuda = True
    dataset, splits, _ = _dataset_and_splits(cli)
    loader = _make_loader(
        PERDatasetAdapter(dataset, splits[0]["train"]), cli, training=False
    )
    manifest = MaskManifest.load(args.mask_manifest)
    masked = []
    for batch in loader:
        masked.append(
            _apply_manifest(
                batch, manifest, torch.device("cpu")
            )
        )
    class MaskedLoader(list):
        pass

    builder = CrossModalPrototypeBuilder(
        n_clusters=args.n_clusters, seed=66
    )
    builder.fit(MaskedLoader(masked))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "card_bank_k2u.jsonl").open("w", encoding="utf-8") as handle:
        for card in builder.cards:
            handle.write(json.dumps(dataclasses.asdict(card), sort_keys=True) + "\n")
    report = {
        "card_bank": "k2u",
        "cards": len(builder.cards),
        "n_clusters": args.n_clusters,
        "by_missing_modality": dict(
            Counter(card.handled_missing_modalities[0] for card in builder.cards)
        ),
        "label_used": False,
        "target_raw_used_for_current_query": False,
        "mask_manifest_sha256": sha256_file(args.mask_manifest),
        "student_checkpoint_sha256": sha256_file(args.student_checkpoint),
        "cards_sha256": sha256_file(output / "card_bank_k2u.jsonl"),
        "test_accessed": False,
    }
    (output / "manifest_k2u.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
