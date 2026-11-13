from __future__ import annotations

import argparse
import json
import random
import sys
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from ComP.per.batch import PERDatasetAdapter
from ComP.train_per import _dataset_and_splits, _make_loader


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = torch.load(
        args.student_checkpoint, map_location="cpu", weights_only=False
    )
    cli_args = Namespace(**payload["args"])
    dataset, splits, _ = _dataset_and_splits(cli_args)
    split = splits[0]
    loader = _make_loader(
        PERDatasetAdapter(dataset, split["train"]), cli_args, training=False
    )
    dialogues = set()
    for batch in loader:
        dialogues.update(str(value) for value in batch.sample_ids)
    dialogues = sorted(dialogues)
    rng = random.Random(args.seed)
    order = dialogues[:]
    rng.shuffle(order)
    folds = [[] for _ in range(args.folds)]
    for index, dialogue in enumerate(order):
        folds[index % args.folds].append(dialogue)
    manifest = {
        "seed": args.seed,
        "folds": args.folds,
        "dialogues": len(dialogues),
        "fold_assignment": {
            f"fold{index}": {
                "validation": sorted(values),
                "train": sorted(
                    dialogue
                    for other, values in enumerate(folds)
                    if other != index
                    for dialogue in values
                ),
            }
            for index, values in enumerate(folds)
        },
        "gamma_set": [0.1, 0.25, 0.5],
        "selection_rules": {
            "non_inferiority": "mean card MAE <= empty MAE + tolerance",
            "oracle_stability": "positive Oracle improvement in >= 4/5 folds",
            "sign_consistency": "card improves in majority of dialogues",
            "controls": "report-only, not used for selection",
        },
        "prepare_only": args.prepare_only,
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "dialogues": len(dialogues),
                "folds": manifests(
                    folds
                ),
            },
            indent=2,
        )
    )
    return 0


def manifests(folds):
    return {f"fold{index}": len(values) for index, values in enumerate(folds)}


if __name__ == "__main__":
    raise SystemExit(main())
