from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.knowledge.msc.card_diversity import (
    near_duplicate_report,
    relation_separation,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-cache", required=True)
    parser.add_argument("--cards", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cache = torch.load(args.semantic_cache, map_location="cpu", weights_only=False)
    cards = {}
    for line in Path(args.cards).open(encoding="utf-8"):
        card = json.loads(line)
        cards[str(card["knowledge_id"])] = card
    vectors = {}
    for row, ids in enumerate(cache["candidate_ids"]):
        for rank, card_id in enumerate(ids):
            card_id = str(card_id)
            if card_id and card_id not in vectors:
                vectors[card_id] = cache["candidate_vectors"][row, rank].numpy()
    ids = sorted(vectors)
    embeddings = np.stack([vectors[card_id] for card_id in ids])
    relations = [
        str(cards.get(card_id, {}).get("relation_type") or "unknown")
        for card_id in ids
    ]
    report = {
        "cards": len(ids),
        **near_duplicate_report(embeddings),
        **relation_separation(embeddings, relations),
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {k: v for k, v in report.items() if not isinstance(v, dict)}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
