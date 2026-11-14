from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_cards(path: str) -> dict:
    cards = {}
    for line in Path(path).open(encoding="utf-8"):
        card = json.loads(line)
        cards[str(card["knowledge_id"])] = card
    return cards


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-cache", required=True)
    parser.add_argument("--train-cards", required=True)
    parser.add_argument("--val-cards", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cache = torch.load(args.semantic_cache, map_location="cpu", weights_only=False)
    ids = {
        str(card_id)
        for row in cache["candidate_ids"]
        for card_id in row
        if card_id
    }
    train_cards = load_cards(args.train_cards)
    val_cards = load_cards(args.val_cards)
    train_texts = {
        str(card.get("evidence_text") or "").strip() for card in train_cards.values()
    }
    val_texts = {
        str(card.get("evidence_text") or "").strip() for card in val_cards.values()
    }
    report = {
        "semantic_cache_queries": len(cache["candidate_ids"]),
        "semantic_cache_unique_cards": len(ids),
        "semantic_cache_missing_from_train_cards": len(ids - set(train_cards)),
        "train_cards": len(train_cards),
        "val_cards": len(val_cards),
        "val_cards_overlap_train_bank": len(set(val_cards) & set(train_cards)),
        "val_text_overlap_train_texts": len(val_texts & train_texts),
        "val_text_total": len(val_texts),
        "train_text_total": len(train_texts),
        "split": cache.get("split"),
        "semantic_model_path": cache.get("semantic_model_path"),
        "semantic_feature_dim": cache.get("semantic_feature_dim"),
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
