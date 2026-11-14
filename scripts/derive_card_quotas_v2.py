from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--cards", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cache = torch.load(args.candidate_cache, map_location="cpu", weights_only=False)
    labels = cache["target_labels"].float().sign().long()
    targets = cache["target_modalities"].long()
    patterns = cache["missing_patterns"]
    distribution = Counter()
    for label, target, pattern in zip(labels.tolist(), targets.tolist(), patterns):
        distribution[f"{target}|{pattern}|{label}"] += 1
    cards = [json.loads(line) for line in Path(args.cards).open(encoding="utf-8")]
    affect = Counter(card.get("affect_prior") for card in cards)
    relation = Counter(card.get("relation_type") for card in cards)
    quotas = {}
    for key, count in distribution.items():
        target, pattern, label = key.split("|")
        quotas[key] = {
            "queries": count,
            "expected_negative_cards": round(count * (affect["negative"] / max(1, len(cards))), 1),
            "expected_positive_cards": round(count * (affect["positive"] / max(1, len(cards))), 1),
            "expected_mixed_cards": round(count * (affect["mixed"] / max(1, len(cards))), 1),
        }
    report = {
        "source": "train candidate cache only",
        "queries": int(labels.numel()),
        "label_distribution": dict(Counter(labels.tolist())),
        "affect_distribution": dict(affect),
        "relation_distribution": dict(relation),
        "quotas": quotas,
        "note": "Descriptive quota table; cards are not sampled by label.",
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"queries": report["queries"], "affect": report["affect_distribution"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
