from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.knowledge.kb_v4.schema import KnowledgeCardV4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cards = [
        json.loads(line)
        for line in Path(args.cards).open(encoding="utf-8")
    ]
    errors = []
    for raw in cards:
        card = KnowledgeCardV4(
            card_id=raw["card_id"],
            claim=raw["claim"],
            source=raw["source"],
            source_type=raw["source_type"],
            source_version=raw["source_version"],
            source_confidence=float(raw["source_confidence"]),
            observed_modalities=tuple(raw["observed_modalities"]),
            handled_missing_modalities=tuple(raw["handled_missing_modalities"]),
            trigger_predicates=tuple(raw["trigger_predicates"]),
            negative_conditions=tuple(raw["negative_conditions"]),
            evidence_relation=raw["evidence_relation"],
            expected_effect_direction=raw["expected_effect_direction"],
            supporting_context=tuple(raw.get("supporting_context", ())),
            conflict_with=tuple(raw.get("conflict_with", ())),
            utility_prior=float(raw.get("utility_prior", 0.0)),
            provenance=dict(raw.get("provenance", {})),
            license=raw.get("license", "research-only"),
        )
        errors.extend(card.validate())
    claims = Counter(card["claim"] for card in cards)
    report = {
        "cards": len(cards),
        "schema_errors": len(errors),
        "error_examples": errors[:5],
        "duplicate_claim_rate": 1 - len(claims) / max(1, len(cards)),
        "label_leak_cards": sum(
            1 for card in cards
            if card.get("provenance", {}).get("label_used") is True
        ),
        "target_raw_leak_cards": sum(
            1 for card in cards
            if card.get("provenance", {}).get(
                "target_raw_used_for_current_query"
            ) is True
        ),
        "provenance_missing": sum(
            1 for card in cards
            if not card.get("source") or not card.get("source_type")
        ),
        "by_missing_modality": dict(
            Counter(card["handled_missing_modalities"][0] for card in cards)
        ),
        "source_versions": dict(
            Counter(card["source_version"] for card in cards)
        ),
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors and report["label_leak_cards"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
