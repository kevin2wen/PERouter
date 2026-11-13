from __future__ import annotations

from typing import Iterable, Mapping


def summarize_action_rows(rows: Iterable[Mapping[str, object]]) -> dict:
    rows = list(rows)
    if not rows:
        return {
            "count": 0,
            "use_evidence_rate": 0.0,
            "skip_rate": 0.0,
            "rank_counts": {},
        }
    rank_counts = {}
    use_count = 0
    for row in rows:
        rank = int(row.get("candidate_rank", -1))
        rank_counts[str(rank)] = rank_counts.get(str(rank), 0) + 1
        use_count += int(bool(row.get("use_evidence")))
    return {
        "count": len(rows),
        "use_evidence_rate": use_count / len(rows),
        "skip_rate": 1.0 - use_count / len(rows),
        "rank_counts": rank_counts,
    }


def summarize_cards(cards: Iterable[Mapping[str, object]]) -> dict:
    cards = list(cards)
    sources = {
        card.get("source_id") for card in cards if card.get("source_id")
    }
    modalities = {}
    affects = {}
    for card in cards:
        for modality in card.get("applicability", ()):
            modalities[modality] = modalities.get(modality, 0) + 1
        affect = card.get("affect_prior")
        affects[affect] = affects.get(affect, 0) + 1
    return {
        "count": len(cards),
        "unique_source_ids": len(sources),
        "applicability_counts": modalities,
        "affect_counts": affects,
    }
