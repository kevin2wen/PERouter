from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np


def normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def card_text_quality(
    cards: Sequence[Mapping[str, object]],
    embeddings: np.ndarray | None = None,
) -> dict:
    texts = [
        normalise_text(
            " ".join(
                str(card.get(key) or "")
                for key in ("evidence_text", "trigger_text", "relation_type")
            )
        )
        for card in cards
    ]
    counts = Counter(texts)
    unique = len(counts)
    total = len(texts)
    lengths = [len(text.split()) for text in texts if text]
    report = {
        "cards": total,
        "unique_texts": unique,
        "duplicate_text_rate": 1.0 - unique / max(1, total),
        "most_common_text_count": counts.most_common(1)[0][1] if counts else 0,
        "mean_words": float(np.mean(lengths)) if lengths else 0.0,
        "p50_words": float(np.median(lengths)) if lengths else 0.0,
        "max_words": int(max(lengths)) if lengths else 0,
        "relation_distribution": dict(
            Counter(str(card.get("relation_type") or "") for card in cards)
        ),
        "affect_distribution": dict(
            Counter(str(card.get("affect_prior") or "") for card in cards)
        ),
        "source_type_distribution": dict(
            Counter(str(card.get("source_type") or "") for card in cards)
        ),
    }
    if embeddings is not None and len(embeddings) > 1:
        values = np.asarray(embeddings, dtype=np.float64)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        normed = values / np.clip(norms, 1e-9, None)
        similarity = normed @ normed.T
        upper = similarity[np.triu_indices(values.shape[0], k=1)]
        report["embedding_cosine_mean"] = float(upper.mean())
        report["embedding_cosine_p10"] = float(np.quantile(upper, 0.1))
        report["embedding_cosine_p90"] = float(np.quantile(upper, 0.9))
        report["embedding_dim"] = int(values.shape[1])
    return report
