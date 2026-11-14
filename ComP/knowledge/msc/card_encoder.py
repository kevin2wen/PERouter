from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


class SentenceCardEncoder:
    """Frozen sentence-embedding encoder for knowledge card text.

    The model is loaded from a local directory so experiments run offline.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        max_length: int = 256,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_path = str(model_path)
        self.model = SentenceTransformer(self.model_path, device=device)
        self.model.max_seq_length = int(max_length)
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    @staticmethod
    def card_text(card: Mapping[str, object]) -> str:
        parts = [
            str(card.get("evidence_text") or ""),
            str(card.get("trigger_text") or ""),
            str(card.get("relation_type") or ""),
        ]
        return " ".join(part for part in parts if part).strip()

    def encode(self, texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        values = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(values, dtype=np.float32)

    def fingerprint(self, texts: Sequence[str]) -> str:
        digest = hashlib.sha256()
        digest.update(self.model_path.encode())
        for text in texts:
            digest.update(str(text).encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()
