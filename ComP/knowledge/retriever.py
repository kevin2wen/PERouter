from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from ComP.per.types import MODALITIES, PERBatch

from .schema import EvidenceQuery, RetrievedContext


class KnowledgeContextRetriever:
    """Deterministic train-only context bank for rule-based knowledge cards."""

    def __init__(
        self,
        modality_dims: Dict[str, int],
        top_k: int,
        *,
        min_df: int = 1,
        max_features: int = 30000,
        exclude_same_video: bool = True,
        exclude_same_dialogue: bool = True,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if set(modality_dims) != set(MODALITIES):
            raise KeyError("modality_dims must contain audio, text, and video")
        self.modality_dims = {
            name: int(modality_dims[name]) for name in MODALITIES
        }
        self.top_k = int(top_k)
        self.min_df = int(min_df)
        self.max_features = int(max_features)
        self.exclude_same_video = bool(exclude_same_video)
        self.exclude_same_dialogue = bool(exclude_same_dialogue)
        self._vectors = {name: [] for name in MODALITIES}
        self._texts: List[str] = []
        self._sample_ids: List[str] = []
        self._utterance_ids: List[str] = []
        self._stacked: Optional[Dict[str, torch.Tensor]] = None
        self._bank_norm: Optional[Dict[str, torch.Tensor]] = None
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._text_matrix = None

    @property
    def evidence_dim(self) -> int:
        return sum(self.modality_dims.values())

    def __len__(self) -> int:
        return len(self._sample_ids)

    def add_batch(self, batch: PERBatch, *, split: str) -> None:
        if split != "train":
            raise PermissionError("knowledge context bank is train-only")
        batch_size, seq_len = batch.sequence_mask.shape
        for batch_index in range(batch_size):
            sample_id = str(batch.sample_ids[batch_index])
            for time_index in range(seq_len):
                if not bool(batch.sequence_mask[batch_index, time_index]):
                    continue
                self._sample_ids.append(sample_id)
                self._utterance_ids.append(
                    str(batch.utterance_ids[batch_index][time_index])
                )
                self._texts.append(
                    self._text_at(batch, batch_index, time_index)
                )
                for modality in MODALITIES:
                    value = batch.features[modality][batch_index, time_index]
                    if value.numel() != self.modality_dims[modality]:
                        raise ValueError(f"invalid {modality} feature dimension")
                    self._vectors[modality].append(
                        value.detach().cpu().float()
                    )
        self._stacked = None
        self._bank_norm = None
        self._vectorizer = None
        self._text_matrix = None

    @staticmethod
    def _text_at(batch: PERBatch, batch_index: int, time_index: int) -> str:
        if batch.raw_texts is None:
            return ""
        value = batch.raw_texts[batch_index][time_index]
        return value.strip() if isinstance(value, str) else ""

    def fit(self) -> None:
        if not self._sample_ids:
            raise ValueError("cannot fit an empty knowledge context bank")
        self._stacked = {}
        self._bank_norm = {}
        for modality in MODALITIES:
            bank = torch.stack(self._vectors[modality])
            if bank.size(-1) != self.modality_dims[modality]:
                raise ValueError(f"invalid bank dimension for {modality}")
            self._stacked[modality] = bank
            self._bank_norm[modality] = F.normalize(bank, p=2, dim=-1)
        nonempty = [bool(text) for text in self._texts]
        if any(nonempty):
            self._vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=self.min_df,
                max_features=self.max_features,
                sublinear_tf=True,
                dtype=np.float32,
            )
            self._text_matrix = self._vectorizer.fit_transform(self._texts)

    def _ensure_fit(self) -> None:
        if self._stacked is None or self._bank_norm is None:
            self.fit()

    def source_fingerprint(self) -> str:
        payload = {
            "sample_ids": self._sample_ids,
            "utterance_ids": self._utterance_ids,
            "texts": self._texts,
            "modality_dims": self.modality_dims,
            "top_k": self.top_k,
            "exclude_same_video": self.exclude_same_video,
            "exclude_same_dialogue": self.exclude_same_dialogue,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "source_split": "train",
                "modality_dims": self.modality_dims,
                "top_k": self.top_k,
                "min_df": self.min_df,
                "max_features": self.max_features,
                "exclude_same_video": self.exclude_same_video,
                "exclude_same_dialogue": self.exclude_same_dialogue,
                "vectors": self._vectors,
                "texts": self._texts,
                "sample_ids": self._sample_ids,
                "utterance_ids": self._utterance_ids,
                "source_fingerprint": self.source_fingerprint(),
            },
            output,
        )

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeContextRetriever":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("source_split") != "train":
            raise PermissionError("knowledge index must use the train split")
        retriever = cls(
            payload["modality_dims"],
            top_k=payload["top_k"],
            min_df=payload["min_df"],
            max_features=payload["max_features"],
            exclude_same_video=payload.get("exclude_same_video", True),
            exclude_same_dialogue=payload.get("exclude_same_dialogue", True),
        )
        retriever._vectors = {
            name: [value.cpu() for value in payload["vectors"][name]]
            for name in MODALITIES
        }
        retriever._texts = list(payload["texts"])
        retriever._sample_ids = list(payload["sample_ids"])
        retriever._utterance_ids = list(payload["utterance_ids"])
        expected = payload.get("source_fingerprint")
        retriever.fit()
        if expected and expected != retriever.source_fingerprint():
            raise ValueError("knowledge index fingerprint mismatch")
        return retriever

    def retrieve(
        self,
        query: EvidenceQuery,
        top_k: int,
    ) -> List[RetrievedContext]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self._ensure_fit()
        assert self._stacked is not None and self._bank_norm is not None
        similarities = []
        if query.audio_query is not None:
            similarities.append(self._feature_similarity("audio", query.audio_query))
        if query.video_query is not None:
            similarities.append(self._feature_similarity("video", query.video_query))
        if (
            query.text_query is not None
            and self._vectorizer is not None
            and self._text_matrix is not None
        ):
            query_text = self._vectorizer.transform([query.text_query])
            text_scores = linear_kernel(query_text, self._text_matrix)[0]
            similarities.append(torch.from_numpy(text_scores.copy()).float())
        if not similarities:
            return []
        score = torch.stack(similarities, dim=0).mean(dim=0)
        same_video = torch.tensor(
            [
                sample == query.sample_id
                for sample in self._sample_ids
            ],
            dtype=torch.bool,
        )
        same_utterance = torch.tensor(
            [
                sample == query.sample_id and utterance == query.utterance_id
                for sample, utterance in zip(
                    self._sample_ids, self._utterance_ids
                )
            ],
            dtype=torch.bool,
        )
        blocked = same_utterance
        if self.exclude_same_video:
            blocked |= same_video
        same_dialogue = torch.tensor(
            [
                sample == query.dialogue_id
                for sample in self._sample_ids
            ],
            dtype=torch.bool,
        )
        if self.exclude_same_dialogue:
            blocked |= same_dialogue
        score = score.masked_fill(blocked, float("-inf"))
        valid_count = min(top_k, int((~blocked).sum().item()))
        if valid_count <= 0:
            return []
        values, indices = torch.topk(score, k=valid_count)
        contexts = []
        observed = tuple(
            name for name, value in zip(MODALITIES, query.availability) if value
        )
        for rank, (score_value, bank_index) in enumerate(
            zip(values.tolist(), indices.tolist())
        ):
            context_text = self._texts[bank_index] or None
            confidence = max(0.0, min(1.0, float(score_value)))
            contexts.append(
                RetrievedContext(
                    query_id=hashlib.sha256(
                        (
                            f"{query.sample_id}|{query.dialogue_id}|"
                            f"{query.utterance_id}|{query.target_modality}"
                        ).encode("utf-8")
                    ).hexdigest(),
                    sample_id=self._sample_ids[bank_index],
                    dialogue_id=self._sample_ids[bank_index],
                    utterance_id=self._utterance_ids[bank_index],
                    target_modality=query.target_modality,
                    rank=rank,
                    text=context_text,
                    audio_feature=self._stacked["audio"][bank_index].clone(),
                    video_feature=self._stacked["video"][bank_index].clone(),
                    retrieval_score=float(score_value),
                    source_modalities=observed,
                    source_id=(
                        f"{self._sample_ids[bank_index]}:"
                        f"{self._utterance_ids[bank_index]}"
                    ),
                    source_confidence=confidence,
                )
            )
        return contexts

    def _feature_similarity(
        self,
        modality: str,
        query: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim != 1 or query.numel() != self.modality_dims[modality]:
            raise ValueError(f"{modality}_query has invalid dimension")
        if not torch.isfinite(query).all():
            raise FloatingPointError(f"{modality}_query contains non-finite values")
        normalized = F.normalize(query.detach().cpu().float(), p=2, dim=0)
        assert self._bank_norm is not None
        return torch.mv(self._bank_norm[modality], normalized)
