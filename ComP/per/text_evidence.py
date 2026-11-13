from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .types import MODALITIES, EvidenceBatch, PERBatch

_TEXT_INDEX = MODALITIES.index("text")


class TextNeighborRetriever:
    """Train-only raw-text TF-IDF nearest-neighbor evidence.

    Retrieval uses raw text only when the text modality is observed. Evidence
    content is the matched training utterance's three modality feature vector.
    No labels are read, and the bank is built exclusively from train batches.
    """

    source_version = "text_tfidf_ngram12_v1"

    def __init__(
        self,
        modality_dims: Dict[str, int],
        top_k: int,
        device: Optional[torch.device] = None,
        *,
        min_df: int = 2,
        max_features: int = 30000,
        exclude_same_video: bool = False,
        exclude_same_dialogue: bool = False,
        exclude_same_speaker: bool = False,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if set(modality_dims) != set(MODALITIES):
            raise KeyError("modality_dims must contain audio, text, and video")
        if exclude_same_speaker:
            raise NotImplementedError(
                "same-speaker metadata is unavailable for CMU-MOSI text evidence"
            )
        self.modality_dims = {name: int(value) for name, value in modality_dims.items()}
        self.top_k = int(top_k)
        self.device = torch.device(device or "cpu")
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=int(min_df),
            max_features=int(max_features),
            sublinear_tf=True,
            dtype=np.float32,
        )
        self._vectors: Dict[str, List[torch.Tensor]] = {
            modality: [] for modality in MODALITIES
        }
        self._sample_ids: List[str] = []
        self._utterance_ids: List[str] = []
        self._texts: List[str] = []
        self._bank_full: Optional[torch.Tensor] = None
        self._bank_matrix = None
        self._fitted = False
        self.exclude_same_video = bool(exclude_same_video)
        self.exclude_same_dialogue = bool(exclude_same_dialogue)
        self.exclude_same_speaker = bool(exclude_same_speaker)

    @property
    def evidence_dim(self) -> int:
        return sum(self.modality_dims.values())

    def __len__(self) -> int:
        return len(self._texts)

    def _stack_bank(self):
        if self._bank_full is None:
            raise RuntimeError("text bank has not been fitted")
        return {"full": self._bank_full}

    def _ensure_index(self) -> None:
        if self._bank_full is None:
            raise RuntimeError("text bank has not been fitted")

    def add(
        self,
        sample_id: str,
        utterance_ids: Sequence[str],
        features: Dict[str, torch.Tensor],
        sequence_mask: torch.BoolTensor,
        raw_texts: Sequence[Optional[str]],
    ) -> None:
        if sequence_mask.dtype != torch.bool or sequence_mask.ndim != 1:
            raise ValueError("sequence_mask must be bool [T]")
        length = sequence_mask.numel()
        if len(utterance_ids) != length or len(raw_texts) != length:
            raise ValueError("utterance_ids and raw_texts must match sequence length")
        for modality in MODALITIES:
            value = features[modality].detach().cpu().float()
            if value.shape != (length, self.modality_dims[modality]):
                raise ValueError(
                    f"bank {modality} shape={tuple(value.shape)}, expected "
                    f"{(length, self.modality_dims[modality])}"
                )
        for time_index in range(length):
            if not bool(sequence_mask[time_index]):
                continue
            text = raw_texts[time_index]
            if not isinstance(text, str) or not text.strip():
                continue
            self._texts.append(text.strip())
            self._sample_ids.append(str(sample_id))
            self._utterance_ids.append(str(utterance_ids[time_index]))
            for modality in MODALITIES:
                self._vectors[modality].append(features[modality][time_index].detach().cpu().float())
        self._fitted = False

    def add_batch(self, batch: PERBatch) -> None:
        if batch.raw_texts is None:
            raise ValueError("raw_texts are required to build text evidence")
        for batch_index, sample_id in enumerate(batch.sample_ids):
            features = {
                modality: batch.features[modality][batch_index]
                for modality in MODALITIES
            }
            self.add(
                sample_id,
                batch.utterance_ids[batch_index],
                features,
                batch.sequence_mask[batch_index],
                batch.raw_texts[batch_index],
            )

    def add_loader(self, loader: Iterable[PERBatch]) -> None:
        for batch in loader:
            self.add_batch(batch)
        self.fit()

    def fit(self) -> None:
        if not self._texts:
            raise ValueError("cannot fit text evidence on an empty train bank")
        self._bank_matrix = self.vectorizer.fit_transform(self._texts)
        ordered = [
            torch.stack(self._vectors[modality]) for modality in MODALITIES
        ]
        self._bank_full = torch.cat(ordered, dim=-1).float()
        self._fitted = True

    def source_fingerprint(self) -> str:
        payload = {
            "source_version": self.source_version,
            "top_k": self.top_k,
            "sample_ids": self._sample_ids,
            "utterance_ids": self._utterance_ids,
            "texts": self._texts,
            "vectorizer": self.vectorizer.get_params(),
            "exclude_same_video": self.exclude_same_video,
            "exclude_same_dialogue": self.exclude_same_dialogue,
            "exclude_same_speaker": self.exclude_same_speaker,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def retrieve(
        self,
        features: Dict[str, torch.Tensor],
        availability: torch.BoolTensor,
        sequence_mask: torch.BoolTensor,
        sample_ids: Sequence[str],
        utterance_ids: Sequence[Sequence[str]],
        *,
        raw_texts: Optional[Sequence[Sequence[Optional[str]]]],
    ) -> EvidenceBatch:
        if not self._fitted or self._bank_full is None or self._bank_matrix is None:
            raise RuntimeError("text retriever must be fitted before retrieval")
        batch_size, seq_len = sequence_mask.shape
        if availability.shape != (batch_size, seq_len, 3):
            raise ValueError("availability must be [B,T,3]")
        if len(sample_ids) != batch_size or len(utterance_ids) != batch_size:
            raise ValueError("sample_ids and utterance_ids must match batch size")
        if raw_texts is None or len(raw_texts) != batch_size:
            raise ValueError("raw_texts must be [B] and available for text evidence")
        device = next(iter(features.values())).device
        n_bank = len(self._texts)
        k = max(1, min(self.top_k, n_bank))
        evidence_features = torch.zeros(
            batch_size, seq_len, k, self.evidence_dim, dtype=torch.float32, device=device
        )
        evidence_mask = torch.zeros(batch_size, seq_len, k, dtype=torch.bool, device=device)
        confidence = torch.zeros(batch_size, seq_len, k, dtype=torch.float32, device=device)
        ids: List[List[List[str]]] = [
            [["" for _ in range(k)] for _ in range(seq_len)]
            for _ in range(batch_size)
        ]

        query_positions = []
        query_texts = []
        for batch_index in range(batch_size):
            if len(raw_texts[batch_index]) < seq_len:
                raise ValueError("raw_texts row is shorter than sequence length")
            for time_index in range(seq_len):
                if not bool(sequence_mask[batch_index, time_index]):
                    continue
                row_availability = availability[batch_index, time_index]
                if bool(row_availability.all()):
                    continue
                # A missing text modality must not be reconstructed from raw text.
                if not bool(row_availability[_TEXT_INDEX]):
                    continue
                text = raw_texts[batch_index][time_index]
                if not isinstance(text, str) or not text.strip():
                    continue
                query_positions.append((batch_index, time_index))
                query_texts.append(text.strip())
        if not query_positions:
            return EvidenceBatch(
                ids=ids,
                features=evidence_features,
                mask=evidence_mask,
                source_confidence=confidence,
                conflict_score=torch.zeros_like(confidence),
            )

        query_matrix = self.vectorizer.transform(query_texts)
        similarities = linear_kernel(query_matrix, self._bank_matrix)
        bank_full = self._bank_full
        for query_index, (batch_index, time_index) in enumerate(query_positions):
            row = torch.from_numpy(similarities[query_index].copy())
            same_item = torch.tensor(
                [
                    sample == str(sample_ids[batch_index])
                    and utterance == str(utterance_ids[batch_index][time_index])
                    for sample, utterance in zip(self._sample_ids, self._utterance_ids)
                ],
                dtype=torch.bool,
            )
            row = row.masked_fill(same_item, float("-inf"))
            blocked = same_item
            if self.exclude_same_video or self.exclude_same_dialogue:
                blocked = blocked | torch.tensor(
                    [
                        sample == str(sample_ids[batch_index])
                        for sample in self._sample_ids
                    ],
                    dtype=torch.bool,
                )
            valid_count = min(k, int((~blocked).sum().item()))
            if valid_count <= 0:
                continue
            values, indices = torch.topk(
                row.masked_fill(blocked, float("-inf")), k=valid_count
            )
            evidence_features[batch_index, time_index, :valid_count] = bank_full[indices].to(device)
            evidence_mask[batch_index, time_index, :valid_count] = True
            confidence[batch_index, time_index, :valid_count] = values.clamp(min=0.0, max=1.0).to(device)
            for rank, bank_index in enumerate(indices.tolist()):
                ids[batch_index][time_index][rank] = (
                    f"{self._sample_ids[bank_index]}:{self._utterance_ids[bank_index]}"
                )
        return EvidenceBatch(
            ids=ids,
            features=evidence_features,
            mask=evidence_mask,
            source_confidence=confidence,
            conflict_score=torch.zeros_like(confidence),
        )
