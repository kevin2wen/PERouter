from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .types import (
    MODALITIES,
    EvidenceBatch,
    PERBatch,
    RawSampleMeta,
    ensure_modality_dict,
)


@dataclass
class SampleRecord:
    features: Dict[str, torch.Tensor]  # raw selected-speaker features [T,D_m]
    label: torch.Tensor  # [T]
    sample_id: str
    utterance_ids: List[str]
    sequence_mask: Optional[torch.BoolTensor] = None  # [T]
    availability: Optional[torch.BoolTensor] = None  # [T,3]
    raw_texts: Optional[List[str]] = None
    metadata: Optional[RawSampleMeta] = None


class PERDatasetAdapter(Dataset):
    """Expose ComP's existing dataset rows through the PER batch contract.

    It reads the same pre-extracted feature arrays as ComP and only exposes
    utterance IDs already present in the pickle. No feature file is rewritten.
    """

    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def _text_at(self, video_id: str, index: int) -> Optional[str]:
        sentences = getattr(self.dataset, "videoSentences", None)
        if sentences is None:
            return None
        values = sentences.get(video_id) if isinstance(sentences, dict) else None
        if values is None or index >= len(values):
            return None
        value = values[index]
        return value if isinstance(value, str) else str(value)

    def __getitem__(self, index: int) -> SampleRecord:
        source_index = self.indices[index]
        video_id = self.dataset.vids[source_index]
        host = {
            "audio": torch.as_tensor(
                self.dataset.videoAudioHost[video_id], dtype=torch.float32
            ),
            "text": torch.as_tensor(
                self.dataset.videoTextHost[video_id], dtype=torch.float32
            ),
            "video": torch.as_tensor(
                self.dataset.videoVisualHost[video_id], dtype=torch.float32
            ),
        }
        guest = {
            "audio": torch.as_tensor(
                self.dataset.videoAudioGuest[video_id], dtype=torch.float32
            ),
            "text": torch.as_tensor(
                self.dataset.videoTextGuest[video_id], dtype=torch.float32
            ),
            "video": torch.as_tensor(
                self.dataset.videoVisualGuest[video_id], dtype=torch.float32
            ),
        }
        qmask = torch.as_tensor(
            self.dataset.videoSpeakersNew[video_id], dtype=torch.float32
        ).view(-1, 1)
        features = {
            modality: torch.where(qmask == 0, host[modality], guest[modality])
            for modality in MODALITIES
        }
        label = torch.as_tensor(
            self.dataset.videoLabelsNew[video_id], dtype=torch.float32
        )
        utterance_ids = [
            str(value) for value in self.dataset.videoIDs[video_id]
        ]
        seq_len = label.numel()
        if any(value.shape[0] != seq_len for value in features.values()):
            raise ValueError(f"feature/label length mismatch for sample {video_id}")
        if len(utterance_ids) != seq_len:
            raise ValueError(f"utterance ID/label length mismatch for sample {video_id}")
        raw_texts = [self._text_at(video_id, i) for i in range(seq_len)]
        metadata = RawSampleMeta(
            sample_id=str(video_id),
            utterance_ids=utterance_ids,
            raw_texts=(
                raw_texts if raw_texts and any(value is not None for value in raw_texts)
                else None
            ),
        )
        return SampleRecord(
            features=features,
            label=label,
            sample_id=str(video_id),
            utterance_ids=utterance_ids,
            sequence_mask=torch.ones(seq_len, dtype=torch.bool),
            availability=torch.ones(seq_len, 3, dtype=torch.bool),
            raw_texts=raw_texts,
            metadata=metadata,
        )


def _target_length(batch: Sequence[SampleRecord], max_lens: Dict[str, int]) -> int:
    if "sequence" not in max_lens:
        raise KeyError("max_lens must contain 'sequence'")
    target = int(max_lens["sequence"])
    observed_max = max(record.label.numel() for record in batch)
    if target < observed_max:
        raise ValueError(f"max sequence length {target} < batch max {observed_max}")
    return target


def collate_per(
    batch: List[SampleRecord],
    max_lens: Dict[str, int],
    evidence_batch_size: int,
    training: bool,
) -> PERBatch:
    """Pad a list of SampleRecord objects without changing feature values."""
    if not batch:
        raise ValueError("cannot collate an empty PER batch")
    if evidence_batch_size <= 0:
        raise ValueError("evidence_batch_size/top_k must be positive")
    seq_len = _target_length(batch, max_lens)
    batch_size = len(batch)

    feature_names = set(MODALITIES)
    padded_features: Dict[str, torch.Tensor] = {}
    for modality in MODALITIES:
        dims = {int(record.features[modality].shape[-1]) for record in batch}
        if len(dims) != 1:
            raise ValueError(f"{modality} feature dimensions differ inside batch: {dims}")
        dim = dims.pop()
        value = torch.zeros(batch_size, seq_len, dim, dtype=torch.float32)
        for batch_index, record in enumerate(batch):
            feature = record.features[modality]
            length = feature.shape[0]
            value[batch_index, :length] = feature
        padded_features[modality] = value

    labels = torch.zeros(batch_size, seq_len, dtype=torch.float32)
    sequence_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    availability = torch.zeros(batch_size, seq_len, 3, dtype=torch.bool)
    raw_texts: Optional[List[List[str]]] = []
    metadata: List[RawSampleMeta] = []
    utterance_ids: List[List[str]] = []
    all_texts_present = True
    for batch_index, record in enumerate(batch):
        length = record.label.numel()
        labels[batch_index, :length] = record.label
        sequence_mask[batch_index, :length] = (
            record.sequence_mask
            if record.sequence_mask is not None
            else torch.ones(length, dtype=torch.bool)
        )
        row_availability = (
            record.availability
            if record.availability is not None
            else torch.ones(length, 3, dtype=torch.bool)
        )
        if row_availability.shape != (length, 3):
            raise ValueError(
                f"availability must be [T,3], got {tuple(row_availability.shape)}"
            )
        availability[batch_index, :length] = row_availability.bool()
        texts = record.raw_texts or [""] * length
        if len(texts) != length:
            raise ValueError("raw_texts length must match label length")
        all_texts_present = all_texts_present and all(value is not None for value in texts)
        raw_texts.append(
            [str(value) if value is not None else "" for value in texts]
            + [""] * (seq_len - length)
        )
        utterance_ids.append(
            list(record.utterance_ids) + [""] * (seq_len - length)
        )
        if record.metadata is not None:
            metadata.append(record.metadata)

    availability &= sequence_mask.unsqueeze(-1)
    for batch_index in range(batch_size):
        valid_rows = availability[batch_index, sequence_mask[batch_index]]
        if valid_rows.numel() and (~valid_rows.any(dim=-1)).any():
            raise ValueError("every valid utterance must have at least one modality")

    return PERBatch(
        features=padded_features,
        availability=availability,
        sequence_mask=sequence_mask,
        labels=labels,
        sample_ids=[record.sample_id for record in batch],
        utterance_ids=utterance_ids,
        evidence=None,
        raw_texts=raw_texts if all_texts_present else None,
        metadata=metadata if len(metadata) == batch_size else None,
    )


def move_batch_to_device(batch: PERBatch, device: torch.device) -> PERBatch:
    """Move tensors recursively while preserving string IDs on CPU."""
    features = {name: value.to(device) for name, value in batch.features.items()}
    availability = batch.availability.to(device=device, dtype=torch.bool)
    sequence_mask = batch.sequence_mask.to(device=device, dtype=torch.bool)
    labels = batch.labels.to(device)
    evidence = batch.evidence
    if evidence is not None:
        evidence = EvidenceBatch(
            ids=evidence.ids,
            features=evidence.features.to(device),
            mask=evidence.mask.to(device=device, dtype=torch.bool),
            source_confidence=evidence.source_confidence.to(device),
            conflict_score=(
                evidence.conflict_score.to(device)
                if evidence.conflict_score is not None
                else None
            ),
        )
    return PERBatch(
        features=features,
        availability=availability,
        sequence_mask=sequence_mask,
        labels=labels,
        sample_ids=list(batch.sample_ids),
        utterance_ids=[list(row) for row in batch.utterance_ids],
        evidence=evidence,
        raw_texts=batch.raw_texts,
        metadata=batch.metadata,
    )


def apply_availability(batch: PERBatch, availability: torch.BoolTensor) -> PERBatch:
    """Return a batch whose unavailable feature columns are exactly zero."""
    if availability.shape != (*batch.sequence_mask.shape, 3):
        raise ValueError("availability must be [B,T,3]")
    availability = availability.bool() & batch.sequence_mask.unsqueeze(-1)
    masked_features = {}
    for modality_index, modality in enumerate(MODALITIES):
        keep = availability[:, :, modality_index].unsqueeze(-1)
        masked_features[modality] = torch.where(
            keep,
            batch.features[modality],
            torch.zeros_like(batch.features[modality]),
        )
    return PERBatch(
        features=masked_features,
        availability=availability,
        sequence_mask=batch.sequence_mask,
        labels=batch.labels,
        sample_ids=batch.sample_ids,
        utterance_ids=batch.utterance_ids,
        evidence=batch.evidence,
        raw_texts=batch.raw_texts,
        metadata=batch.metadata,
    )


def full_availability(batch: PERBatch) -> torch.BoolTensor:
    return batch.sequence_mask.unsqueeze(-1).expand(-1, -1, 3).clone()


def validate_feature_batch(batch: PERBatch, dim: int) -> None:
    b, t = batch.sequence_mask.shape
    dtype = next(iter(batch.features.values())).dtype
    device = batch.sequence_mask.device
    ensure_modality_dict(
        batch.features,
        batch_size=b,
        seq_len=t,
        dim=dim,
        device=device,
        dtype=dtype,
        name="features",
    )
