from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import (
    MODALITIES,
    EvidenceBatch,
    EvidenceQuery,
    RawSampleMeta,
)


def build_evidence_query(
    features: Dict[str, torch.Tensor],
    availability: torch.BoolTensor,
    raw_metadata: List[RawSampleMeta],
    target_modalities: torch.BoolTensor,
) -> EvidenceQuery:
    """Build one query per sample/utterance without reading labels."""
    if availability.shape != target_modalities.shape:
        raise ValueError("availability and target_modalities must both be [B,T,3]")
    batch_size, seq_len = target_modalities.shape[:2]
    if len(raw_metadata) != batch_size:
        raise ValueError("raw_metadata length must equal batch size")
    columns = []
    for modality_index, modality in enumerate(MODALITIES):
        value = features[modality]
        if value.shape[:2] != (batch_size, seq_len):
            raise ValueError(f"features[{modality}] must share [B,T]")
        keep = availability[:, :, modality_index].unsqueeze(-1)
        columns.append(torch.where(keep, value, torch.zeros_like(value)))
    return EvidenceQuery(
        sample_ids=[item.sample_id for item in raw_metadata],
        utterance_ids=[list(item.utterance_ids) for item in raw_metadata],
        features=torch.cat(columns, dim=-1),
        target_modalities=target_modalities,
    )


class EvidenceEncoder(nn.Module):
    """Pool top-k evidence and project it to ComP's [B,T,D_e] space."""

    def __init__(
        self,
        evidence_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.evidence_dim = int(evidence_dim)
        self.hidden_dim = int(hidden_dim)
        self.projections = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear(self.evidence_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                for modality in MODALITIES
            }
        )

    def forward(
        self,
        evidence: EvidenceBatch,
        availability: torch.BoolTensor,
        target_modalities: torch.BoolTensor,
        seq_lens: Optional[Dict[str, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        evidence.validate(
            batch_size=availability.shape[0], seq_len=availability.shape[1]
        )
        if target_modalities.shape != availability.shape:
            raise ValueError("target_modalities must match availability")
        if evidence.features.shape[-1] != self.evidence_dim:
            raise ValueError(
                f"evidence dim={evidence.features.shape[-1]}, "
                f"expected {self.evidence_dim}"
            )
        mask = evidence.mask
        count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
        finite_evidence = torch.where(
            mask.unsqueeze(-1),
            evidence.features,
            torch.zeros_like(evidence.features),
        )
        if not torch.isfinite(finite_evidence).all():
            raise FloatingPointError("valid evidence features are non-finite")
        pooled = finite_evidence.sum(dim=2) / count.to(evidence.features.dtype)
        pooled = pooled.to(next(self.parameters()).dtype)
        any_evidence = mask.any(dim=-1).unsqueeze(-1)
        outputs = {}
        for modality_index, modality in enumerate(MODALITIES):
            projected = self.projections[modality](pooled)
            projected = projected * any_evidence.to(projected.dtype)
            projected = projected * target_modalities[
                :, :, modality_index
            ].unsqueeze(-1).to(projected.dtype)
            outputs[modality] = projected
        return outputs


class FeatureNeighborRetriever:
    """Deterministic feature-neighbor evidence from a training-only bank.

    This is a degraded evidence path explicitly permitted when no licensed
    semantic KB is available. It never reads labels and never retrieves from
    validation or test data.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        top_k: int,
        device: Optional[torch.device] = None,
        *,
        exclude_same_utterance: bool = True,
        exclude_same_video: bool = False,
        exclude_same_dialogue: bool = False,
        exclude_same_speaker: bool = False,
    ):
        if exclude_same_speaker:
            raise NotImplementedError("same-speaker metadata is unavailable for CMU-MOSI")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if set(modality_dims) != set(MODALITIES):
            raise KeyError("modality_dims must contain audio, text, and video")
        self.modality_dims = {name: int(value) for name, value in modality_dims.items()}
        self.top_k = int(top_k)
        self.device = torch.device(device or "cpu")
        self.exclude_same_utterance = bool(exclude_same_utterance)
        self.exclude_same_video = bool(exclude_same_video)
        self.exclude_same_dialogue = bool(exclude_same_dialogue)
        self.exclude_same_speaker = bool(exclude_same_speaker)
        self._vectors: Dict[str, List[torch.Tensor]] = {
            modality: [] for modality in MODALITIES
        }
        self._sample_ids: List[str] = []
        self._utterance_ids: List[str] = []
        self._stacked: Optional[Dict[str, torch.Tensor]] = None
        self._bank_norm: Optional[Dict[str, torch.Tensor]] = None
        self._bank_full: Optional[torch.Tensor] = None

    @property
    def evidence_dim(self) -> int:
        return sum(self.modality_dims.values())

    def __len__(self) -> int:
        return len(self._sample_ids)

    def add(
        self,
        sample_id: str,
        utterance_ids: Sequence[str],
        features: Dict[str, torch.Tensor],
        sequence_mask: torch.BoolTensor,
    ) -> None:
        if sequence_mask.dtype != torch.bool or sequence_mask.ndim != 1:
            raise ValueError("sequence_mask must be bool [T] for one sample")
        length = sequence_mask.numel()
        if len(utterance_ids) != length:
            raise ValueError("utterance_ids length must match sequence_mask")
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
                self._vectors[modality].append(value[time_index])
                if modality == MODALITIES[0]:
                    self._sample_ids.append(str(sample_id))
                    self._utterance_ids.append(str(utterance_ids[time_index]))
        self._stacked = None
        self._bank_norm = None
        self._bank_full = None

    def _stack_bank(self) -> Dict[str, torch.Tensor]:
        if self._stacked is not None:
            return self._stacked
        if not self._sample_ids:
            self._stacked = {
                modality: torch.empty(
                    0, self.modality_dims[modality], device=self.device
                )
                for modality in MODALITIES
            }
            return self._stacked
        self._stacked = {
            modality: torch.stack(self._vectors[modality]).to(self.device)
            for modality in MODALITIES
        }
        return self._stacked

    def _ensure_index(self) -> None:
        bank = self._stack_bank()
        if self._bank_norm is None:
            self._bank_norm = {
                modality: F.normalize(bank[modality], p=2, dim=-1)
                for modality in MODALITIES
            }
            self._bank_full = torch.cat(
                [bank[modality] for modality in MODALITIES], dim=-1
            )

    def retrieve(
        self,
        features: Dict[str, torch.Tensor],
        availability: torch.BoolTensor,
        sequence_mask: torch.BoolTensor,
        sample_ids: Sequence[str],
        utterance_ids: Sequence[Sequence[str]],
    ) -> EvidenceBatch:
        batch_size, seq_len = sequence_mask.shape
        if availability.shape != (batch_size, seq_len, 3):
            raise ValueError("availability must be [B,T,3]")
        if len(sample_ids) != batch_size or len(utterance_ids) != batch_size:
            raise ValueError("sample_ids and utterance_ids must match batch size")
        device = next(iter(features.values())).device
        bank = self._stack_bank()
        n_bank = len(self._sample_ids)
        k = max(1, min(self.top_k, n_bank))
        evidence_dim = self.evidence_dim
        evidence_features = torch.zeros(
            batch_size, seq_len, k, evidence_dim, dtype=torch.float32, device=device
        )
        evidence_mask = torch.zeros(
            batch_size, seq_len, k, dtype=torch.bool, device=device
        )
        confidence = torch.zeros(
            batch_size, seq_len, k, dtype=torch.float32, device=device
        )
        ids: List[List[List[str]]] = [
            [["" for _ in range(k)] for _ in range(seq_len)]
            for _ in range(batch_size)
        ]
        if n_bank == 0:
            return EvidenceBatch(
                ids=ids,
                features=evidence_features,
                mask=evidence_mask,
                source_confidence=confidence,
                conflict_score=torch.zeros_like(confidence),
            )

        query_parts = [
            features[modality].to(self.device).float() for modality in MODALITIES
        ]
        self._ensure_index()
        bank_norm = self._bank_norm
        bank_full = self._bank_full
        assert bank_norm is not None and bank_full is not None

        for batch_index in range(batch_size):
            for time_index in range(seq_len):
                if not bool(sequence_mask[batch_index, time_index]):
                    continue
                row_availability = availability[batch_index, time_index]
                if bool(row_availability.all()):
                    # There is no missing source at a complete utterance.
                    continue
                available_modalities = [
                    index
                    for index in range(3)
                    if bool(row_availability[index])
                ]
                if not available_modalities:
                    continue
                similarities = []
                for modality_index in available_modalities:
                    modality = MODALITIES[modality_index]
                    query = query_parts[modality_index][batch_index, time_index]
                    query = F.normalize(query, p=2, dim=-1)
                    similarities.append(torch.mv(bank_norm[modality], query))
                similarity = torch.stack(similarities).mean(dim=0)
                same_sample = torch.tensor(
                    [sample == str(sample_ids[batch_index]) for sample in self._sample_ids],
                    dtype=torch.bool,
                    device=self.device,
                )
                same_utterance = torch.tensor(
                    [
                        sample == str(sample_ids[batch_index])
                        and utterance == str(utterance_ids[batch_index][time_index])
                        for sample, utterance in zip(self._sample_ids, self._utterance_ids)
                    ],
                    dtype=torch.bool,
                    device=self.device,
                )
                blocked = torch.zeros_like(same_sample)
                if self.exclude_same_utterance:
                    blocked |= same_utterance
                if self.exclude_same_video or self.exclude_same_dialogue:
                    blocked |= same_sample
                similarity = similarity.masked_fill(blocked, float("-inf"))
                valid_count = min(k, int((~blocked).sum().item()))
                if valid_count == 0:
                    continue
                values, indices = torch.topk(similarity, k=valid_count)
                evidence_features[
                    batch_index, time_index, :valid_count
                ] = bank_full[indices].to(device)
                evidence_mask[batch_index, time_index, :valid_count] = True
                confidence[batch_index, time_index, :valid_count] = values.clamp(
                    min=0.0, max=1.0
                ).to(device)
                for rank, bank_index in enumerate(indices.tolist()):
                    ids[batch_index][time_index][rank] = (
                        f"{self._sample_ids[bank_index]}:"
                        f"{self._utterance_ids[bank_index]}"
                    )
        return EvidenceBatch(
            ids=ids,
            features=evidence_features,
            mask=evidence_mask,
            source_confidence=confidence,
            conflict_score=torch.zeros_like(confidence),
        )


def empty_evidence(
    batch_size: int,
    seq_len: int,
    top_k: int,
    evidence_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> EvidenceBatch:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if evidence_dim <= 0:
        evidence_dim = 1
    ids = [
        [["" for _ in range(top_k)] for _ in range(seq_len)]
        for _ in range(batch_size)
    ]
    return EvidenceBatch(
        ids=ids,
        features=torch.zeros(
            batch_size, seq_len, top_k, evidence_dim, dtype=dtype, device=device
        ),
        mask=torch.zeros(batch_size, seq_len, top_k, dtype=torch.bool, device=device),
        source_confidence=torch.zeros(
            batch_size, seq_len, top_k, dtype=dtype, device=device
        ),
        conflict_score=torch.zeros(
            batch_size, seq_len, top_k, dtype=dtype, device=device
        ),
    )


def perturb_evidence(
    evidence: EvidenceBatch,
    *,
    dropout_prob: float = 0.0,
    shuffle_prob: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> EvidenceBatch:
    """Apply configurable evidence dropout / sample shuffling for training."""
    if not 0.0 <= dropout_prob <= 1.0 or not 0.0 <= shuffle_prob <= 1.0:
        raise ValueError("evidence perturbation probabilities must be in [0,1]")
    mask = evidence.mask
    features = evidence.features
    confidence = evidence.source_confidence
    conflict = evidence.conflict_score
    ids = evidence.ids
    if dropout_prob > 0:
        keep = torch.rand(
            mask.shape, generator=generator, device=mask.device
        ) >= dropout_prob
        mask = mask & keep
        confidence = confidence * mask.to(confidence.dtype)
        if conflict is not None:
            conflict = conflict * mask.to(conflict.dtype)
    if shuffle_prob > 0:
        probabilities = torch.rand(
            (mask.shape[0],), generator=generator, device=mask.device
        )
        swap = probabilities < shuffle_prob
        if int(swap.sum().item()) > 1:
            source = torch.nonzero(swap).flatten().roll(1)
            destination = torch.nonzero(swap).flatten()
            features = features.clone()
            mask = mask.clone()
            confidence = confidence.clone()
            ids = [list(row) for row in evidence.ids]
            if conflict is not None:
                conflict = conflict.clone()
            features[destination] = evidence.features[source]
            mask[destination] = evidence.mask[source]
            confidence[destination] = evidence.source_confidence[source]
            for destination_index, source_index in zip(
                destination.tolist(), source.tolist()
            ):
                ids[destination_index] = [
                    list(row) for row in evidence.ids[source_index]
                ]
            if conflict is not None:
                conflict[destination] = evidence.conflict_score[source]
    return EvidenceBatch(
        ids=ids,
        features=features,
        mask=mask,
        source_confidence=confidence,
        conflict_score=conflict,
    )


def source_only_evidence(evidence: EvidenceBatch) -> EvidenceBatch:
    """Zero evidence content while preserving availability and confidence."""
    return EvidenceBatch(
        ids=[list(row) for row in evidence.ids],
        features=torch.zeros_like(evidence.features),
        mask=evidence.mask.clone(),
        source_confidence=evidence.source_confidence.clone(),
        conflict_score=(
            evidence.conflict_score.clone()
            if evidence.conflict_score is not None
            else None
        ),
    )


def shuffled_evidence(
    evidence: EvidenceBatch,
    seed: int,
    *,
    query_sample_ids: Optional[Sequence[str]] = None,
) -> EvidenceBatch:
    """Deterministically permute whole evidence rows across the batch."""
    batch_size = evidence.features.size(0)
    if batch_size <= 1:
        result = EvidenceBatch(
            ids=[list(row) for row in evidence.ids],
            features=evidence.features.clone(),
            mask=evidence.mask.clone(),
            source_confidence=evidence.source_confidence.clone(),
            conflict_score=(
                evidence.conflict_score.clone()
                if evidence.conflict_score is not None
                else None
            ),
        )
    elif query_sample_ids is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        permutation = torch.randperm(batch_size, generator=generator)
    else:
        if len(query_sample_ids) != batch_size:
            raise ValueError("query_sample_ids must match batch size")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        priority = torch.rand(batch_size, generator=generator)
        destination_order = torch.argsort(priority, descending=True).tolist()
        available = set(range(batch_size))
        selected = [None] * batch_size
        for destination in destination_order:
            choices = [
                source
                for source in available
                if str(query_sample_ids[source])
                != str(query_sample_ids[destination])
            ]
            if not choices:
                choices = list(available)
            source = max(choices, key=lambda value: (float(priority[value]), -value))
            selected[destination] = source
            available.remove(source)
        permutation = torch.tensor(selected, dtype=torch.long)
    if batch_size > 1:
        device = evidence.features.device
        permutation = permutation.to(device)
        result = EvidenceBatch(
            ids=[evidence.ids[index] for index in permutation.cpu().tolist()],
            features=evidence.features[permutation],
            mask=evidence.mask[permutation],
            source_confidence=evidence.source_confidence[permutation],
            conflict_score=(
                evidence.conflict_score[permutation]
                if evidence.conflict_score is not None
                else None
            ),
        )
    if query_sample_ids is not None:
        mask = result.mask.clone()
        ids = [[list(rank_ids) for rank_ids in row] for row in result.ids]
        for batch_index, sample_id in enumerate(query_sample_ids):
            for time_index in range(mask.size(1)):
                for rank in range(mask.size(2)):
                    neighbor = ids[batch_index][time_index][rank]
                    owner = neighbor.partition(":")[0] if neighbor else ""
                    if owner == str(sample_id):
                        mask[batch_index, time_index, rank] = False
                        ids[batch_index][time_index][rank] = ""
        result = EvidenceBatch(
            ids=ids,
            features=result.features,
            mask=mask,
            source_confidence=result.source_confidence
            * mask.to(result.source_confidence.dtype),
            conflict_score=(
                result.conflict_score * mask.to(result.conflict_score.dtype)
                if result.conflict_score is not None
                else None
            ),
        )
    return result


def randomize_evidence(
    reference: EvidenceBatch,
    retriever: FeatureNeighborRetriever,
    seed: int,
    *,
    query_sample_ids: Optional[Sequence[str]] = None,
) -> EvidenceBatch:
    """Replace valid evidence with deterministic random training-bank rows."""
    retriever._stack_bank()
    retriever._ensure_index()
    n_bank = len(retriever._sample_ids)
    features = torch.zeros_like(reference.features)
    mask = reference.mask.clone()
    if n_bank == 0:
        mask.zero_()
        return EvidenceBatch(
            ids=[[["" for _ in row] for row in sample] for sample in reference.ids],
            features=features,
            mask=mask,
            source_confidence=reference.source_confidence.clone() * 0.0,
            conflict_score=(
                reference.conflict_score.clone() * 0.0
                if reference.conflict_score is not None
                else None
            ),
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    valid_mask = mask.clone()
    if query_sample_ids is None:
        indices = torch.randint(
            0,
            n_bank,
            tuple(reference.mask.shape),
            generator=generator,
            device="cpu",
        )
    else:
        if len(query_sample_ids) != reference.mask.size(0):
            raise ValueError("query_sample_ids must match batch size")
        indices = torch.zeros(reference.mask.shape, dtype=torch.long)
        for batch_index, sample_id in enumerate(query_sample_ids):
            valid_bank = [
                index
                for index, bank_sample in enumerate(retriever._sample_ids)
                if str(bank_sample) != str(sample_id)
            ]
            if not valid_bank:
                valid_mask[batch_index] = False
                continue
            values = torch.tensor(valid_bank, dtype=torch.long)
            sampled = values[
                torch.randint(
                    0,
                    values.numel(),
                    (reference.mask.size(1), reference.mask.size(2)),
                    generator=generator,
                )
            ]
            indices[batch_index] = sampled
    assert retriever._bank_full is not None
    selected = retriever._bank_full[indices].to(reference.features.device)
    keep = valid_mask.unsqueeze(-1)
    features = torch.where(keep, selected, features)
    ids = []
    for batch_index in range(reference.mask.size(0)):
        sample_ids = []
        for time_index in range(reference.mask.size(1)):
            rank_ids = []
            for rank in range(reference.mask.size(2)):
                if bool(valid_mask[batch_index, time_index, rank]):
                    bank_index = int(indices[batch_index, time_index, rank])
                    rank_ids.append(
                        f"{retriever._sample_ids[bank_index]}:"
                        f"{retriever._utterance_ids[bank_index]}"
                    )
                else:
                    rank_ids.append("")
            sample_ids.append(rank_ids)
        ids.append(sample_ids)
    return EvidenceBatch(
        ids=ids,
        features=features,
        mask=valid_mask,
        source_confidence=reference.source_confidence * valid_mask.to(
            reference.source_confidence.dtype
        ),
        conflict_score=(
            reference.conflict_score
            * valid_mask.to(reference.conflict_score.dtype)
            if reference.conflict_score is not None
            else None
        ),
    )
