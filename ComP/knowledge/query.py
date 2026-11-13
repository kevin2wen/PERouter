from __future__ import annotations

from typing import List, Optional

import torch

from ComP.per.types import MODALITIES, PERBatch

from .schema import EvidenceQuery


def _dialogue_context(
    batch: PERBatch,
    batch_index: int,
    target_index: int,
) -> Optional[torch.Tensor]:
    valid = batch.sequence_mask[batch_index]
    if not valid.any():
        return None
    columns = []
    for modality_index, modality in enumerate(MODALITIES):
        if modality_index == target_index:
            continue
        value = batch.features[modality][batch_index, valid]
        if value.numel():
            columns.append(value.mean(dim=0))
    if not columns:
        return None
    context = torch.cat(columns, dim=-1)
    if not torch.isfinite(context).all():
        raise FloatingPointError("dialogue context contains non-finite values")
    return context.detach()


def build_knowledge_queries(
    batch: PERBatch,
    target_modalities: Optional[torch.BoolTensor] = None,
) -> List[EvidenceQuery]:
    """Build one query per valid missing utterance-modality pair.

    Labels are never read. A missing target modality is never copied into the
    query, including when its feature tensor happens to contain nonzero values.
    """

    if batch.availability.dtype != torch.bool:
        raise TypeError("availability must be torch.bool")
    if batch.sequence_mask.dtype != torch.bool:
        raise TypeError("sequence_mask must be torch.bool")
    if batch.availability.shape != (*batch.sequence_mask.shape, 3):
        raise ValueError("availability must be [B,T,3]")
    if not torch.equal(
        batch.availability & ~batch.sequence_mask.unsqueeze(-1),
        torch.zeros_like(batch.availability),
    ):
        raise ValueError("padding positions must be unavailable")
    if target_modalities is None:
        target_modalities = (~batch.availability) & batch.sequence_mask.unsqueeze(-1)
    if target_modalities.shape != batch.availability.shape:
        raise ValueError("target_modalities must match availability [B,T,3]")
    if target_modalities.dtype != torch.bool:
        raise TypeError("target_modalities must be torch.bool")
    if (target_modalities & batch.availability).any():
        raise ValueError("an observed modality cannot be an evidence target")
    if (target_modalities & ~batch.sequence_mask.unsqueeze(-1)).any():
        raise ValueError("padding positions cannot be evidence targets")
    if len(batch.sample_ids) != batch.availability.size(0):
        raise ValueError("sample_ids must match batch size")
    if len(batch.utterance_ids) != batch.availability.size(0):
        raise ValueError("utterance_ids must match batch size")

    queries: List[EvidenceQuery] = []
    batch_size, seq_len = batch.sequence_mask.shape
    for batch_index in range(batch_size):
        sample_id = str(batch.sample_ids[batch_index])
        if len(batch.utterance_ids[batch_index]) != seq_len:
            raise ValueError("utterance_ids row must match sequence length")
        for time_index in range(seq_len):
            row_targets = torch.nonzero(
                target_modalities[batch_index, time_index], as_tuple=False
            ).flatten()
            if row_targets.numel() == 0:
                continue
            availability = tuple(
                bool(value)
                for value in batch.availability[batch_index, time_index].tolist()
            )
            missing_pattern = tuple(not value for value in availability)
            utterance_id = str(batch.utterance_ids[batch_index][time_index])
            for target_index_tensor in row_targets:
                target_index = int(target_index_tensor.item())
                target_modality = MODALITIES[target_index]
                text_query = None
                if (
                    target_modality != "text"
                    and availability[1]
                    and batch.raw_texts is not None
                ):
                    text = batch.raw_texts[batch_index][time_index]
                    if isinstance(text, str) and text.strip():
                        text_query = text.strip()
                audio_query = None
                if target_modality != "audio" and availability[0]:
                    audio_query = (
                        batch.features["audio"][batch_index, time_index]
                        .detach()
                        .clone()
                    )
                video_query = None
                if target_modality != "video" and availability[2]:
                    video_query = (
                        batch.features["video"][batch_index, time_index]
                        .detach()
                        .clone()
                    )
                query = EvidenceQuery(
                    sample_id=sample_id,
                    dialogue_id=sample_id,
                    utterance_id=utterance_id,
                    availability=availability,
                    missing_pattern=missing_pattern,
                    target_modality=target_modality,
                    text_query=text_query,
                    audio_query=audio_query,
                    video_query=video_query,
                    dialogue_context=_dialogue_context(
                        batch, batch_index, target_index
                    ),
                )
                queries.append(query)
    return queries
