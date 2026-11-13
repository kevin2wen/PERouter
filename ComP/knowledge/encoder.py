from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn

from ComP.per.types import MODALITIES
from .schema import KnowledgeCard, RetrievedContext


class RuleKnowledgeCardEncoder:
    """Deterministic card encoder for the first content gate.

    It preserves modality evidence and metadata, but does not learn semantic
    language embeddings. The existing target-specific EvidenceEncoder remains
    responsible for projection into ComP's hidden space.
    """

    def __init__(self, audio_dim: int, video_dim: int) -> None:
        if audio_dim <= 0 or video_dim <= 0:
            raise ValueError("audio_dim and video_dim must be positive")
        self.audio_dim = int(audio_dim)
        self.video_dim = int(video_dim)
        # affect(4) + applicability(3) + confidence(1)
        self.output_dim = self.audio_dim + self.video_dim + 8

    def encode(
        self,
        card: KnowledgeCard,
        contexts: Mapping[str, RetrievedContext],
    ) -> torch.Tensor:
        audio = []
        video = []
        for source_id in card.supporting_context_ids:
            context = contexts.get(source_id)
            if context is None:
                continue
            if context.audio_feature is not None:
                if context.audio_feature.numel() != self.audio_dim:
                    raise ValueError("audio context has invalid dimension")
                audio.append(context.audio_feature.float())
            if context.video_feature is not None:
                if context.video_feature.numel() != self.video_dim:
                    raise ValueError("video context has invalid dimension")
                video.append(context.video_feature.float())
        affected = tuple(
            name for name in card.applicability if name in {"audio", "text", "video"}
        )
        applicable = torch.tensor(
            [name in affected for name in ("audio", "text", "video")],
            dtype=torch.float32,
        )
        affect = torch.tensor(
            [
                card.affect_prior == "positive",
                card.affect_prior == "negative",
                card.affect_prior == "neutral",
                card.affect_prior == "mixed",
            ],
            dtype=torch.float32,
        )
        audio_feature = (
            torch.stack(audio).mean(dim=0)
            if audio
            else torch.zeros(self.audio_dim)
        )
        video_feature = (
            torch.stack(video).mean(dim=0)
            if video
            else torch.zeros(self.video_dim)
        )
        return torch.cat(
            [
                audio_feature,
                video_feature,
                affect,
                applicable,
                torch.tensor([float(card.source_confidence)]),
            ]
        )

    def encode_cards(
        self,
        cards: Sequence[KnowledgeCard],
        contexts: Sequence[RetrievedContext],
    ) -> dict[str, torch.Tensor]:
        by_id = {context.source_id: context for context in contexts}
        return {card.knowledge_id: self.encode(card, by_id) for card in cards}


class TargetSpecificCardProjector(nn.Module):
    """Project card features independently for each missing target modality."""

    def __init__(
        self,
        card_dim: int,
        evidence_dim: int,
        *,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if card_dim <= 0 or evidence_dim <= 0:
            raise ValueError("card_dim and evidence_dim must be positive")
        self.card_dim = int(card_dim)
        self.evidence_dim = int(evidence_dim)
        self.projections = nn.ModuleDict(
            {
                name: nn.Linear(self.card_dim, self.evidence_dim)
                for name in MODALITIES
            }
        )
        if zero_init:
            for projection in self.projections.values():
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)

    def forward(self, features: torch.Tensor, target_modality: str) -> torch.Tensor:
        if target_modality not in MODALITIES:
            raise ValueError(f"unknown target modality: {target_modality}")
        if features.ndim != 4 or features.size(-1) != self.card_dim:
            raise ValueError("features must be [B,T,K,card_dim]")
        return self.projections[target_modality](features)
