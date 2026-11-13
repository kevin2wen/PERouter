from __future__ import annotations

import hashlib
from typing import Callable, Iterable, List, Mapping, Sequence

import torch

from ComP.per.types import MODALITIES

from ..schema import (
    KnowledgeCard,
    RetrievedContext,
    StructuredComponent,
    ValidationResult,
    validate_card,
)


def _normalise_target(value: str | int) -> str:
    if isinstance(value, int):
        if value not in range(3):
            raise ValueError("target modality index must be 0, 1, or 2")
        return MODALITIES[value]
    if value not in MODALITIES:
        raise ValueError(f"unknown target modality: {value}")
    return value


class KnowledgeBridgerAdapter:
    """Convert observed inputs and retrieved contexts into validated components.

    The extractor is injected so that an online VLM is never called implicitly.
    The adapter deliberately accepts no label or ground-truth modality input.
    """

    def __init__(
        self,
        extractor: Callable[
            [Mapping[str, object], Sequence[RetrievedContext], Sequence[bool]],
            Iterable[StructuredComponent | Mapping[str, object]],
        ],
        *,
        source_type: str = "knowledge_bridger_adapter",
        source_version: str = "v1",
        source_license: str = "unknown",
    ) -> None:
        if not callable(extractor):
            raise TypeError("extractor must be callable")
        self.extractor = extractor
        self.source_type = str(source_type)
        self.source_version = str(source_version)
        self.source_license = str(source_license)

    def extract_components(
        self,
        observed_inputs,
        retrieved_contexts: List[RetrievedContext],
        availability,
    ) -> List[StructuredComponent]:
        if isinstance(observed_inputs, Mapping):
            leaked = {
                "label",
                "labels",
                "GT_text",
                "GT_image",
                "GT_audio",
                "gt_text",
                "gt_image",
                "gt_audio",
            } & set(observed_inputs)
            if leaked:
                raise PermissionError(
                    f"forbidden label/ground-truth keys: {sorted(leaked)}"
                )
        if isinstance(availability, torch.Tensor):
            if availability.dtype != torch.bool:
                raise TypeError("availability must be bool")
            availability = availability.detach().cpu().tolist()
        if len(availability) != 3 or any(
            not isinstance(value, bool) for value in availability
        ):
            raise ValueError("availability must contain three bool values")
        for context in retrieved_contexts:
            if not isinstance(context, RetrievedContext):
                raise TypeError("retrieved_contexts must contain RetrievedContext")
        raw_components = self.extractor(
            observed_inputs, tuple(retrieved_contexts), tuple(availability)
        )
        components = []
        for raw in raw_components:
            if isinstance(raw, StructuredComponent):
                component = raw
            elif isinstance(raw, Mapping):
                component = StructuredComponent(**dict(raw))
            else:
                raise TypeError("extractor must return StructuredComponent or mapping")
            if not set(component.applicability) <= set(MODALITIES):
                raise ValueError("component has an unknown applicability modality")
            components.append(component)
        return components

    def construct_cards(
        self,
        components,
        target_modality: str | int,
        top_k: int,
    ) -> List[KnowledgeCard]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        target = _normalise_target(target_modality)
        cards_by_signature = {}
        for component in components:
            if not isinstance(component, StructuredComponent):
                raise TypeError("components must contain StructuredComponent")
            if target not in component.applicability:
                continue
            payload = "|".join(
                (
                    component.component_id,
                    component.source_id,
                    component.relation_type,
                    component.evidence_text,
                    ",".join(component.supporting_context_ids),
                    self.source_version,
                    target,
                )
            ).encode("utf-8")
            knowledge_id = hashlib.sha256(payload).hexdigest()
            card = KnowledgeCard(
                knowledge_id=knowledge_id,
                trigger_type=component.trigger_type,
                trigger_modalities=component.trigger_modalities,
                trigger_text=component.trigger_text,
                trigger_visual_ref=component.trigger_visual_ref,
                trigger_audio_ref=component.trigger_audio_ref,
                scope=component.scope,
                relation_type=component.relation_type,
                affect_prior=component.affect_prior,
                applicability=component.applicability,
                evidence_text=component.evidence_text,
                supporting_context_ids=component.supporting_context_ids,
                conflict_context_ids=component.conflict_context_ids,
                source_id=component.source_id,
                source_type=self.source_type,
                source_confidence=float(component.source_confidence),
                license=self.source_license,
                version=self.source_version,
            )
            result = validate_card(card)
            if not result.valid:
                raise ValueError("; ".join(result.errors))
            signature = (
                card.source_id,
                card.relation_type,
                card.evidence_text,
                card.supporting_context_ids,
            )
            existing = cards_by_signature.get(signature)
            if existing is None or card.source_confidence > existing.source_confidence:
                cards_by_signature[signature] = card
        cards = sorted(
            cards_by_signature.values(),
            key=lambda card: (-card.source_confidence, card.knowledge_id),
        )
        return cards[:top_k]

    @staticmethod
    def validate_card(card: KnowledgeCard) -> ValidationResult:
        return validate_card(card)
