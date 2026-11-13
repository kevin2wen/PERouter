from __future__ import annotations

import hashlib
import re
from typing import List, Mapping, Sequence

from .schema import RetrievedContext, StructuredComponent


_POSITIVE = frozenset(
    "amazing awesome beautiful enjoy enjoyed good great happy interesting love "
    "liked nice pleasant positive recommend wonderful".split()
)
_NEGATIVE = frozenset(
    "angry awful bad boring dislike disgusting dull hate hated horrible negative "
    "poor sad terrible ugly unpleasant worse worst".split()
)


class RuleBasedKnowledgeExtractor:
    """Offline schema-constrained extractor used before an external LMM."""

    def __call__(
        self,
        observed_inputs: Mapping[str, object],
        retrieved_contexts: Sequence[RetrievedContext],
        availability: Sequence[bool],
    ) -> List[StructuredComponent]:
        observed = tuple(
            name
            for name, value in zip(("audio", "text", "video"), availability)
            if value
        )
        trigger_text = observed_inputs.get("raw_text")
        if not isinstance(trigger_text, str) or not trigger_text.strip():
            trigger_text = None
        components = []
        for context in retrieved_contexts:
            text = context.text or ""
            tokens = set(re.findall(r"[a-z]+", text.lower()))
            positive = len(tokens & _POSITIVE)
            negative = len(tokens & _NEGATIVE)
            if positive and negative:
                affect = "mixed"
            elif positive:
                affect = "positive"
            elif negative:
                affect = "negative"
            else:
                affect = "neutral"
            relation = f"retrieved_{affect}_context"
            identity = "|".join(
                (
                    context.source_id,
                    context.target_modality,
                    relation,
                    text,
                    "knowledge-rule-v1",
                )
            )
            component_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            components.append(
                StructuredComponent(
                    component_id=component_id,
                    relation_type=relation,
                    affect_prior=affect,
                    trigger_type="observed_modalities",
                    trigger_modalities=observed,
                    applicability=("audio", "text", "video"),
                    evidence_text=(
                        text.strip()
                        or "feature-neighbor context with "
                        f"{affect} affect prior"
                    ),
                    supporting_context_ids=(context.source_id,),
                    conflict_context_ids=(),
                    source_id=context.source_id,
                    source_type="train_retrieved_rule",
                    source_confidence=context.source_confidence,
                    scope="utterance",
                    trigger_text=trigger_text,
                    trigger_audio_ref=(
                        "observed_audio" if "audio" in observed else None
                    ),
                    trigger_visual_ref=(
                        "observed_video" if "video" in observed else None
                    ),
                    license="research-only",
                    version="knowledge-rule-v1",
                    metadata={"retrieval_score": context.retrieval_score},
                )
            )
        return components
