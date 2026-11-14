from __future__ import annotations

import hashlib
import re
from typing import List, Mapping, Sequence

from ComP.per.types import MODALITIES

from .schema import RetrievedContext, StructuredComponent

NEGATION = {"not", "never", "no", "nobody", "nothing", "none", "cannot", "can't", "don't", "doesn't", "isn't", "wasn't", "won't"}
CONTRAST = {"but", "however", "although", "though", "yet", "whereas", "instead"}
CONCESSION = {"despite", "nevertheless", "nonetheless", "even"}
POSITIVE = {
    "good", "great", "happy", "love", "like", "enjoy", "best", "amazing",
    "wonderful", "funny", "fun", "nice", "excellent", "positive", "glad",
    "excited", "beautiful", "perfect", "fantastic", "awesome",
}
NEGATIVE = {
    "bad", "terrible", "hate", "sad", "angry", "worst", "awful", "boring",
    "annoying", "disappointing", "negative", "upset", "stupid", "poor",
    "horrible", "disgusting", "wrong", "problem", "fail", "failed",
}

TEMPLATES = {
    "negation_scope": (
        "The observed cue contains a negation marker, so its literal sentiment "
        "may be reversed: {text}"
    ),
    "contrast_shift": (
        "A contrast marker signals a possible affective shift in the utterance: {text}"
    ),
    "concession_shift": (
        "A concession marker suggests a qualified or mixed affective stance: {text}"
    ),
    "conflict_cue": (
        "Positive and negative cues co-occur, indicating an affective conflict: {text}"
    ),
    "negative_cue": (
        "The observed cue carries negative affect that may support a missing target: {text}"
    ),
    "positive_cue": (
        "The observed cue carries positive affect that may support a missing target: {text}"
    ),
    "neutral_context": (
        "The observed cue provides neutral contextual evidence for the target: {text}"
    ),
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower()))


class RuleBasedKnowledgeExtractorV2:
    """Rule extractor with explicit negation/contrast/concession/affect cues.

    Text is derived only from retrieved contexts and observed modality inputs;
    labels and target-modality content are never read.
    """

    def __call__(
        self,
        observed_inputs: Mapping[str, object],
        retrieved_contexts: Sequence[RetrievedContext],
        availability: Sequence[bool],
    ) -> List[StructuredComponent]:
        observed = tuple(
            name for name, value in zip(MODALITIES, availability) if value
        )
        components = []
        for context in retrieved_contexts:
            text = (context.text or "").strip()
            tokens = _tokens(text)
            has_neg = bool(tokens & NEGATION)
            has_contrast = bool(tokens & CONTRAST)
            has_concession = bool(tokens & CONCESSION)
            pos = len(tokens & POSITIVE)
            neg = len(tokens & NEGATIVE)
            if pos and neg:
                affect = "mixed"
                relation = "conflict_cue"
            elif has_contrast:
                affect = "mixed" if (pos or neg) else "neutral"
                relation = "contrast_shift"
            elif has_concession:
                affect = "mixed" if (pos or neg) else "neutral"
                relation = "concession_shift"
            elif has_neg:
                affect = "negative" if neg >= pos else "neutral"
                relation = "negation_scope"
            elif neg:
                affect = "negative"
                relation = "negative_cue"
            elif pos:
                affect = "positive"
                relation = "positive_cue"
            else:
                affect = "neutral"
                relation = "neutral_context"
            evidence_text = TEMPLATES[relation].format(
                text=text[:220] if text else "(no raw text)"
            )
            evidence_text = (
                f"{evidence_text} [source: {context.source_id}; "
                f"target: {context.target_modality}; affect: {affect}]"
            )
            identity = "|".join(
                (
                    context.source_id,
                    context.target_modality,
                    relation,
                    affect,
                    "knowledge-rule-v2",
                )
            )
            component_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
            raw_text = observed_inputs.get("raw_text")
            trigger_text = raw_text if isinstance(raw_text, str) and raw_text.strip() else None
            trigger_visual_ref = "observed_video" if "video" in observed else None
            trigger_audio_ref = "observed_audio" if "audio" in observed else None
            components.append(
                StructuredComponent(
                    component_id=component_id,
                    relation_type=relation,
                    affect_prior=affect,
                    trigger_type="observed_modalities",
                    trigger_modalities=observed,
                    applicability=MODALITIES,
                    evidence_text=evidence_text,
                    supporting_context_ids=(context.source_id,),
                    conflict_context_ids=(
                        (context.source_id,) if relation == "conflict_cue" else ()
                    ),
                    source_id=context.source_id,
                    source_type="train_retrieved_rule_v2",
                    source_confidence=float(context.source_confidence),
                    scope="utterance",
                    trigger_text=trigger_text,
                    trigger_visual_ref=trigger_visual_ref,
                    trigger_audio_ref=trigger_audio_ref,
                    license="research-only",
                    version="knowledge-rule-v2",
                )
            )
        return components
