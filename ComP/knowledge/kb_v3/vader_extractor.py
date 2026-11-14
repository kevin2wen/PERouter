from __future__ import annotations

import hashlib
import re
from typing import List, Mapping, Sequence

from ComP.per.types import MODALITIES

from ..schema import RetrievedContext, StructuredComponent

NEGATION = {"not", "never", "no", "nobody", "nothing", "none", "cannot",
            "can't", "don't", "doesn't", "isn't", "wasn't", "won't"}
CONTRAST = {"but", "however", "although", "though", "yet", "whereas", "instead"}
CONCESSION = {"despite", "nevertheless", "nonetheless"}


class VaderKnowledgeExtractor:
    """Build structured knowledge components from the VADER affect lexicon.

    VADER is an external, citable affect lexicon (Hutto & Gilbert, 2014,
    MIT-licensed distribution). Only observed context text is read; labels and
    target-modality content are never used.
    """

    def __init__(self) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self.analyzer = SentimentIntensityAnalyzer()
        self.lexicon = self.analyzer.lexicon

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return re.findall(r"[a-z']+", text.lower())

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
            tokens = self._tokens(text)
            hits = [
                (token, float(self.lexicon[token]))
                for token in tokens
                if token in self.lexicon
            ]
            scores = self.analyzer.polarity_scores(text) if text else {
                "neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0
            }
            has_neg = bool(set(tokens) & NEGATION)
            has_contrast = bool(set(tokens) & CONTRAST)
            has_concession = bool(set(tokens) & CONCESSION)
            compound = float(scores["compound"])
            if has_contrast:
                relation = "vader_contrast"
            elif has_concession:
                relation = "vader_concession"
            elif has_neg and compound < 0:
                relation = "vader_negation_negative"
            elif has_neg and compound > 0:
                relation = "vader_negation_positive"
            elif compound >= 0.3:
                relation = "vader_positive"
            elif compound <= -0.3:
                relation = "vader_negative"
            else:
                relation = "vader_neutral"
            if compound >= 0.3 and (has_contrast or has_concession):
                affect = "mixed"
            elif compound >= 0.3:
                affect = "positive"
            elif compound <= -0.3:
                affect = "negative"
            elif has_contrast or has_concession:
                affect = "mixed"
            else:
                affect = "neutral"
            cue = max(hits, key=lambda item: abs(item[1])) if hits else None
            if cue is None:
                evidence = (
                    f"VADER found no lexicon cue (compound {compound:+.2f}); "
                    f"relation={relation}; text: {text[:180]}"
                )
                confidence = 0.2
            else:
                token, valence = cue
                evidence = (
                    f"VADER cue '{token}' (valence {valence:+.1f}, "
                    f"compound {compound:+.2f}) indicates {affect} affect; "
                    f"relation={relation}; text: {text[:180]}"
                )
                confidence = min(1.0, 0.4 + abs(valence) / 4.0)
            identity = "|".join(
                (context.source_id, context.target_modality, relation, affect,
                 "kb_v3_vader")
            )
            component_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
            raw_text = observed_inputs.get("raw_text")
            trigger_text = (
                raw_text if isinstance(raw_text, str) and raw_text.strip() else None
            )
            components.append(
                StructuredComponent(
                    component_id=component_id,
                    relation_type=relation,
                    affect_prior=affect,
                    trigger_type="observed_modalities",
                    trigger_modalities=observed,
                    applicability=MODALITIES,
                    evidence_text=(
                        f"{evidence} [source: VADER; card: {context.source_id}; "
                        f"target: {context.target_modality}]"
                    ),
                    supporting_context_ids=(context.source_id,),
                    conflict_context_ids=(
                        (context.source_id,) if affect == "mixed" else ()
                    ),
                    source_id=context.source_id,
                    source_type="vader_lexicon",
                    source_confidence=float(confidence),
                    scope="utterance",
                    trigger_text=trigger_text,
                    trigger_visual_ref=(
                        "observed_video" if "video" in observed else None
                    ),
                    trigger_audio_ref=(
                        "observed_audio" if "audio" in observed else None
                    ),
                    license="MIT (vaderSentiment lexicon)",
                    version="kb_v3",
                    metadata={
                        "vader_compound": compound,
                        "lexicon_size": len(self.lexicon),
                    },
                )
            )
        return components
