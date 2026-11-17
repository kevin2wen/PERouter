from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class KnowledgeCardV4:
    card_id: str
    claim: str
    source: str
    source_type: str
    source_version: str
    source_confidence: float
    observed_modalities: Tuple[str, ...]
    handled_missing_modalities: Tuple[str, ...]
    trigger_predicates: Tuple[str, ...]
    negative_conditions: Tuple[str, ...]
    evidence_relation: str
    expected_effect_direction: str
    supporting_context: Tuple[str, ...] = ()
    conflict_with: Tuple[str, ...] = ()
    utility_prior: float = 0.0
    provenance: Dict[str, object] = field(default_factory=dict)
    license: str = "research-only"

    def validate(self) -> list[str]:
        errors = []
        if not self.card_id or not self.claim:
            errors.append("card_id and claim must be non-empty")
        if not self.source or not self.source_type:
            errors.append("source provenance missing")
        if not self.observed_modalities:
            errors.append("observed modalities missing")
        if not self.handled_missing_modalities:
            errors.append("handled missing modalities missing")
        if self.source_confidence < 0 or self.source_confidence > 1:
            errors.append("source_confidence out of range")
        if "label" in self.provenance or "current_label" in self.provenance:
            errors.append("label must not be stored in provenance")
        return errors
