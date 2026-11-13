"""Structured knowledge retrieval and card contracts for ComP-PER."""

from .schema import (
    AFFECT_PRIORS,
    CARD_SCOPES,
    EvidenceCandidate,
    EvidencePolicyOutput,
    EvidenceQuery,
    KnowledgeCard,
    RetrievedContext,
    StructuredComponent,
    ValidationResult,
    validate_card,
)
from .encoder import RuleKnowledgeCardEncoder, TargetSpecificCardProjector
from .extractor import RuleBasedKnowledgeExtractor
from .pipeline import KnowledgePipeline, knowledge_forward
from .retriever import KnowledgeContextRetriever

__all__ = [
    "AFFECT_PRIORS",
    "CARD_SCOPES",
    "EvidenceCandidate",
    "EvidencePolicyOutput",
    "EvidenceQuery",
    "KnowledgeCard",
    "KnowledgeContextRetriever",
    "KnowledgePipeline",
    "RetrievedContext",
    "RuleBasedKnowledgeExtractor",
    "RuleKnowledgeCardEncoder",
    "StructuredComponent",
    "TargetSpecificCardProjector",
    "ValidationResult",
    "knowledge_forward",
    "validate_card",
]
