"""Independent Filter v2 implementation.

This package is separate from the legacy KnowledgeEvidenceFilter and
EvidenceReranker. Rollback means continuing to use the legacy scripts; no
existing filter file is modified by this package.
"""

from .model import FilterV2Model
from .action import FilterV2ActionSelector

__all__ = ["FilterV2Model", "FilterV2ActionSelector"]
