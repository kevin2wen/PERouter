"""Adapters that keep third-party retrieval/extraction outside ComP's core."""

from .knowledge_bridger_adapter import KnowledgeBridgerAdapter
from .ragpt_adapter import RAGPTAdapter

__all__ = ["KnowledgeBridgerAdapter", "RAGPTAdapter"]
