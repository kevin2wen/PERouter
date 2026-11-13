from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from ComP.per.types import MODALITIES

from ..schema import EvidenceQuery, RetrievedContext


def _target_name(target_modality: str | int) -> str:
    if isinstance(target_modality, int):
        if target_modality not in range(3):
            raise ValueError("target modality index must be 0, 1, or 2")
        return MODALITIES[target_modality]
    if target_modality not in MODALITIES:
        raise ValueError(f"unknown target modality: {target_modality}")
    return target_modality


def _query_id(query: EvidenceQuery) -> str:
    payload = (
        f"{query.sample_id}|{query.dialogue_id}|{query.utterance_id}|"
        f"{query.availability}|{query.target_modality}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class RAGPTAdapter:
    """Availability-aware adapter around an injected, train-only retriever.

    ``retrieve_fn(query, top_k)`` may use RAGPT's retrieval scoring or a cached
    context store. Generator and dynamic-prompt modules are deliberately
    outside this class. This callback boundary keeps third-party APIs out of
    ComP and makes the adapter testable without model assets.
    """

    def __init__(
        self,
        retrieve_fn: Callable[[EvidenceQuery, int], Iterable[RetrievedContext]],
        *,
        top_k: int,
        bank_split: str = "train",
        context_texts: Optional[Mapping[str, str]] = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if bank_split != "train":
            raise PermissionError("knowledge retrieval bank must use the train split")
        if not callable(retrieve_fn):
            raise TypeError("retrieve_fn must be callable")
        self.retrieve_fn = retrieve_fn
        self.top_k = int(top_k)
        self.bank_split = bank_split
        self.context_texts = dict(context_texts or {})
        self._trace: List[Dict[str, object]] = []

    def build_query(
        self,
        observed_inputs: Mapping[str, Any],
        availability: Sequence[bool] | torch.BoolTensor,
        target_modality: str | int,
    ) -> EvidenceQuery:
        forbidden = {
            "label",
            "labels",
            "target",
            "targets",
            "GT_text",
            "GT_image",
            "GT_audio",
            "gt_text",
            "gt_image",
            "gt_audio",
        }
        leaked = forbidden & set(observed_inputs)
        if leaked:
            raise PermissionError(f"forbidden label/ground-truth keys: {sorted(leaked)}")
        if isinstance(availability, torch.Tensor):
            if availability.dtype != torch.bool:
                raise TypeError("availability must be bool")
            availability = availability.detach().cpu().tolist()
        if len(availability) != 3 or any(
            not isinstance(value, bool) for value in availability
        ):
            raise ValueError("availability must contain three bool values")
        availability = tuple(availability)
        target = _target_name(target_modality)
        target_index = MODALITIES.index(target)
        if availability[target_index]:
            raise ValueError("target modality must be missing")
        for name in ("sample_id", "dialogue_id", "utterance_id"):
            value = observed_inputs.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"observed_inputs[{name!r}] must be a non-empty string")

        target_key = target
        if observed_inputs.get(target_key) is not None:
            raise PermissionError(
                f"target modality {target!r} must not be supplied to query construction"
            )
        text_query = None
        if target != "text" and availability[1]:
            raw_text = observed_inputs.get("raw_text")
            if raw_text is not None:
                if not isinstance(raw_text, str):
                    raise TypeError("raw_text must be a string")
                text_query = raw_text.strip() or None
        audio_query = None
        if target != "audio" and availability[0]:
            audio_query = observed_inputs.get("audio")
            if not isinstance(audio_query, torch.Tensor):
                raise TypeError("observed audio must be a torch.Tensor")
            audio_query = audio_query.detach().clone()
        video_query = None
        if target != "video" and availability[2]:
            video_query = observed_inputs.get("video")
            if not isinstance(video_query, torch.Tensor):
                raise TypeError("observed video must be a torch.Tensor")
            video_query = video_query.detach().clone()
        dialogue_context = observed_inputs.get("dialogue_context")
        if dialogue_context is not None and not isinstance(
            dialogue_context, torch.Tensor
        ):
            raise TypeError("dialogue_context must be a torch.Tensor or None")
        return EvidenceQuery(
            sample_id=observed_inputs["sample_id"],
            dialogue_id=observed_inputs["dialogue_id"],
            utterance_id=observed_inputs["utterance_id"],
            availability=availability,
            missing_pattern=tuple(not value for value in availability),
            target_modality=target,
            text_query=text_query,
            audio_query=audio_query,
            video_query=video_query,
            dialogue_context=dialogue_context,
        )

    def retrieve_context(
        self,
        query: EvidenceQuery,
        top_k: int,
    ) -> List[RetrievedContext]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_k > self.top_k:
            raise ValueError(
                f"requested top_k={top_k} exceeds adapter top_k={self.top_k}"
            )
        qid = _query_id(query)
        observed_sources = tuple(
            name for name, observed in zip(MODALITIES, query.availability) if observed
        )
        contexts: List[RetrievedContext] = []
        for raw_context in self.retrieve_fn(query, top_k):
            if not isinstance(raw_context, RetrievedContext):
                raise TypeError("retrieve_fn must return RetrievedContext objects")
            if raw_context.target_modality != query.target_modality:
                raise ValueError("retrieved context target modality does not match query")
            if raw_context.sample_id in {query.sample_id, query.dialogue_id}:
                continue
            source_sample = raw_context.sample_id
            source_utterance = raw_context.utterance_id
            source_id = raw_context.source_id
            context_text = raw_context.text
            if context_text is None:
                context_text = self.context_texts.get(source_id)
            context = RetrievedContext(
                query_id=qid,
                sample_id=source_sample,
                dialogue_id=raw_context.dialogue_id,
                utterance_id=source_utterance,
                target_modality=raw_context.target_modality,
                rank=len(contexts),
                text=context_text,
                audio_feature=raw_context.audio_feature,
                video_feature=raw_context.video_feature,
                retrieval_score=float(raw_context.retrieval_score),
                source_modalities=tuple(observed_sources),
                source_id=source_id,
                source_confidence=float(raw_context.source_confidence),
            )
            contexts.append(context)
            self._trace.append(
                {
                    "query_id": qid,
                    "sample_id": query.sample_id,
                    "dialogue_id": query.dialogue_id,
                    "utterance_id": query.utterance_id,
                    "target_modality": query.target_modality,
                    "availability": list(query.availability),
                    "missing_pattern": list(query.missing_pattern),
                    "query_source_modalities": list(observed_sources),
                    "rank": context.rank,
                    "source_id": context.source_id,
                    "source_sample_id": context.sample_id,
                    "source_utterance_id": context.utterance_id,
                    "retrieval_score": context.retrieval_score,
                    "source_confidence": context.source_confidence,
                    "source_split": self.bank_split,
                    "same_video": context.sample_id == query.sample_id,
                    "same_dialogue": (
                        context.dialogue_id == query.dialogue_id
                    ),
                    "same_utterance": (
                        context.sample_id == query.sample_id
                        and context.utterance_id == query.utterance_id
                    ),
                }
            )
        return contexts

    def export_retrieval_trace(self) -> List[dict]:
        return [dict(row) for row in self._trace]
