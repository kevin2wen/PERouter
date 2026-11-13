from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

import torch

from ComP.per.types import MODALITIES, EvidenceBatch, PERBatch, PEROutput

from .adapters.knowledge_bridger_adapter import KnowledgeBridgerAdapter
from .adapters.ragpt_adapter import RAGPTAdapter
from .cards import KnowledgeCandidateBuilder
from .encoder import RuleKnowledgeCardEncoder
from .query import build_knowledge_queries
from .schema import (
    EvidenceCandidate,
    EvidenceQuery,
    KnowledgeCard,
    RetrievedContext,
)


@dataclass
class KnowledgeBatchResult:
    queries: List[EvidenceQuery]
    contexts: List[RetrievedContext]
    cards: List[KnowledgeCard]
    candidates: Dict[str, List[EvidenceCandidate]]
    evidence_by_target: Dict[str, EvidenceBatch]


class KnowledgePipeline:
    def __init__(
        self,
        retrieval_adapter: RAGPTAdapter,
        bridger_adapter: KnowledgeBridgerAdapter,
        encoder: RuleKnowledgeCardEncoder,
        *,
        top_k: int,
        projector=None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.retrieval_adapter = retrieval_adapter
        self.bridger_adapter = bridger_adapter
        self.encoder = encoder
        self.projector = projector
        self.candidate_builder = KnowledgeCandidateBuilder()
        self.top_k = int(top_k)

    def build(self, batch: PERBatch) -> KnowledgeBatchResult:
        if (batch.availability & ~batch.sequence_mask.unsqueeze(-1)).any():
            raise ValueError("padding positions must be unavailable")
        positions = {}
        for batch_index, sample_id in enumerate(batch.sample_ids):
            for time_index, utterance_id in enumerate(batch.utterance_ids[batch_index]):
                if not bool(batch.sequence_mask[batch_index, time_index]):
                    continue
                for target in MODALITIES:
                    positions[(str(sample_id), str(utterance_id), target)] = (
                        batch_index,
                        time_index,
                    )
        queries = build_knowledge_queries(batch)
        contexts: List[RetrievedContext] = []
        cards: List[KnowledgeCard] = []
        candidates_by_target = {name: [] for name in MODALITIES}
        position_by_query_id = {}
        for query in queries:
            query_contexts = self.retrieval_adapter.retrieve_context(
                query, self.top_k
            )
            observed_inputs = self._observed_inputs(query)
            components = self.bridger_adapter.extract_components(
                observed_inputs, query_contexts, query.availability
            )
            query_cards = self.bridger_adapter.construct_cards(
                components, query.target_modality, self.top_k
            )
            encoded = self.encoder.encode_cards(query_cards, query_contexts)
            query_id = self._query_id(query)
            position_by_query_id[query_id] = positions[
                (query.sample_id, query.utterance_id, query.target_modality)
            ]
            candidates = self.candidate_builder.build(
                query_id,
                query.target_modality,
                query_contexts,
                query_cards,
                encoded,
                top_k=self.top_k,
            )
            contexts.extend(query_contexts)
            cards.extend(query_cards)
            candidates_by_target[query.target_modality].extend(candidates)
        evidence_by_target = {
            target: self._materialize(
                batch,
                position_by_query_id,
                candidates_by_target[target],
                target,
            )
            for target in MODALITIES
        }
        return KnowledgeBatchResult(
            queries=queries,
            contexts=contexts,
            cards=cards,
            candidates=candidates_by_target,
            evidence_by_target=evidence_by_target,
        )

    @staticmethod
    def _query_id(query: EvidenceQuery) -> str:
        payload = (
            f"{query.sample_id}|{query.dialogue_id}|{query.utterance_id}|"
            f"{query.availability}|{query.target_modality}"
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _observed_inputs(query: EvidenceQuery) -> Mapping[str, object]:
        result: Dict[str, object] = {
            "sample_id": query.sample_id,
            "dialogue_id": query.dialogue_id,
            "utterance_id": query.utterance_id,
            "dialogue_context": query.dialogue_context,
        }
        if query.availability[0]:
            result["audio"] = query.audio_query
        if query.availability[1]:
            result["raw_text"] = query.text_query
        if query.availability[2]:
            result["video"] = query.video_query
        return result

    def _materialize(
        self,
        batch: PERBatch,
        position_by_query_id: Mapping[str, Tuple[int, int]],
        candidates,
        target: str,
    ) -> EvidenceBatch:
        batch_size, seq_len = batch.sequence_mask.shape
        feature_dim = self.encoder.output_dim
        projected_dim = (
            self.projector.evidence_dim
            if self.projector is not None
            else feature_dim
        )
        features = torch.zeros(
            batch_size,
            seq_len,
            self.top_k,
            projected_dim if self.projector is None else feature_dim,
            dtype=batch.features[MODALITIES[0]].dtype,
            device=batch.features[MODALITIES[0]].device,
        )
        mask = torch.zeros(
            batch_size,
            seq_len,
            self.top_k,
            dtype=torch.bool,
            device=features.device,
        )
        confidence = torch.zeros(
            batch_size,
            seq_len,
            self.top_k,
            dtype=features.dtype,
            device=features.device,
        )
        conflict = torch.zeros_like(confidence)
        ids = [
            [["" for _ in range(self.top_k)] for _ in range(seq_len)]
            for _ in range(batch_size)
        ]
        for candidate in candidates:
            matched = position_by_query_id.get(candidate.query_id)
            if matched is None:
                continue
            batch_index, time_index = matched
            if candidate.rank >= self.top_k:
                continue
            features[batch_index, time_index, candidate.rank] = (
                candidate.encoded_feature.to(features)
            )
            mask[batch_index, time_index, candidate.rank] = True
            confidence[batch_index, time_index, candidate.rank] = float(
                candidate.source_confidence
            )
            ids[batch_index][time_index][candidate.rank] = candidate.card.knowledge_id
        if self.projector is not None:
            features = self.projector(features, target)
        result = EvidenceBatch(
            ids=ids,
            features=features,
            mask=mask,
            source_confidence=confidence,
            conflict_score=conflict,
        )
        result.validate(batch_size=batch_size, seq_len=seq_len)
        return result


def merge_evidence_for_forward(
    evidence_by_target: Mapping[str, EvidenceBatch],
) -> EvidenceBatch:
    first = evidence_by_target[MODALITIES[0]]
    features = first.features.clone()
    mask = first.mask.clone()
    confidence = first.source_confidence.clone()
    conflict = (
        first.conflict_score.clone()
        if first.conflict_score is not None
        else None
    )
    ids = [[list(rank_ids) for rank_ids in sample] for sample in first.ids]
    for target in MODALITIES[1:]:
        evidence = evidence_by_target[target]
        insert = evidence.mask & ~mask
        expanded = insert.unsqueeze(-1)
        features = torch.where(expanded, evidence.features, features)
        confidence = torch.where(insert, evidence.source_confidence, confidence)
        if conflict is not None and evidence.conflict_score is not None:
            conflict = torch.where(insert, evidence.conflict_score, conflict)
        mask = mask | evidence.mask
        for batch_index in range(mask.size(0)):
            for time_index in range(mask.size(1)):
                for rank in range(mask.size(2)):
                    if bool(insert[batch_index, time_index, rank]):
                        ids[batch_index][time_index][rank] = evidence.ids[
                            batch_index
                        ][time_index][rank]
    return EvidenceBatch(
        ids=ids,
        features=features,
        mask=mask,
        source_confidence=confidence,
        conflict_score=conflict,
    )


def encode_target_specific_evidence(
    model,
    batch: PERBatch,
    evidence_by_target: Mapping[str, EvidenceBatch],
) -> Tuple[Dict[str, torch.Tensor], torch.BoolTensor]:
    encoder = getattr(model, "evidence_encoder", None)
    if encoder is None:
        raise ValueError("model has no evidence_encoder")
    batch_size, seq_len = batch.sequence_mask.shape
    hidden = model.dim
    dtype = batch.features[MODALITIES[0]].dtype
    device = batch.features[MODALITIES[0]].device
    vectors = {
        name: torch.zeros(batch_size, seq_len, hidden, dtype=dtype, device=device)
        for name in MODALITIES
    }
    active = torch.zeros(
        batch_size, seq_len, 3, dtype=torch.bool, device=device
    )
    for target_index, target in enumerate(MODALITIES):
        evidence = evidence_by_target[target]
        target_mask = torch.zeros_like(batch.availability)
        target_mask[:, :, target_index] = (
            (~batch.availability[:, :, target_index])
            & evidence.mask.any(dim=-1)
        )
        encoded = encoder(evidence, batch.availability, target_mask)
        vectors[target] = vectors[target] + encoded[target]
        active[:, :, target_index] = target_mask[:, :, target_index]
    empty = ~active.any(dim=-1)
    if empty.any():
        for name in MODALITIES:
            vectors[name] = vectors[name] * (~empty).unsqueeze(-1).to(dtype)
    return vectors, active


def knowledge_forward(
    model,
    batch: PERBatch,
    result: KnowledgeBatchResult,
    *,
    gamma: float = 0.5,
    zero_content: bool = False,
) -> PEROutput:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0,1]")
    if zero_content:
        vectors = {
            name: torch.zeros(
                *batch.sequence_mask.shape,
                model.dim,
                dtype=batch.features[MODALITIES[0]].dtype,
                device=batch.features[MODALITIES[0]].device,
            )
            for name in MODALITIES
        }
        active = torch.stack(
            [
                (~batch.availability[:, :, index])
                & result.evidence_by_target[target].mask.any(dim=-1)
                for index, target in enumerate(MODALITIES)
            ],
            dim=-1,
        )
    else:
        vectors, active = encode_target_specific_evidence(
            model, batch, result.evidence_by_target
        )
    evidence = merge_evidence_for_forward(result.evidence_by_target)
    route = active.to(vectors[MODALITIES[0]].dtype) * float(gamma)
    return model(
        batch.features,
        batch.availability,
        evidence=evidence,
        return_intermediates=True,
        route_override=route,
        evidence_vectors_override=vectors,
    )
