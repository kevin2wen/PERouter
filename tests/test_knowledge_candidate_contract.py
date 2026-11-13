from __future__ import annotations

import unittest

import torch

from ComP.knowledge.cards import KnowledgeCandidateBuilder
from ComP.knowledge.schema import KnowledgeCard, RetrievedContext


def make_context():
    return RetrievedContext(
        query_id="q",
        sample_id="train",
        dialogue_id="train",
        utterance_id="1",
        target_modality="audio",
        rank=0,
        text="context",
        audio_feature=torch.ones(2),
        video_feature=torch.ones(2),
        retrieval_score=0.9,
        source_modalities=("text", "video"),
        source_id="train:1",
        source_confidence=0.8,
    )


def make_card(applicability):
    return KnowledgeCard(
        knowledge_id="k1",
        trigger_type="observed_text",
        trigger_modalities=("text",),
        trigger_text="observed",
        trigger_visual_ref=None,
        trigger_audio_ref=None,
        scope="utterance",
        relation_type="supports",
        affect_prior="neutral",
        applicability=applicability,
        evidence_text="A neutral relation.",
        supporting_context_ids=("train:1",),
        conflict_context_ids=(),
        source_id="train:1",
        source_type="offline",
        source_confidence=0.7,
        license="MIT",
        version="v1",
    )


class KnowledgeCandidateContractTest(unittest.TestCase):
    def test_candidates_are_target_specific_and_contiguous(self):
        builder = KnowledgeCandidateBuilder()
        candidates = builder.build(
            "q",
            "audio",
            [make_context()],
            [make_card(("audio",)), make_card(("video",))],
            {"k1": torch.ones(4)},
            top_k=5,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].rank, 0)
        self.assertEqual(candidates[0].target_modality, "audio")

    def test_missing_encoded_feature_is_an_error(self):
        builder = KnowledgeCandidateBuilder()
        with self.assertRaises(KeyError):
            builder.build(
                "q",
                "audio",
                [make_context()],
                [make_card(("audio",))],
                {},
                top_k=5,
            )
