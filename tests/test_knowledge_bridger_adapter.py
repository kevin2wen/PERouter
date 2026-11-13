from __future__ import annotations

import unittest

from ComP.knowledge.adapters.knowledge_bridger_adapter import (
    KnowledgeBridgerAdapter,
)
from ComP.knowledge.schema import RetrievedContext, StructuredComponent
import torch


def component():
    return StructuredComponent(
        component_id="c1",
        relation_type="supports",
        affect_prior="neutral",
        trigger_type="observed_text",
        trigger_modalities=("text",),
        applicability=("audio",),
        evidence_text="The retrieved context is affect-neutral.",
        supporting_context_ids=("train:1",),
        conflict_context_ids=(),
        source_id="train:1",
        source_type="offline_extractor",
        source_confidence=0.7,
        trigger_text="neutral wording",
        license="MIT",
        version="v1",
    )


class KnowledgeBridgerAdapterTest(unittest.TestCase):
    def test_label_and_ground_truth_keys_are_rejected(self):
        adapter = KnowledgeBridgerAdapter(lambda *args: [component()])
        with self.assertRaises(PermissionError):
            adapter.extract_components(
                {"label": 1},
                [],
                (False, True, True),
            )

    def test_cards_are_target_specific_and_deduplicated(self):
        adapter = KnowledgeBridgerAdapter(lambda *args: [component(), component()])
        contexts = [
            RetrievedContext(
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
        ]
        components = adapter.extract_components(
            {"raw_text": "observed"}, contexts, (False, True, True)
        )
        cards = adapter.construct_cards(components, "audio", 5)
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].applicability, ("audio",))
        self.assertEqual(adapter.construct_cards(components, "video", 5), [])
