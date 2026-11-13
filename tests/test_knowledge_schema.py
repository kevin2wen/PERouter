from __future__ import annotations

import unittest

import torch

from ComP.knowledge.schema import EvidenceQuery, KnowledgeCard, validate_card


def make_card():
    return KnowledgeCard(
        knowledge_id="k1",
        trigger_type="observed_text",
        trigger_modalities=("text",),
        trigger_text="calm sentence",
        trigger_visual_ref=None,
        trigger_audio_ref=None,
        scope="utterance",
        relation_type="supports",
        affect_prior="neutral",
        applicability=("audio",),
        evidence_text="The observed sentence is neutral.",
        supporting_context_ids=("ctx1",),
        conflict_context_ids=(),
        source_id="ctx1",
        source_type="offline_extraction",
        source_confidence=0.8,
        license="MIT",
        version="v1",
    )


class KnowledgeSchemaTest(unittest.TestCase):
    def test_card_requires_applicability_and_trigger(self):
        card = make_card()
        self.assertTrue(validate_card(card).valid)
        with self.assertRaises(ValueError):
            KnowledgeCard(
                **{
                    **card.__dict__,
                    "applicability": (),
                }
            )

    def test_query_missing_pattern_and_target_are_exclusive(self):
        query = EvidenceQuery(
            sample_id="s",
            dialogue_id="d",
            utterance_id="u",
            availability=(False, True, True),
            missing_pattern=(True, False, False),
            target_modality="audio",
            text_query="observed text",
            audio_query=None,
            video_query=torch.ones(2),
            dialogue_context=None,
        )
        self.assertIsNone(query.audio_query)
        with self.assertRaises(ValueError):
            EvidenceQuery(
                sample_id="s",
                dialogue_id="d",
                utterance_id="u",
                availability=(True, True, True),
                missing_pattern=(False, False, False),
                target_modality="audio",
                text_query="observed text",
                audio_query=torch.ones(2),
                video_query=torch.ones(2),
                dialogue_context=None,
            )
