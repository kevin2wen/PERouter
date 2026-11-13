from __future__ import annotations

import unittest

import torch

from ComP.knowledge.adapters.ragpt_adapter import RAGPTAdapter
from ComP.knowledge.schema import RetrievedContext


def context(sample_id: str, rank: int):
    return RetrievedContext(
        query_id="q",
        sample_id=sample_id,
        dialogue_id=sample_id,
        utterance_id="u",
        target_modality="audio",
        rank=rank,
        text="context",
        audio_feature=torch.ones(2),
        video_feature=torch.ones(2),
        retrieval_score=1.0 - 0.1 * rank,
        source_modalities=("text", "video"),
        source_id=f"{sample_id}:u",
        source_confidence=0.9 - 0.1 * rank,
    )


class RAGPTAdapterContractTest(unittest.TestCase):
    def test_query_does_not_read_missing_target(self):
        adapter = RAGPTAdapter(lambda query, k: [], top_k=5)
        query = adapter.build_query(
            {
                "sample_id": "s",
                "dialogue_id": "d",
                "utterance_id": "u",
                "raw_text": "observed",
                "video": torch.ones(2),
            },
            (False, True, True),
            "audio",
        )
        self.assertIsNone(query.audio_query)
        self.assertEqual(query.text_query, "observed")
        with self.assertRaises(PermissionError):
            adapter.build_query(
                {
                    "sample_id": "s",
                    "dialogue_id": "d",
                    "utterance_id": "u",
                    "raw_text": "observed",
                    "video": torch.ones(2),
                    "audio": torch.ones(2),
                },
                (False, True, True),
                "audio",
            )

    def test_same_video_is_filtered_and_ranks_are_contiguous(self):
        adapter = RAGPTAdapter(
            lambda query, k: [context("s", 0), context("other", 1)],
            top_k=5,
        )
        query = adapter.build_query(
            {
                "sample_id": "s",
                "dialogue_id": "s",
                "utterance_id": "u",
                "raw_text": "observed",
                "video": torch.ones(2),
            },
            (False, True, True),
            "audio",
        )
        contexts = adapter.retrieve_context(query, 5)
        self.assertEqual([item.sample_id for item in contexts], ["other"])
        self.assertEqual(contexts[0].rank, 0)
        self.assertTrue(all(not row["same_video"] for row in adapter.export_retrieval_trace()))
