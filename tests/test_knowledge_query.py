from __future__ import annotations

import unittest

import torch

from ComP.knowledge.query import build_knowledge_queries
from ComP.per.types import PERBatch


class KnowledgeQueryTest(unittest.TestCase):
    def test_missing_target_is_not_copied_into_query(self):
        batch = PERBatch(
            features={
                "audio": torch.tensor([[[9.0, 9.0], [1.0, 2.0]]]),
                "text": torch.tensor([[[3.0], [4.0]]]),
                "video": torch.tensor([[[5.0], [6.0]]]),
            },
            availability=torch.tensor([[[False, True, True], [True, True, True]]]),
            sequence_mask=torch.tensor([[True, True]]),
            labels=torch.zeros(1, 2),
            sample_ids=["sample"],
            utterance_ids=[["u0", "u1"]],
            evidence=None,
            raw_texts=[["do not use", "observed text"]],
        )
        queries = build_knowledge_queries(batch)
        self.assertEqual(len(queries), 1)
        query = queries[0]
        self.assertEqual(query.target_modality, "audio")
        self.assertIsNone(query.audio_query)
        self.assertEqual(query.text_query, "do not use")
        self.assertEqual(query.availability, (False, True, True))
        self.assertEqual(query.missing_pattern, (True, False, False))

    def test_padding_cannot_be_target(self):
        batch = PERBatch(
            features={
                "audio": torch.zeros(1, 2, 2),
                "text": torch.zeros(1, 2, 1),
                "video": torch.zeros(1, 2, 1),
            },
            availability=torch.zeros(1, 2, 3, dtype=torch.bool),
            sequence_mask=torch.tensor([[True, False]]),
            labels=torch.zeros(1, 2),
            sample_ids=["sample"],
            utterance_ids=[["u0", "u1"]],
            evidence=None,
        )
        with self.assertRaises(ValueError):
            build_knowledge_queries(batch)
