from __future__ import annotations

import unittest
import tempfile

import torch

from ComP.knowledge.adapters.knowledge_bridger_adapter import (
    KnowledgeBridgerAdapter,
)
from ComP.knowledge.adapters.ragpt_adapter import RAGPTAdapter
from ComP.knowledge.encoder import RuleKnowledgeCardEncoder
from ComP.knowledge.encoder import TargetSpecificCardProjector
from ComP.knowledge.extractor import RuleBasedKnowledgeExtractor
from ComP.knowledge.pipeline import (
    KnowledgePipeline,
    encode_target_specific_evidence,
    knowledge_forward,
)
from ComP.knowledge.retriever import KnowledgeContextRetriever
from ComP.per.types import PERBatch
from tests.helpers import tiny_model


def make_batch(sample_ids, availability=None):
    batch_size = len(sample_ids)
    seq_len = 4
    return PERBatch(
        features={
            "audio": torch.randn(batch_size, seq_len, 5),
            "text": torch.randn(batch_size, seq_len, 6),
            "video": torch.randn(batch_size, seq_len, 7),
        },
        availability=(
            torch.ones(batch_size, seq_len, 3, dtype=torch.bool)
            if availability is None
            else availability
        ),
        sequence_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
        labels=torch.zeros(batch_size, seq_len),
        sample_ids=list(sample_ids),
        utterance_ids=[
            [
                f"{sample_id}-u0",
                f"{sample_id}-u1",
                f"{sample_id}-u2",
                f"{sample_id}-u3",
            ]
            for sample_id in sample_ids
        ],
        evidence=None,
        raw_texts=[
            [
                f"{sample_id} good happy",
                f"{sample_id} bad sad",
                f"{sample_id} neutral",
                f"{sample_id} great",
            ]
            for sample_id in sample_ids
        ],
    )


class KnowledgePipelineTest(unittest.TestCase):
    def make_pipeline(self):
        train = make_batch(["train-a", "train-b", "train-c"])
        bank = KnowledgeContextRetriever(
            {"audio": 5, "text": 6, "video": 7},
            top_k=2,
            min_df=1,
        )
        bank.add_batch(train, split="train")
        bank.fit()
        retrieval = RAGPTAdapter(bank.retrieve, top_k=2)
        bridger = KnowledgeBridgerAdapter(
            RuleBasedKnowledgeExtractor(),
            source_license="research-only",
            source_version="knowledge-rule-v1",
        )
        encoder = RuleKnowledgeCardEncoder(audio_dim=5, video_dim=7)
        return KnowledgePipeline(
            retrieval, bridger, encoder, top_k=2
        ), encoder

    def test_end_to_end_target_specific_forward(self):
        pipeline, encoder = self.make_pipeline()
        availability = torch.ones(1, 4, 3, dtype=torch.bool)
        availability[:, :, 1] = False
        batch = make_batch(["validation"], availability)
        result = pipeline.build(batch)
        self.assertTrue(result.queries)
        self.assertTrue(result.cards)
        self.assertTrue(
            result.evidence_by_target["text"].mask.any()
        )
        model = tiny_model(evidence_dim=encoder.output_dim).eval()
        with torch.no_grad():
            output = knowledge_forward(model, batch, result, gamma=0.5)
        self.assertGreaterEqual(float(output.route[0, 0, 1]), 0.0)
        self.assertTrue(
            torch.allclose(
                output.route[batch.availability],
                torch.zeros_like(output.route[batch.availability]),
            )
        )

        with torch.no_grad():
            source_only_output = knowledge_forward(
                model, batch, result, gamma=0.5, zero_content=True
            )
        for value in source_only_output.evidence.values():
            self.assertTrue(torch.equal(value, torch.zeros_like(value)))

    def test_non_train_bank_is_rejected(self):
        batch = make_batch(["train-a"])
        bank = KnowledgeContextRetriever(
            {"audio": 5, "text": 6, "video": 7}, top_k=1
        )
        with self.assertRaises(PermissionError):
            bank.add_batch(batch, split="validation")

    def test_train_index_roundtrip(self):
        train = make_batch(["train-a", "train-b"])
        bank = KnowledgeContextRetriever(
            {"audio": 5, "text": 6, "video": 7}, top_k=2, min_df=1
        )
        bank.add_batch(train, split="train")
        bank.fit()
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/knowledge_index.pt"
            bank.save(path)
            loaded = KnowledgeContextRetriever.load(path)
        self.assertEqual(len(loaded), len(bank))
        self.assertEqual(
            loaded.source_fingerprint(), bank.source_fingerprint()
        )

    def test_projector_has_target_specific_gradients(self):
        pipeline, encoder = self.make_pipeline()
        projector = TargetSpecificCardProjector(
            encoder.output_dim,
            evidence_dim=9,
            zero_init=False,
        )
        pipeline.projector = projector
        availability = torch.ones(1, 4, 3, dtype=torch.bool)
        availability[:, :, 1] = False
        batch = make_batch(["validation"], availability)
        result = pipeline.build(batch)
        vectors, active = encode_target_specific_evidence(
            tiny_model(evidence_dim=9), batch, result.evidence_by_target
        )
        loss = sum(value.square().mean() for value in vectors.values())
        loss.backward()
        self.assertIsNotNone(projector.projections["text"].weight.grad)
        self.assertTrue(active[:, :, 1].any())


if __name__ == "__main__":
    unittest.main()
