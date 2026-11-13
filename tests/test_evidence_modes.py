from __future__ import annotations

import unittest

import torch

from ComP.per.evidence import (
    FeatureNeighborRetriever,
    randomize_evidence,
    shuffled_evidence,
    source_only_evidence,
)


def make_retriever():
    retriever = FeatureNeighborRetriever(
        {"audio": 3, "text": 4, "video": 5}, top_k=2
    )
    for sample in range(4):
        features = {
            "audio": torch.randn(3, 3),
            "text": torch.randn(3, 4),
            "video": torch.randn(3, 5),
        }
        retriever.add(
            f"s{sample}",
            [f"u{sample}_{i}" for i in range(3)],
            features,
            torch.ones(3, dtype=torch.bool),
        )
    return retriever


class EvidenceModeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.retriever = make_retriever()
        features = {
            "audio": torch.randn(1, 3, 3),
            "text": torch.randn(1, 3, 4),
            "video": torch.randn(1, 3, 5),
        }
        availability = torch.ones(1, 3, 3, dtype=torch.bool)
        availability[:, :, 1] = False
        self.reference = self.retriever.retrieve(
            features,
            availability,
            torch.ones(1, 3, dtype=torch.bool),
            ["query"],
            [["q0", "q1", "q2"]],
        )

    def test_random_is_deterministic_and_preserves_coverage(self):
        first = randomize_evidence(self.reference, self.retriever, 123)
        second = randomize_evidence(self.reference, self.retriever, 123)
        torch.testing.assert_close(first.features, second.features)
        torch.testing.assert_close(first.mask, self.reference.mask)
        torch.testing.assert_close(
            first.source_confidence, self.reference.source_confidence
        )

    def test_shuffled_preserves_shape_and_total_coverage(self):
        result = shuffled_evidence(self.reference, 123)
        self.assertEqual(result.features.shape, self.reference.features.shape)
        self.assertEqual(result.mask.shape, self.reference.mask.shape)
        self.assertEqual(int(result.mask.sum()), int(self.reference.mask.sum()))

    def test_random_and_shuffled_controls_exclude_query_sample(self):
        batch_size = self.reference.features.size(0)
        query_ids = ["s0"] * batch_size
        randomized = randomize_evidence(
            self.reference,
            self.retriever,
            seed=123,
            query_sample_ids=query_ids,
        )
        shuffled = shuffled_evidence(
            self.reference,
            seed=123,
            query_sample_ids=query_ids,
        )
        for evidence in (randomized, shuffled):
            for batch_index in range(batch_size):
                for time_index in range(evidence.mask.size(1)):
                    for rank in range(evidence.mask.size(2)):
                        if evidence.mask[batch_index, time_index, rank]:
                            self.assertFalse(
                                evidence.ids[batch_index][time_index][rank].startswith(
                                    "s0:"
                                )
                            )

    def test_source_only_zeros_features_but_keeps_source_metadata(self):
        result = source_only_evidence(self.reference)
        self.assertTrue((result.features == 0).all())
        torch.testing.assert_close(result.mask, self.reference.mask)
        torch.testing.assert_close(
            result.source_confidence, self.reference.source_confidence
        )


if __name__ == "__main__":
    unittest.main()
