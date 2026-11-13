import unittest

import torch

from ComP.per.evidence import randomize_evidence
from ComP.per.text_evidence import TextNeighborRetriever


def features(words):
    return {
        "audio": torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])[:len(words)],
        "text": torch.tensor([[4.0, 0.0], [5.0, 0.0], [6.0, 0.0]])[:len(words)],
        "video": torch.tensor([[7.0, 0.0], [8.0, 0.0], [9.0, 0.0]])[:len(words)],
    }


class TextEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.retriever = TextNeighborRetriever(
            {"audio": 2, "text": 2, "video": 2}, top_k=2, min_df=1
        )
        self.retriever.add(
            "train-a", ["u0", "u1"], features(["x", "x"]),
            torch.tensor([True, True]), ["good movie", "bad movie"]
        )
        self.retriever.add(
            "train-b", ["u0"], features(["x"]),
            torch.tensor([True]), ["excellent movie"]
        )
        self.retriever.fit()

    def test_text_evidence_retrieves_when_text_is_observed(self):
        availability = torch.tensor([[
            [True, True, True],
            [False, True, False],
            [True, False, True],
        ]])
        evidence = self.retriever.retrieve(
            features(["q0", "q1", "q2"]),
            availability,
            torch.tensor([[True, True, True]]),
            ["validation-a"],
            [["v0", "v1", "v2"]],
            raw_texts=[["complete", "good movie", "secret missing text"]],
        )
        self.assertTrue(evidence.mask[0, 0].sum().item() == 0)
        self.assertTrue(evidence.mask[0, 1].any())
        self.assertFalse(evidence.mask[0, 2].any())
        self.assertEqual(evidence.features.shape[:3], (1, 3, 2))
        randomized = randomize_evidence(evidence, self.retriever, seed=66)
        self.assertEqual(randomized.features.shape, evidence.features.shape)
        self.assertTrue(torch.equal(randomized.mask, evidence.mask))

    def test_text_bank_fingerprint_is_stable(self):
        self.assertEqual(self.retriever.source_fingerprint(), self.retriever.source_fingerprint())
        self.assertEqual(len(self.retriever.source_fingerprint()), 64)

    def test_clean_text_retriever_excludes_same_video(self):
        retriever = TextNeighborRetriever(
            {"audio": 2, "text": 2, "video": 2},
            top_k=2,
            min_df=1,
            exclude_same_video=True,
            exclude_same_dialogue=True,
        )
        retriever.add(
            "validation-a",
            ["v0"],
            features(["q0"]),
            torch.tensor([True]),
            ["good movie"],
        )
        retriever.add(
            "train-clean",
            ["u0"],
            features(["c0"]),
            torch.tensor([True]),
            ["good film"],
        )
        retriever.fit()
        evidence = retriever.retrieve(
            features(["q0"]),
            torch.tensor([[[False, True, True]]]),
            torch.tensor([[True]]),
            ["validation-a"],
            [["v0"]],
            raw_texts=[["good movie"]],
        )
        self.assertTrue(evidence.mask.any())
        self.assertTrue(
            all(
                not value.startswith("validation-a:")
                for value in evidence.ids[0][0]
                if value
            )
        )
