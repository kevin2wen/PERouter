import torch
import unittest

from ComP.per.evidence import FeatureNeighborRetriever


class CleanBankExclusionTest(unittest.TestCase):
    def make_features(self):
        return {
            "audio": torch.tensor([[1.0, 0.0]]),
            "text": torch.tensor([[2.0, 0.0]]),
            "video": torch.tensor([[3.0, 0.0]]),
        }

    def test_same_video_is_excluded(self):
        retriever = FeatureNeighborRetriever(
            {"audio": 2, "text": 2, "video": 2},
            2,
            exclude_same_video=True,
            exclude_same_dialogue=True,
        )
        retriever.add("v1", ["u0"], self.make_features(), torch.tensor([True]))
        retriever.add("v1", ["u1"], self.make_features(), torch.tensor([True]))
        retriever.add("v2", ["u0"], self.make_features(), torch.tensor([True]))
        evidence = retriever.retrieve(
            {name: value.unsqueeze(0) for name, value in self.make_features().items()},
            torch.tensor([[[True, False, True]]]),
            torch.tensor([[True]]),
            ["v1"],
            [["u0"]],
        )
        self.assertTrue(evidence.mask[0, 0].any())
        self.assertTrue(all(value.startswith("v2:") for value in evidence.ids[0][0] if value))
        self.assertTrue(all(not value.startswith("v1:") for value in evidence.ids[0][0] if value))
