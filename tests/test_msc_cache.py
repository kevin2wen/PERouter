from __future__ import annotations

import unittest

import torch

from ComP.knowledge.msc.cache import build_state_cache, derive_state, query_id


def fake_cache():
    return {
        "split": "train",
        "cache_key": "k",
        "cache_sha256": "sha",
        "records": [
            {"sample_id": "s0", "utterance_id": "u0", "target_modality": 0},
            {"sample_id": "s0", "utterance_id": "u1", "target_modality": 2},
        ],
        "missing_patterns": ["010", "100"],
        "target_modalities": torch.tensor([0, 2]),
        "candidate_ids": [["a"] * 5, ["b"] * 5],
        "candidate_mask": torch.ones(2, 5, dtype=torch.bool),
        "candidate_vectors": torch.randn(2, 5, 4),
        "prompt_vectors": torch.randn(2, 4),
        "source_confidence": torch.rand(2, 5),
        "conflict_score": torch.rand(2, 5),
        "utility": torch.randn(2, 5),
        "skip_utility": torch.zeros(2),
        "target_action": torch.tensor([1, 2]),
        "target_labels": torch.ones(2),
    }


class MSCCacheTest(unittest.TestCase):
    def test_derive_state(self):
        state = derive_state(fake_cache())
        self.assertEqual(tuple(state["observed_mask"].shape), (2, 3))
        self.assertAlmostEqual(float(state["missing_rate"][0]), 1 / 3)
        self.assertEqual(int(state["missing_pattern"][0]), int("010", 2))

    def test_build_state_cache_roundtrip(self):
        payload = build_state_cache(fake_cache())
        self.assertEqual(payload["msc_format_version"], 1)
        self.assertEqual(len(payload["query_ids"]), 2)
        self.assertEqual(
            payload["query_ids"][0], query_id("s0", "u0", "0")
        )


if __name__ == "__main__":
    unittest.main()
