from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import torch

from ComP.per.missing import MaskManifest, MissingPatternSampler

COMP_DIR = Path(__file__).resolve().parents[1] / "ComP"
sys.path.insert(0, str(COMP_DIR))
from train_mr import random_mask
sys.path.pop(0)


class MissingProtocolTest(unittest.TestCase):
    def test_legacy_degeneracy_at_and_above_two_thirds(self):
        for missing_rate in (2.0 / 3.0, 0.7, 1.0):
            np.random.seed(123)
            matrix = random_mask(3, 97, missing_rate)
            self.assertTrue(np.all(matrix.sum(axis=1) == 1))
            self.assertTrue(set(np.unique(matrix)).issubset({0, 1}))

    def test_sampler_can_reproduce_legacy_function_exactly(self):
        np.random.seed(991)
        expected = random_mask(3, 60, 0.7)
        np.random.seed(991)
        sampler = MissingPatternSampler(
            seed=17, missing_rate=0.7, protocol="legacy"
        )
        sequence_mask = torch.ones(4, 15, dtype=torch.bool)
        actual = sampler.sample(
            ["a", "b", "c", "d"], sequence_mask, epoch=3
        )
        actual = actual.reshape(4, 15, 3).reshape(-1, 3).numpy().astype(np.int32)
        np.testing.assert_array_equal(actual, expected)

    def test_fixed_pattern_is_separate_and_padding_is_unavailable(self):
        sampler = MissingPatternSampler(seed=1, missing_rate=0.7)
        sequence_mask = torch.tensor(
            [[True, True, False], [True, False, False]]
        )
        availability = sampler.fixed(["x", "y"], (1, 1, 0), sequence_mask)
        self.assertEqual(tuple(availability.shape), (2, 3, 3))
        self.assertTrue(availability[0, 0].tolist() == [True, True, False])
        self.assertFalse(availability[0, 2].any())
        self.assertFalse(availability[1, 1:].any())

    def test_categorical_and_bernoulli_never_make_all_missing_rows(self):
        sequence_mask = torch.ones(3, 8, dtype=torch.bool)
        for protocol in ("categorical", "bernoulli"):
            sampler = MissingPatternSampler(
                seed=4, missing_rate=0.7, protocol=protocol
            )
            availability = sampler.sample(["a", "b", "c"], sequence_mask, epoch=0)
            self.assertTrue(availability[sequence_mask].any(dim=-1).all())

    def test_manifest_contains_and_recovers_fixed_test_mask(self):
        sequence_mask = torch.tensor(
            [[True, True, False], [True, True, False]]
        )
        availability = torch.tensor(
            [
                [[True, False, False], [False, True, False], [False, False, False]],
                [[False, False, True], [True, True, False], [False, False, False]],
            ],
            dtype=torch.bool,
        )
        utterance_ids = [["u0", "u1", ""], ["u2", "u3", ""]]
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/mask.csv"
            MaskManifest.save(
                path,
                ["sample0", "sample1"],
                utterance_ids,
                availability,
                sequence_mask,
                seed=66,
                mask_rate=0.7,
                sampler_version="categorical-v1",
            )
            manifest = MaskManifest.load(path)
            recovered = MaskManifest.to_availability(
                manifest, ["sample0", "sample1"], sequence_mask
            )
            torch.testing.assert_close(recovered, availability)
            with open(path, encoding="utf-8") as handle:
                header = handle.readline().strip().split(",")
            self.assertEqual(header[:5], [
                "seed", "sample_id", "utterance_index", "utterance_id", "availability"
            ])


if __name__ == "__main__":
    unittest.main()
