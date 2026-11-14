from __future__ import annotations

import unittest

import torch

from ComP.knowledge.msc.state import MSCStateEncoder


class MSCStateTest(unittest.TestCase):
    def test_state_encoder_shape_and_bounds(self):
        encoder = MSCStateEncoder(hidden_dim=64, pattern_vocab_size=16)
        availability = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        target = torch.nn.functional.one_hot(
            torch.tensor([1, 0]), num_classes=3
        ).float()
        pattern = torch.tensor([1, 4], dtype=torch.long)
        rate = torch.tensor([0.3333, 0.6667])
        out = encoder(availability, target, pattern, rate)
        self.assertEqual(tuple(out.shape), (2, 64))
        self.assertTrue(torch.isfinite(out).all())

    def test_state_encoder_rejects_wrong_modality_count(self):
        with self.assertRaises(ValueError):
            MSCStateEncoder(modality_count=2)


if __name__ == "__main__":
    unittest.main()
