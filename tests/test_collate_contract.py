from __future__ import annotations

import unittest

import torch

from ComP.dataloader_cmumosi import CMUMOSIDataset
from ComP.per.batch import SampleRecord, collate_per, move_batch_to_device


def _record(sample_id: str, length: int):
    return SampleRecord(
        features={
            "audio": torch.arange(length * 5, dtype=torch.float32).reshape(length, 5),
            "text": torch.arange(length * 6, dtype=torch.float32).reshape(length, 6),
            "video": torch.arange(length * 7, dtype=torch.float32).reshape(length, 7),
        },
        label=torch.arange(length, dtype=torch.float32),
        sample_id=sample_id,
        utterance_ids=[f"{sample_id}-u{index}" for index in range(length)],
        sequence_mask=torch.ones(length, dtype=torch.bool),
        availability=torch.ones(length, 3, dtype=torch.bool),
        raw_texts=[f"text-{index}" for index in range(length)],
    )


class CollateContractTest(unittest.TestCase):
    def test_per_fields_are_added_without_changing_feature_values(self):
        records = [_record("a", 3), _record("b", 1)]
        batch = collate_per(records, {"sequence": 5}, evidence_batch_size=2, training=True)
        self.assertEqual(batch.features["audio"].shape, (2, 5, 5))
        self.assertEqual(batch.features["text"].shape, (2, 5, 6))
        self.assertEqual(batch.features["video"].shape, (2, 5, 7))
        self.assertEqual(batch.labels.shape, (2, 5))
        self.assertEqual(batch.sequence_mask.shape, (2, 5))
        self.assertEqual(batch.availability.shape, (2, 5, 3))
        self.assertEqual(batch.sample_ids, ["a", "b"])
        self.assertEqual(batch.utterance_ids[0], ["a-u0", "a-u1", "a-u2", "", ""])
        torch.testing.assert_close(batch.features["audio"][0, :3], records[0].features["audio"])
        self.assertTrue((batch.features["text"][1, 1:] == 0).all())
        self.assertFalse(batch.sequence_mask[1, 1:].any())
        self.assertFalse(batch.availability[1, 1:].any())

    def test_per_batch_moves_without_losing_string_ids(self):
        batch = collate_per([_record("a", 2)], {"sequence": 2}, 1, False)
        moved = move_batch_to_device(batch, torch.device("cpu"))
        self.assertEqual(moved.sample_ids, ["a"])
        self.assertEqual(moved.utterance_ids, [["a-u0", "a-u1"]])

    def test_original_comP_collate_contract_remains_ten_items(self):
        data = [
            (
                torch.randn(2, 5),
                torch.randn(2, 6),
                torch.randn(2, 7),
                torch.randn(2, 5),
                torch.randn(2, 6),
                torch.randn(2, 7),
                torch.tensor([0.0, 1.0]),
                torch.ones(2),
                torch.tensor([1.0, -1.0]),
                "video-id",
            )
        ]
        # collate_fn is pure; B0 still receives exactly ten list entries and the
        # conversation ID remains the final item used by train_mr.py.
        legacy = CMUMOSIDataset.collate_fn(None, data)
        self.assertEqual(len(legacy), 10)
        self.assertEqual(legacy[0].shape, (2, 1, 5))
        self.assertEqual(legacy[8].shape, (1, 2))
        self.assertEqual(legacy[-1], ["video-id"])


if __name__ == "__main__":
    unittest.main()
