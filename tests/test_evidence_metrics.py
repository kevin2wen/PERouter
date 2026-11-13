import torch
import unittest

from ComP.per.evidence_metrics import grouped_metric_masks, grouped_metrics


class EvidenceMetricsTest(unittest.TestCase):
    def test_pattern_and_target_masks_exclude_padding(self):
        availability = torch.tensor([[
            [True, True, True],
            [False, True, True],
            [False, False, True],
        ]])
        sequence = torch.tensor([[True, True, False]])
        masks = grouped_metric_masks(availability, sequence)
        self.assertEqual(int(masks["pattern_111"].sum()), 1)
        self.assertEqual(int(masks["pattern_011"].sum()), 1)
        self.assertEqual(int(masks["audio_missing"].sum()), 1)
        self.assertEqual(int(masks["text_missing"].sum()), 0)
        self.assertEqual(int(masks["video_missing"].sum()), 0)

    def test_grouped_metrics_uses_group_mask(self):
        labels = torch.tensor([[1.0, 2.0, 3.0]])
        predictions = torch.tensor([[1.0, 3.0, 3.0]])
        sequence = torch.tensor([[True, True, False]])
        availability = torch.tensor([[[True, True, True], [False, True, True], [True, True, True]]])

        def count_metric(labels, predictions, mask, dataset):
            return {"count": int(mask.sum()), "mae": float((labels[mask] - predictions[mask]).abs().mean())}

        result = grouped_metrics(labels, predictions, sequence, availability, "CMUMOSI", count_metric)
        self.assertEqual(result["pattern_111"]["count"], 1)
        self.assertEqual(result["pattern_011"]["count"], 1)
        self.assertAlmostEqual(result["audio_missing"]["mae"], 1.0)
