import unittest

from ComP.per.evidence_lineage import aggregate_relations, classify_relation


class EvidenceLineageTest(unittest.TestCase):
    def test_relation_classes(self):
        same = classify_relation("v1", "v1_003", "v1:v1_004")
        self.assertTrue(same.same_video)
        self.assertTrue(same.adjacent_utterance)
        self.assertFalse(same.same_utterance)
        cross = classify_relation("v1", "v1_003", "v2:v2_010")
        self.assertTrue(cross.cross_video)

    def test_aggregate(self):
        values = [
            classify_relation("v1", "v1_001", "v1:v1_002"),
            classify_relation("v2", "v2_001", "v2:v2_001"),
        ]
        result = aggregate_relations(values)
        self.assertEqual(result["candidates"], 2)
        self.assertEqual(result["same_video_rate"], 1.0)
        self.assertEqual(result["same_utterance_rate"], 0.5)
