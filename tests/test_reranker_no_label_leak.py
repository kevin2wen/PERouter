import inspect
import unittest

from ComP.per.evidence_reranker import EvidenceReranker


class RerankerNoLabelLeakTest(unittest.TestCase):
    def test_forward_signature_excludes_labels_utility_teacher(self):
        parameters = set(inspect.signature(EvidenceReranker.forward).parameters)
        forbidden = {"labels", "targets", "utility", "teacher_utility", "true_action"}
        self.assertFalse(parameters & forbidden)
