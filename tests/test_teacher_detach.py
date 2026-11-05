from __future__ import annotations

import unittest
import tempfile

import torch

from ComP.per.teacher import (
    CompleteViewTeacher,
    TeacherTargetCache,
    make_teacher_cache_key,
)
from ComP.per.types import TeacherTargets
from tests.helpers import text_availability, tiny_features, tiny_model


class TeacherDetachTest(unittest.TestCase):
    def test_teacher_outputs_and_parameters_are_detached(self):
        student = tiny_model(use_router=False)
        teacher = CompleteViewTeacher(
            tiny_model(use_router=False).backbone,
            task="regression",
            dim=8,
            teacher_ckpt_hash="teacher-sha",
        )
        features = tiny_features()
        availability = text_availability()
        labels = torch.randn(2, 4)
        with torch.no_grad():
            output = student(features, availability, None)
        scores = teacher.score_candidates(
            features,
            availability,
            output.prompt_candidates,
            output.evidence,
            labels,
        )
        self.assertEqual(tuple(scores.utility.shape), (2, 4, 3, 2))
        self.assertFalse(scores.utility.requires_grad)
        self.assertFalse(scores.prompt_logits.requires_grad)
        self.assertFalse(scores.evidence_logits.requires_grad)
        self.assertTrue(all(not parameter.requires_grad for parameter in teacher.parameters()))

    def test_cache_key_changes_when_required_inputs_change(self):
        base = {
            "dataset_version": "v1",
            "split": "train",
            "sample_id": "s",
            "utterance_id": "u",
            "availability": [True, False, True],
            "student_candidate_ckpt_hash": "student",
            "teacher_ckpt_hash": "teacher",
            "kb_version": "kb",
            "top_k": 5,
            "seed": 66,
        }
        first = make_teacher_cache_key(**base)
        self.assertEqual(len(first), 64)
        for key, value in (
            ("dataset_version", "v2"),
            ("split", "val"),
            ("sample_id", "s2"),
            ("utterance_id", "u2"),
            ("availability", [True, True, False]),
            ("student_candidate_ckpt_hash", "student2"),
            ("teacher_ckpt_hash", "teacher2"),
            ("kb_version", "kb2"),
            ("top_k", 4),
            ("seed", 67),
        ):
            changed = dict(base)
            changed[key] = value
            self.assertNotEqual(first, make_teacher_cache_key(**changed), key)

    def test_cache_round_trip_stores_detached_tensors(self):
        key = "a" * 64
        targets = TeacherTargets(
            teacher_utility=torch.randn(1, 2, 3, 2, requires_grad=True),
            route_target=torch.rand(1, 2, 3, requires_grad=True),
            teacher_logits=torch.randn(1, 2, 1, requires_grad=True),
            teacher_confidence=torch.rand(1, 2, 3, requires_grad=True),
        )
        with tempfile.TemporaryDirectory() as directory:
            cache = TeacherTargetCache(directory)
            cache.put(key, targets)
            restored = cache.get(key)
        self.assertIsNotNone(restored)
        self.assertFalse(restored.teacher_utility.requires_grad)
        self.assertFalse(restored.route_target.requires_grad)


if __name__ == "__main__":
    unittest.main()
