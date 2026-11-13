from __future__ import annotations

import unittest

import torch

from ComP.per.teacher import CompleteViewTeacher
from tests.helpers import text_availability, tiny_features, tiny_model


class TeacherCandidateApplicationTest(unittest.TestCase):
    def test_candidates_are_applied_at_missing_target_modality(self):
        student = tiny_model(use_router=True, evidence_dim=1).eval()
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
            evidence = {
                name: value + torch.randn_like(value) * 5.0
                for name, value in output.prompt_candidates.items()
            }
            scores = teacher.score_candidates(
                features,
                availability,
                output.prompt_candidates,
                evidence,
                labels,
            )

        utility_gap = scores.utility[:, :, 1, 1] - scores.utility[:, :, 1, 0]
        missing = ~availability[:, :, 1]
        observed = availability[:, :, 1]
        self.assertTrue((utility_gap[missing].abs() > 1e-8).any())
        self.assertTrue(torch.allclose(
            utility_gap[observed],
            torch.zeros_like(utility_gap[observed]),
        ))


if __name__ == "__main__":
    unittest.main()
