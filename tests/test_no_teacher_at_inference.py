from __future__ import annotations

import unittest
from unittest import mock
from pathlib import Path

import torch

from ComP.per.teacher import CompleteViewTeacher
from tests.helpers import text_availability, tiny_features, tiny_model


class NoTeacherAtInferenceTest(unittest.TestCase):
    def test_inference_path_never_calls_teacher(self):
        model = tiny_model(use_router=False).eval()
        self.assertFalse(any(isinstance(module, CompleteViewTeacher) for module in model.modules()))
        with mock.patch.object(
            CompleteViewTeacher,
            "score_candidates",
            side_effect=AssertionError("teacher must not be called"),
        ):
            with torch.no_grad():
                output = model(tiny_features(), text_availability(), evidence=None)
        self.assertTrue(torch.isfinite(output.logits).all())

    def test_eval_per_artifact_labels_teacher_free(self):
        # The executable evaluator writes this explicit field; verifying the
        # field here prevents silently changing the deployment claim.
        source = Path(
            str(__import__("pathlib").Path(__file__).resolve().parents[1] / "ComP" / "eval_per.py")
        ).read_text(encoding="utf-8")
        self.assertIn('"teacher_used_at_inference": False', source)


if __name__ == "__main__":
    unittest.main()
