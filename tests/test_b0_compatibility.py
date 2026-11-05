from __future__ import annotations

import subprocess
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from ComP.model import ComP
from tests.helpers import text_availability, tiny_features, tiny_model


class B0CompatibilityTest(unittest.TestCase):
    def test_per_student_off_is_numerically_equal_to_comP_backbone(self):
        student = tiny_model(use_router=False, evidence_dim=1).eval()
        args = SimpleNamespace(
            hidden=8,
            n_classes=1,
            depth=2,
            num_heads=2,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            no_cuda=True,
            lbd=0.3,
            seq_len=4,
            device=torch.device("cpu"),
        )
        backbone = ComP(
            args,
            5,
            6,
            7,
            8,
            1,
            depth=2,
            num_heads=2,
            lbd=0.3,
        ).eval()
        backbone.load_state_dict(student.backbone.state_dict(), strict=True)
        features = tiny_features()
        availability = text_availability()
        packed = student._pack_features(features, availability)
        with torch.no_grad():
            student_output = student(features, availability, None)
            direct = backbone(
                packed,
                torch.zeros(2, 4),
                availability.permute(1, 0, 2),
                availability.any(dim=-1).to(torch.float32),
                False,
            )
        torch.testing.assert_close(student_output.logits, direct[1])
        self.assertEqual(
            sum(parameter.numel() for parameter in student.parameters()),
            sum(parameter.numel() for parameter in backbone.parameters()),
        )

    def test_default_forward_matches_original_commit_numerically(self):
        repo = Path(__file__).resolve().parents[1]
        try:
            source = subprocess.check_output(
                ["git", "show", "28192d3:ComP/model.py"],
                cwd=repo,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("original Git object is unavailable")
        module = types.ModuleType("original_comp_model")
        old_path = list(sys.path)
        sys.path.insert(0, str(repo / "ComP"))
        try:
            exec(compile(source, "original_ComP/model.py", "exec"), module.__dict__)
        finally:
            sys.path[:] = old_path

        args = SimpleNamespace(
            hidden=8,
            drop_rate=0.0,
            seq_len=4,
            device=torch.device("cpu"),
        )
        torch.manual_seed(9)
        original = module.ComP(args, 5, 6, 7, 8, 1, depth=2, num_heads=2)
        current = ComP(args, 5, 6, 7, 8, 1, depth=2, num_heads=2)
        current.load_state_dict(original.state_dict(), strict=True)
        original.eval()
        current.eval()
        inputs = torch.randn(4, 2, 18)
        labels = torch.zeros(2, 4)
        mask = torch.ones(4, 2, 3)
        mask[:, :, 1] = 0
        umask = torch.ones(2, 4)
        with torch.no_grad():
            expected = original(inputs, labels, mask, umask, first_stage=False)
            actual = current(inputs, labels, mask, umask, first_stage=False)
        self.assertEqual(len(actual), 11)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
        for index in range(2, 6):
            torch.testing.assert_close(actual[index], expected[index])


if __name__ == "__main__":
    unittest.main()
