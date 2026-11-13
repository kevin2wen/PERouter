from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_teacher_oracle.py"
    spec = importlib.util.spec_from_file_location("eval_teacher_oracle", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EvidenceAblationCliTest(unittest.TestCase):
    def test_source_contract_for_modes_gamma_and_fixed_override(self):
        module = load_module()
        source = Path(module.__file__).read_text(encoding="utf-8")
        self.assertIn('"empty", "feature", "random", "shuffled", "source_only"', source)
        self.assertIn('"--fixed-source-weight"', source)
        self.assertIn("cli.fixed_source_weight", source)
        self.assertIn("override = torch.full_like(", source)
        fixed_block = source.split('cli.route_override == "fixed"', 1)[1][:250]
        self.assertNotIn("student_output.route, 0.5", fixed_block.replace("\n", " "))


if __name__ == "__main__":
    unittest.main()
