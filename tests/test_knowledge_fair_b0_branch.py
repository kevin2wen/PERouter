from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from ComP.train_knowledge import _build_knowledge_student
from ComP.train_per import _build_student
from tests.helpers import tiny_args


def fair_b0_args():
    base = vars(tiny_args()).copy()
    base.update(
        {
            "per_mode": "off",
            "top_k": 2,
            "dataset": "CMUMOSI",
            "router_hidden": None,
            "router_dropout": 0.0,
            "fix_private_branch": True,
            "fixed_source_weight": 0.5,
        }
    )
    return SimpleNamespace(**base)


class KnowledgeFairB0BranchTest(unittest.TestCase):
    def test_fair_b0_payload_gains_evidence_branch(self):
        args = fair_b0_args()
        device = torch.device("cpu")
        baseline = _build_student(args, device, 5, 6, 7, 18)
        self.assertIsNone(baseline.evidence_encoder)
        payload = {"args": vars(args), "model": baseline.state_dict()}

        rebuilt = _build_knowledge_student(
            args, device, 5, 6, 7, 18, payload
        )
        self.assertIsNotNone(rebuilt.evidence_encoder)
        self.assertIsNotNone(rebuilt.observed_projections)
        for name, tensor in baseline.state_dict().items():
            self.assertTrue(
                torch.equal(tensor, rebuilt.state_dict()[name]),
                msg=f"backbone weight drifted: {name}",
            )

    def test_missing_non_branch_parameter_is_rejected(self):
        args = fair_b0_args()
        device = torch.device("cpu")
        baseline = _build_student(args, device, 5, 6, 7, 18)
        state = dict(baseline.state_dict())
        state.pop(next(iter(state)))
        payload = {"args": vars(args), "model": state}
        with self.assertRaises(RuntimeError):
            _build_knowledge_student(args, device, 5, 6, 7, 18, payload)


if __name__ == "__main__":
    unittest.main()
