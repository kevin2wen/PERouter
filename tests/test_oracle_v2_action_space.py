import unittest
import torch

from ComP.per.evidence_oracle_v2 import (
    OracleAction,
    action_grid,
    action_histogram,
    normalize_action,
    select_actions,
    validate_oracle_v2_split,
)


class OracleV2ActionSpaceTest(unittest.TestCase):
    def test_skip_is_gamma_zero(self):
        self.assertEqual(
            normalize_action(OracleAction(True, 2, 0.0)),
            OracleAction(False, -1, 0.0),
        )

    def test_oracle_v2_rejects_test(self):
        with self.assertRaises(PermissionError):
            validate_oracle_v2_split("test")
        validate_oracle_v2_split("validation")

    def test_action_grid_is_unique(self):
        actions = action_grid(3)
        self.assertEqual(len(actions), 1 + 3 * 5)
        self.assertEqual(len(set(actions)), len(actions))

    def test_action_selection_excludes_inactive(self):
        scores = torch.zeros(1, 1, 3, 3)
        scores[0, 0, 0, 1] = 0.2
        scores[0, 0, 1, 2] = 0.8
        active = torch.tensor([[[True, False, False]]])
        selected = select_actions(scores, active, 3)
        self.assertEqual(selected.tolist(), [[[1, 0, 0]]])

    def test_actions_can_differ_by_target_modality(self):
        scores = torch.zeros(1, 1, 3, 3)
        scores[0, 0, 0, 1] = 1.0
        scores[0, 0, 1, 2] = 1.0
        active = torch.tensor([[[True, True, False]]])
        selected = select_actions(scores, active, 3)
        self.assertEqual(selected.tolist(), [[[1, 2, 0]]])

    def test_action_histogram_marks_skip(self):
        actions = [OracleAction(False, -1, 0.0), OracleAction(True, 0, 0.5)]
        selected = torch.tensor([[[0, 1, 0]]])
        active = torch.tensor([[[True, True, False]]])
        histogram = action_histogram(selected, active, actions)
        self.assertEqual(histogram["skip_evidence"], 1)
        self.assertEqual(histogram["rank0_gamma0.5"], 1)
