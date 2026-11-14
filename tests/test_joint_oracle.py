from __future__ import annotations

import unittest

from ComP.knowledge.msc.joint_action import (
    joint_action_space,
    joint_interaction_error,
    local_sum,
)


class JointActionTest(unittest.TestCase):
    def test_action_space(self):
        space = joint_action_space([[None, 0, 1, 2, 3, 4], [None, 0, 1, 2, 3, 4]])
        self.assertEqual(len(space), 36)
        self.assertIn((None, None), space)

    def test_utility_formula(self):
        self.assertAlmostEqual(local_sum([0.2, -0.1]), 0.1)
        self.assertAlmostEqual(joint_interaction_error(0.05, 0.1), -0.05)


if __name__ == "__main__":
    unittest.main()
