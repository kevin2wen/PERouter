from __future__ import annotations

import unittest

from ComP.per.teacher_diagnostics import validate_route_selection


class TeacherOracleGuardTest(unittest.TestCase):
    def test_oracle_is_validation_only(self):
        validate_route_selection("validation", "oracle")
        with self.assertRaises((ValueError, PermissionError)):
            validate_route_selection("test", "oracle")

    def test_unknown_override_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_route_selection("validation", "unknown")


if __name__ == "__main__":
    unittest.main()
