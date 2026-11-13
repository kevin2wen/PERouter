import unittest

from ComP.per.e175_gate import evaluate_strategy, gain_capture


BASE = {"acc": 0.70, "f1": 0.69, "mae": 1.20, "corr": 0.50}


def shifted(**changes):
    value = dict(BASE)
    value.update(changes)
    return value


class E175GateTest(unittest.TestCase):
    def test_gain_capture(self):
        self.assertAlmostEqual(gain_capture(0.75, 0.70, 0.90), 0.25)

    def test_rejects_learned_below_empty(self):
        learned = shifted(acc=0.68, f1=0.67)
        result = evaluate_strategy(learned, BASE, shifted(acc=0.65), shifted(acc=0.69), shifted(acc=0.67), shifted(acc=0.66), shifted(acc=0.90))
        self.assertFalse(result["pass"])

    def test_accepts_clear_improvement(self):
        learned = shifted(acc=0.78, f1=0.77, mae=1.18, corr=0.52)
        result = evaluate_strategy(learned, BASE, shifted(acc=0.69), shifted(acc=0.70), shifted(acc=0.70), shifted(acc=0.71), shifted(acc=0.90))
        self.assertTrue(result["pass"])
