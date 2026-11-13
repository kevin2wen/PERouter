from __future__ import annotations

from typing import Dict


DEFAULT_THRESHOLDS = {
    "minimum_action_improvement": 0.01,
    "minimum_gain_capture": 0.25,
    "maximum_mae_degradation": 0.02,
    "minimum_source_only_margin": 0.005,
}


def gain_capture(learned: float, empty: float, oracle: float, *, lower_is_better: bool = False) -> float:
    denominator = (empty - oracle) if lower_is_better else (oracle - empty)
    numerator = (empty - learned) if lower_is_better else (learned - empty)
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


def evaluate_strategy(
    learned: Dict[str, float],
    empty: Dict[str, float],
    random_evidence: Dict[str, float],
    source_only: Dict[str, float],
    confidence: Dict[str, float],
    random_action: Dict[str, float],
    oracle_task: Dict[str, float],
    thresholds: Dict[str, float] | None = None,
) -> Dict[str, object]:
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    checks = {
        "better_than_confidence": learned["acc"] > confidence["acc"] or learned["f1"] > confidence["f1"],
        "better_than_random_action": learned["acc"] > random_action["acc"] or learned["f1"] > random_action["f1"],
        "beats_empty_by_1pp": (
            learned["acc"] - empty["acc"] >= t["minimum_action_improvement"]
            or learned["f1"] - empty["f1"] >= t["minimum_action_improvement"]
        ),
        "beats_random_evidence_by_1pp": (
            learned["acc"] - random_evidence["acc"] >= t["minimum_action_improvement"]
            or learned["f1"] - random_evidence["f1"] >= t["minimum_action_improvement"]
        ),
        "not_source_only_explained": (
            learned["acc"] - source_only["acc"] >= t["minimum_source_only_margin"]
            or learned["f1"] - source_only["f1"] >= t["minimum_source_only_margin"]
        ),
        "mae_not_degraded": learned["mae"] - empty["mae"] <= t["maximum_mae_degradation"],
    }
    capture_acc = gain_capture(learned["acc"], empty["acc"], oracle_task["acc"])
    capture_f1 = gain_capture(learned["f1"], empty["f1"], oracle_task["f1"])
    checks["gain_capture_at_least_25pct"] = max(capture_acc, capture_f1) >= t["minimum_gain_capture"]
    checks["corr_not_catastrophic"] = learned["corr"] >= empty["corr"] - 0.05
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "gain_capture_acc": capture_acc,
        "gain_capture_f1": capture_f1,
        "delta_vs_empty": {key: learned[key] - empty[key] for key in ("acc", "f1", "mae", "corr")},
        "delta_vs_confidence": {key: learned[key] - confidence[key] for key in ("acc", "f1", "mae", "corr")},
        "delta_vs_random_action": {key: learned[key] - random_action[key] for key in ("acc", "f1", "mae", "corr")},
        "delta_vs_random_evidence": {key: learned[key] - random_evidence[key] for key in ("acc", "f1", "mae", "corr")},
        "delta_vs_source_only": {key: learned[key] - source_only[key] for key in ("acc", "f1", "mae", "corr")},
    }
