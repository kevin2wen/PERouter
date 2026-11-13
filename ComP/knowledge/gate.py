from __future__ import annotations

from typing import Mapping


def _score(metrics: Mapping[str, float], lower_is_better: bool) -> float:
    key = "mae" if lower_is_better else "acc"
    if key not in metrics:
        raise KeyError(f"missing metric {key}")
    return -float(metrics[key]) if lower_is_better else float(metrics[key])


def evaluate_ek3(
    results: Mapping[str, Mapping[str, float]],
    *,
    lower_is_better: bool = True,
    min_gain: float = 0.0,
) -> dict:
    required = {"empty", "card", "random_card", "shuffled_card", "source_only"}
    missing = required - set(results)
    if missing:
        raise KeyError(f"missing EK3 modes: {sorted(missing)}")
    scores = {
        mode: _score(metrics, lower_is_better)
        for mode, metrics in results.items()
    }
    oracle_ok = True
    oracle_gain = None
    if "oracle_card" in scores:
        oracle_gain = scores["oracle_card"] - scores["empty"]
        oracle_ok = oracle_gain > min_gain
    content_ok = (
        scores["card"] > scores["random_card"] + min_gain
        and scores["card"] > scores["shuffled_card"] + min_gain
    )
    source_only_ok = scores["card"] > scores["source_only"] + min_gain
    raw_ok = True
    if "raw_feature" in scores:
        raw_ok = scores["card"] > scores["raw_feature"] + min_gain
    passed = oracle_ok and content_ok and source_only_ok and raw_ok
    if not oracle_ok:
        reason = "ORACLE_CARD_NO_GAIN"
    elif not content_ok:
        reason = "CARD_NOT_BETTER_THAN_RANDOM_OR_SHUFFLED"
    elif not source_only_ok:
        reason = "SOURCE_ONLY_EXPLAINS_GAIN"
    elif not raw_ok:
        reason = "CARD_NOT_BETTER_THAN_RAW_CONTEXT"
    else:
        reason = "PASS"
    return {
        "passed": passed,
        "reason": reason,
        "scores": scores,
        "oracle_gain": oracle_gain,
        "oracle_ok": oracle_ok,
        "content_ok": content_ok,
        "source_only_ok": source_only_ok,
        "raw_context_ok": raw_ok,
        "min_gain": min_gain,
        "lower_is_better": lower_is_better,
    }


def evaluate_ek4(
    results: Mapping[str, Mapping[str, float]],
    *,
    lower_is_better: bool = True,
    min_gain: float = 0.0,
) -> dict:
    required = {"learned_filter", "confidence", "random_action", "card"}
    missing = required - set(results)
    if missing:
        raise KeyError(f"missing EK4 modes: {sorted(missing)}")
    scores = {
        mode: _score(metrics, lower_is_better)
        for mode, metrics in results.items()
    }
    checks = {
        "beats_confidence": scores["learned_filter"]
        > scores["confidence"] + min_gain,
        "beats_random_action": scores["learned_filter"]
        > scores["random_action"] + min_gain,
        "reference_card_delta": scores["learned_filter"] - scores["card"],
    }
    passed = checks["beats_confidence"] and checks["beats_random_action"]
    return {
        "passed": passed,
        "checks": checks,
        "scores": scores,
        "reason": "PASS" if passed else "LEARNED_FILTER_FAILED",
    }
