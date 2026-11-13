from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.knowledge.statistics import bootstrap_family


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=66)
    args = parser.parse_args()
    path = Path(args.predictions)
    output_dir = Path(args.output_dir or path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in csv.DictReader(path.open(newline="", encoding="utf-8")):
        row["abs_error"] = float(row["abs_error"])
        row["label"] = float(row["label"])
        row["prediction"] = float(row["prediction"])
        rows.append(row)
    results = []
    for family in ("A", "B", "C"):
        results.extend(
            bootstrap_family(
                rows,
                family,
                n_boot=args.n_boot,
                n_perm=args.n_perm,
                seed=args.seed,
            )
        )
    with (output_dir / "bootstrap_ci.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = (
            "family",
            "comparison",
            "baseline",
            "metric",
            "clusters",
            "pairs",
            "delta",
            "ci_low",
            "ci_high",
            "boot_mean",
            "boot_std",
            "p_perm",
            "p_holm",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    with (output_dir / "permutation_tests.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "family",
                "comparison",
                "baseline",
                "p_perm",
                "p_holm",
                "delta",
            ),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(results)
    family_a = [row for row in results if row["family"] == "A"]
    family_b = [row for row in results if row["family"] == "B"]
    family_c = [row for row in results if row["family"] == "C"]
    content_ok = bool(family_a) and all(
        row["ci_high"] < 0.0 for row in family_a
    )
    oracle_ok = bool(family_c) and all(
        row["ci_high"] < 0.0 for row in family_c
    )
    filter_ok = bool(family_b) and all(
        row["ci_high"] < 0.0 for row in family_b
    ) if family_b else False
    if not oracle_ok:
        reason = "ORACLE_CI_NOT_NEGATIVE"
    elif not content_ok:
        reason = "CARD_CONTROL_CI_CROSSES_ZERO"
    else:
        reason = "PASS"
    decision = {
        "passed": content_ok and oracle_ok,
        "reason": reason,
        "oracle_ok": oracle_ok,
        "content_ok": content_ok,
        "filter_stat_ok": filter_ok,
        "family_a": family_a,
        "family_b": family_b,
        "family_c": family_c,
        "n_boot": args.n_boot,
        "n_perm": args.n_perm,
        "alpha": 0.05,
        "cluster_key": "dialogue_id (=sample_id for CMU-MOSI)",
    }
    (output_dir / "ek3_stat_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
