from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(Path(args.bootstrap_csv).open(newline="")))
    lines = [
        "# CI endpoint audit",
        "",
        "Rule: a 95% CI containing 0, endpoints included "
        "(`ci_low <= 0 <= ci_high`), is NOT significant.",
        "",
        "| comparison | baseline | metric | delta | ci_low | ci_high | significant |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    significant = []
    for row in rows:
        low = float(row["ci_low"])
        high = float(row["ci_high"])
        sig = not (low <= 0.0 <= high)
        lines.append(
            f"| {row['comparison']} | {row['baseline']} | {row['metric']} | "
            f"{float(row['delta']):+.4f} | {low:+.4f} | {high:+.4f} | "
            f"{'YES' if sig else 'NO'} |"
        )
        if sig:
            significant.append(
                {
                    "comparison": row["comparison"],
                    "baseline": row["baseline"],
                    "metric": row["metric"],
                    "delta": float(row["delta"]),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    lines += [
        "",
        f"Significant comparisons: {len(significant)} / {len(rows)}",
        "",
    ]
    (output / "ci_rule_audit.md").write_text("\n".join(lines))
    with (output / "significance_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "comparison", "baseline", "metric", "delta",
                "ci_low", "ci_high", "significant",
            ),
        )
        writer.writeheader()
        for row in rows:
            low = float(row["ci_low"])
            high = float(row["ci_high"])
            writer.writerow(
                {
                    "comparison": row["comparison"],
                    "baseline": row["baseline"],
                    "metric": row["metric"],
                    "delta": row["delta"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "significant": not (low <= 0.0 <= high),
                }
            )
    print("\n".join(lines[-4:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
