from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ComP.per.manifest_utils import audit_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for manifest in args.manifest:
        report = audit_manifest(manifest)
        reports.append(report)
        print(json.dumps(report, indent=2, sort_keys=True))
    (output / "mask_audit.json").write_text(
        json.dumps(reports, indent=2, sort_keys=True), encoding="utf-8"
    )
    fields = [
        "manifest", "sha256", "sample_count", "utterance_count",
        "mean_observed_modalities", "mean_missing_modalities",
        "effective_missing_fraction", "full_visible_rate",
        "one_missing_rate", "two_missing_rate",
    ]
    with (output / "mask_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
