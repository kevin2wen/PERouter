from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ComP.per.evidence_lineage import aggregate_relations, classify_relation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-cache-root", type=Path, required=True)
    parser.add_argument("--clean-cache-root", type=Path, required=True)
    parser.add_argument("--validation-trace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for bank, root in (("original", args.original_cache_root), ("clean", args.clean_cache_root)):
        for rate in ("0.5", "0.7"):
            cache = torch.load(root / f"mr{rate}" / "candidate_action_cache.pt", map_location="cpu", weights_only=False)
            by_target = {}
            for record in cache["records"]:
                target = record["target_modality"]
                by_target.setdefault(target, [])
                for candidate_id in record["candidate_ids"]:
                    if not candidate_id:
                        continue
                    by_target[target].append(classify_relation(
                        record["sample_id"], record["utterance_id"], candidate_id
                    ))
            for target, relations in sorted(by_target.items()):
                rows.append({"origin": "train", "bank": bank, "rate": rate, "target_modality": target, **aggregate_relations(relations)})

    for rate in ("0.5", "0.7"):
        path = args.validation_trace_root / f"mr{rate}" / "feature" / "evidence_trace.csv"
        by_target = {}
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("evidence_available", "")).lower() not in {"true", "1"}:
                    continue
                candidate_id = f"{row['neighbor_sample_id']}:{row['neighbor_utterance_id']}"
                by_target.setdefault(row["target_modality"], []).append(classify_relation(
                    row["sample_id"], row["utterance_id"], candidate_id
                ))
        for target, relations in sorted(by_target.items()):
            rows.append({"origin": "validation", "bank": "validation", "rate": rate, "target_modality": target, **aggregate_relations(relations)})

    with (args.output_dir / "evidence_lineage_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "rows": rows,
        "high_risk": any(
            row["origin"] == "train" and row["same_video_rate"] > 0.1 for row in rows
        ),
        "validation_same_video_max": max(
            (row["same_video_rate"] for row in rows if row["origin"] == "validation"),
            default=0.0,
        ),
        "speaker_metadata_available": False,
    }
    (args.output_dir / "evidence_lineage_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
