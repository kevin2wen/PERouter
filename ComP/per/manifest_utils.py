from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_manifest_path(override: str | None, default_path: str | Path) -> Path:
    path = Path(override if override else default_path).expanduser().resolve()
    if override and not path.is_file():
        raise FileNotFoundError(path)
    return path


def audit_manifest(path: str | Path) -> Dict[str, object]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    counts = Counter()
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "utterance_index", "availability"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"manifest missing fields: {sorted(required)}")
        for row in reader:
            observed = [bool(value) for value in json.loads(row["availability"])]
            if len(observed) != 3 or not any(observed):
                raise ValueError(f"invalid availability row: {row}")
            missing_count = sum(not value for value in observed)
            counts[f"missing_{missing_count}"] += 1
            pattern = "".join("1" if value else "0" for value in observed)
            counts[f"observed_pattern_{pattern}"] += 1
            rows.append(row)
    utterance_count = len(rows)
    sample_ids = {row["sample_id"] for row in rows}
    observed_total = sum(
        sum(bool(value) for value in json.loads(row["availability"]))
        for row in rows
    )
    result: Dict[str, object] = {
        "manifest": str(path),
        "sha256": sha256_file(path),
        "sample_count": len(sample_ids),
        "utterance_count": utterance_count,
        "mean_observed_modalities": observed_total / max(1, utterance_count),
        "mean_missing_modalities": (
            3.0 - observed_total / max(1, utterance_count)
        ),
        "effective_missing_fraction": (
            (3 * utterance_count - observed_total) / max(1, 3 * utterance_count)
        ),
        "full_visible_rate": counts.get("missing_0", 0) / max(1, utterance_count),
        "one_missing_rate": counts.get("missing_1", 0) / max(1, utterance_count),
        "two_missing_rate": counts.get("missing_2", 0) / max(1, utterance_count),
        "exact_pattern_counts": {
            key.removeprefix("observed_pattern_"): value
            for key, value in sorted(counts.items())
            if key.startswith("observed_pattern_")
        },
    }
    return result
