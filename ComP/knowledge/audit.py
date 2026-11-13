from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_retrieval_rows(rows: Iterable[Mapping[str, object]]) -> None:
    for row in rows:
        if row.get("source_split") != "train":
            raise PermissionError("retrieval trace contains a non-train source")
        target = row.get("target_modality")
        query_sources = row.get("query_source_modalities")
        if not isinstance(query_sources, (list, tuple)):
            raise ValueError("query_source_modalities must be a sequence")
        if target in query_sources:
            raise PermissionError("target modality appears in the query source list")
        if row.get("same_video") is True:
            raise ValueError("retrieval trace contains a same-video candidate")
        if row.get("same_utterance") is True:
            raise ValueError("retrieval trace contains the same utterance")


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object")
            rows.append(value)
    return rows
