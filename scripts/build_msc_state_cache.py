from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.knowledge.msc.cache import build_state_cache, save_state_cache


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--card-adapter-checkpoint", required=True)
    parser.add_argument("--mask-manifest", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    cache = torch.load(args.candidate_cache, map_location="cpu", weights_only=False)
    if cache.get("split") != args.split:
        raise SystemExit(
            f"candidate cache split={cache.get('split')} != --split {args.split}"
        )
    cache["cache_sha256"] = sha256_file(args.candidate_cache)
    payload = build_state_cache(cache)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_state_cache(payload, output / "msc_state_cache.pt")
    observed = payload["observed_mask"]
    target = payload["target_modality"]
    manifest = {
        "msc_format_version": payload["msc_format_version"],
        "split": payload["split"],
        "source_cache_sha256": payload["source_cache_sha256"],
        "student_checkpoint_sha256": sha256_file(args.student_checkpoint),
        "card_adapter_sha256": sha256_file(args.card_adapter_checkpoint),
        "mask_manifest_sha256": sha256_file(args.mask_manifest),
        "query_count": len(payload["query_ids"]),
        "queries_by_target": {
            name: int((target.argmax(dim=-1) == index).sum())
            for index, name in enumerate(("audio", "text", "video"))
        },
        "queries_by_missing_count": {
            str(count): int(((1.0 - observed).sum(dim=-1).long() == count).sum())
            for count in range(4)
        },
        "mean_missing_rate": float(payload["missing_rate"].mean()),
        "valid_candidate_rate": float(payload["candidate_mask"].float().mean()),
    }
    (output / "msc_state_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
