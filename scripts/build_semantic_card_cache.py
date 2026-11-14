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

from ComP.knowledge.msc.card_encoder import SentenceCardEncoder


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-cache", required=True)
    parser.add_argument("--cards", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cache = torch.load(args.single_cache, map_location="cpu", weights_only=False)
    cards = {}
    for line in Path(args.cards).open(encoding="utf-8"):
        card = json.loads(line)
        cards[str(card["knowledge_id"])] = card
    encoder = SentenceCardEncoder(args.model_path, device=args.device)
    unique_ids = sorted(cards)
    texts = [SentenceCardEncoder.card_text(cards[card_id]) for card_id in unique_ids]
    embeddings = encoder.encode(texts)
    vector_by_id = {
        card_id: torch.from_numpy(embeddings[index])
        for index, card_id in enumerate(unique_ids)
    }
    candidate_ids = cache["candidate_ids"]
    candidate_vectors = torch.zeros(
        len(candidate_ids),
        int(cache["candidate_mask"].size(1)),
        embeddings.shape[1],
        dtype=torch.float32,
    )
    missing = 0
    for row, ids in enumerate(candidate_ids):
        for rank, card_id in enumerate(ids):
            vector = vector_by_id.get(str(card_id))
            if vector is None:
                missing += 1
                continue
            candidate_vectors[row, rank] = vector
    payload = dict(cache)
    payload["candidate_vectors"] = candidate_vectors
    payload["semantic_model_path"] = args.model_path
    payload["semantic_feature_dim"] = int(embeddings.shape[1])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output / "semantic_single_cache.pt")
    provenance = {
        "feature_name": "card_sentence_embedding",
        "source": "sentence-transformers/all-MiniLM-L6-v2",
        "model_path": args.model_path,
        "dimension": int(embeddings.shape[1]),
        "kind": "semantic",
        "builder_version": "semantic-card-v1",
        "cards": len(unique_ids),
        "queries": len(candidate_ids),
        "missing_card_vectors": missing,
        "encoder_fingerprint": encoder.fingerprint(texts),
        "single_cache_sha256": sha256_file(args.single_cache),
        "cache_sha256": sha256_file(output / "semantic_single_cache.pt"),
        "test_accessed": False,
    }
    (output / "feature_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
