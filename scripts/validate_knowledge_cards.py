from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.knowledge.audit import read_jsonl
from ComP.knowledge.schema import KnowledgeCard


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.cards)
    errors = []
    seen = set()
    for index, row in enumerate(rows):
        try:
            card = KnowledgeCard(**row)
        except (TypeError, ValueError, FloatingPointError) as exc:
            errors.append({"row": index, "error": str(exc)})
            continue
        if card.knowledge_id in seen:
            errors.append(
                {"row": index, "error": f"duplicate knowledge_id {card.knowledge_id}"}
            )
        seen.add(card.knowledge_id)
    result = {
        "card_count": len(rows),
        "unique_knowledge_ids": len(seen),
        "valid": not errors,
        "errors": errors,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
