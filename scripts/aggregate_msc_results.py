from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.root).rglob("result.json")):
        payload = json.loads(path.read_text())
        metrics = payload.get("metrics", {})
        rows.append(
            {
                "path": str(path.relative_to(args.root)),
                "mode": payload.get("mode"),
                "split": payload.get("split"),
                "gamma": payload.get("gamma"),
                "acc": metrics.get("acc"),
                "f1": metrics.get("f1"),
                "mae": metrics.get("mae"),
                "corr": metrics.get("corr"),
                "use_rate": payload.get("use_rate"),
                "skip_rate": payload.get("skip_rate"),
            }
        )
    with Path(args.output).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
