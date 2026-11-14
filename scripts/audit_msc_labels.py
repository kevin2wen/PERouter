from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComP.knowledge.msc.audit import audit_single_labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    cache = torch.load(args.single_cache, map_location="cpu", weights_only=False)
    report = audit_single_labels(cache)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "label_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if report["skip_label_mismatch"] or report["target_action_mismatch"] not in (None, 0):
        raise SystemExit("label audit failed: skip/target_action mismatch")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
