from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    return subprocess.call(
        [
            sys.executable,
            "-u",
            "scripts/build_knowledge_index.py",
            *sys.argv[1:],
        ],
        cwd=REPO_ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
