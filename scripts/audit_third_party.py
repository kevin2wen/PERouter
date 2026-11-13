from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_snapshot(name: str, repo: Path) -> dict[str, object]:
    files = [
        path
        for path in repo.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and (
            path.suffix in {".py", ".yaml", ".yml", ".json", ".md", ".txt"}
            or path.name in {"LICENSE", "LICENCE"}
        )
    ]
    hashes = {
        str(path.relative_to(repo)): sha256_file(path)
        for path in sorted(files)
    }
    license_path = next(
        (
            repo / candidate
            for candidate in ("LICENSE", "LICENCE")
            if (repo / candidate).is_file()
        ),
        None,
    )
    return {
        "name": name,
        "path": str(repo),
        "commit_sha": git_commit(repo),
        "license_file": str(license_path) if license_path else None,
        "license_sha256": sha256_file(license_path) if license_path else None,
        "file_count": len(hashes),
        "files": hashes,
    }


def parse_repo(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("repository must be NAME=PATH")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {resolved}")
    return name, resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", action="append", type=parse_repo, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {
        name: source_snapshot(name, path)
        for name, path in args.repo
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
