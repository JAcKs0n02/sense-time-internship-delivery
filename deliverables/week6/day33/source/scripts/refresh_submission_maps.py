#!/usr/bin/env python3
"""Refresh mapped teacher-package bytes and root checksums deterministically."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
SUBMISSION = REPO_ROOT / "Submission/Week6"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    for mapping_path in sorted(SUBMISSION.glob("Day*/Submission_Map.json")):
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        day = mapping_path.parent
        for record in mapping["files"]:
            source = REPO_ROOT / record["formal_source"]
            target = day / record["submission_path"]
            if not source.is_file() or not target.parent.is_dir():
                raise FileNotFoundError(source)
            shutil.copy2(source, target)
            record["bytes"] = source.stat().st_size
            record["sha256"] = _sha256(source)
        mapping_path.write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    checksum_path = SUBMISSION / "SHA256SUMS.txt"
    paths = sorted(
        path for path in SUBMISSION.rglob("*") if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "".join(
            f"{_sha256(path)}  ./{path.relative_to(SUBMISSION).as_posix()}\n"
            for path in paths
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
