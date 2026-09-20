#!/usr/bin/env python3
"""Build a deterministic, Week4-only teacher submission archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import zipfile


MANIFEST_NAME = "SHA256SUMS.txt"
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth"}
FORBIDDEN_NAME_PARTS = {"shutdown", "private_mapping"}
MARKDOWN_LINK = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_files(source: Path) -> list[Path]:
    if not source.is_dir() or source.name != "Week4":
        raise ValueError("source must be the Week4 teacher directory")
    paths = sorted(source.rglob("*"), key=lambda path: path.relative_to(source).as_posix())
    if any(path.is_symlink() for path in paths):
        raise ValueError("symlink is forbidden in the teacher submission")
    files = [path for path in paths if path.is_file() and path.name != MANIFEST_NAME]
    for path in files:
        lowered = path.name.lower()
        if path.suffix.lower() in WEIGHT_SUFFIXES or any(part in lowered for part in FORBIDDEN_NAME_PARTS):
            raise ValueError(f"forbidden teacher artifact: {path.relative_to(source)}")
        if path.suffix.lower() == ".md":
            content = path.read_text(encoding="utf-8")
            if "deliverables/week4" in content:
                raise ValueError(f"repository-only path in {path.relative_to(source)}")
            for target in MARKDOWN_LINK.findall(content):
                target = target.split("#", 1)[0]
                if not target or "://" in target or target.startswith(("mailto:", "#")):
                    continue
                resolved = (path.parent / target).resolve()
                if not resolved.is_relative_to(source.resolve()) or not resolved.exists():
                    raise ValueError(f"non-self-contained link in {path.relative_to(source)}: {target}")
    return files


def write_manifest(source: Path, files: list[Path]) -> Path:
    manifest = source / MANIFEST_NAME
    lines = [f"{sha256(path)}  {path.relative_to(source).as_posix()}" for path in files]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def package_week4(source: Path, output: Path) -> dict[str, object]:
    source = source.resolve()
    files = collect_files(source)
    manifest = write_manifest(source, files)
    archive_files = sorted(files + [manifest], key=lambda path: path.relative_to(source).as_posix())
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in archive_files:
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(f"Week4/{relative}", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    result = {
        "status": "PASS",
        "source": str(source),
        "archive": str(output.resolve()),
        "file_count": len(archive_files),
        "archive_bytes": output.stat().st_size,
        "archive_sha256": sha256(output),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = package_week4(args.source, args.output)
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    sidecar.write_text(f"{result['archive_sha256']}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
