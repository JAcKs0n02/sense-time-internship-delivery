#!/usr/bin/env python3
"""Build a reproducible manifest for the five frozen Week 5 images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


FIELDNAMES = (
    "image_id",
    "relative_path",
    "image_type",
    "source",
    "license",
    "width",
    "height",
    "file_bytes",
    "sha256",
)
EXPECTED_TYPES = {
    "table_screenshot",
    "natural_scene",
    "logo",
    "handwritten_formula",
    "ui_interface",
}


def read_sources(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("image source metadata must be a JSON list")
    return payload


def safe_image_path(images_dir: Path, filename: str) -> Path:
    root = images_dir.resolve()
    path = (root / filename).resolve()
    if path.parent != root:
        raise ValueError(f"image path must be a direct child of images directory: {filename}")
    if not path.is_file():
        raise ValueError(f"image file does not exist: {filename}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_rows(images_dir: Path, source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(source_rows) != 5:
        raise ValueError(f"expected exactly five source records, got {len(source_rows)}")
    required = {"image_id", "filename", "image_type", "source", "license"}
    rows: list[dict[str, Any]] = []
    for index, source_row in enumerate(source_rows, start=1):
        missing = sorted(required - source_row.keys())
        if missing:
            raise ValueError(f"source record {index} is missing fields: {', '.join(missing)}")
        for field in required:
            if not isinstance(source_row[field], str) or not source_row[field].strip():
                raise ValueError(f"source record {index} has an empty {field}")
        filename = source_row["filename"]
        path = safe_image_path(images_dir, filename)
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError(f"unsupported raster image format for {filename}: {image.format}")
        except UnidentifiedImageError as exc:
            raise ValueError(f"Pillow cannot decode image: {filename}") from exc
        rows.append(
            {
                "image_id": source_row["image_id"],
                "relative_path": filename,
                "image_type": source_row["image_type"],
                "source": source_row["source"],
                "license": source_row["license"],
                "width": width,
                "height": height,
                "file_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    image_ids = [row["image_id"] for row in rows]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("image_id values must be unique")
    actual_types = {row["image_type"] for row in rows}
    if actual_types != EXPECTED_TYPES:
        raise ValueError(
            f"image types mismatch: actual={sorted(actual_types)}, expected={sorted(EXPECTED_TYPES)}"
        )
    return rows


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = build_rows(args.images_dir, read_sources(args.sources))
        write_manifest(args.output, rows)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        args.output.unlink(missing_ok=True)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {len(rows)} image records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
