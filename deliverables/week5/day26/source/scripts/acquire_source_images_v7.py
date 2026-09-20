#!/usr/bin/env python3
"""Acquire the real, license-traceable images for the Day26 v7 dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote


CATEGORIES = ("natural_scene", "ocr", "chart_table", "ui", "formula")

SCIKIT_REVISION = "e8a42ba85aaf5fd9322ef9ca51bc21063b22fcae"
PADDLE_REVISION = "2661c7c0ef5c613e8f93c6e93b2e052399f0f854"
PIX2TEXT_REVISION = "f881e9df93d1ee6e90e823a66776ae0bb501f810"
WINDOWS_DOCS_REVISION = "9a712a52ab5ccb1fbe24e97e582a9f884388a3cc"

REPOSITORIES = {
    "scikit-image": {
        "owner_repo": "scikit-image/scikit-image",
        "revision": SCIKIT_REVISION,
        "author": "scikit-image team and credited sample-data contributors",
        "license": "BSD-3-Clause",
        "license_path": "LICENSE.txt",
    },
    "paddleocr": {
        "owner_repo": "PaddlePaddle/PaddleOCR",
        "revision": PADDLE_REVISION,
        "author": "PaddleOCR contributors",
        "license": "Apache-2.0",
        "license_path": "LICENSE",
    },
    "pix2text": {
        "owner_repo": "breezedeus/Pix2Text",
        "revision": PIX2TEXT_REVISION,
        "author": "BreezeDeus and Pix2Text contributors",
        "license": "MIT",
        "license_path": "LICENSE",
    },
    "windows-dev-docs": {
        "owner_repo": "MicrosoftDocs/windows-dev-docs",
        "revision": WINDOWS_DOCS_REVISION,
        "author": "Microsoft and Windows documentation contributors",
        "license": "CC-BY-4.0",
        "license_path": "LICENSE",
    },
}


NEW_HELDOUT_SOURCES = (
    ("ocr-v7-dev-01", "dev", "ocr", "paddleocr", "tests/test_files/book.jpg"),
    ("ocr-v7-dev-02", "dev", "ocr", "paddleocr", "tests/test_files/seal.png"),
    ("ocr-v7-final-01", "final", "ocr", "paddleocr", "tests/test_files/textline.png"),
    ("ocr-v7-final-02", "final", "ocr", "pix2text", "docs/examples/english.jpg"),
    ("ui-v7-dev-01", "dev", "ui", "windows-dev-docs", "hub/powertoys/images/advanced-paste/advanced-paste.png"),
    ("ui-v7-dev-02", "dev", "ui", "windows-dev-docs", "hub/powertoys/images/fancyzones/fancy-zones.png"),
    ("ui-v7-final-01", "final", "ui", "windows-dev-docs", "hub/powertoys/images/color-picker/settings.png"),
    ("ui-v7-final-02", "final", "ui", "windows-dev-docs", "hub/powertoys/images/general/settings-home.png"),
    ("formula-v7-dev-01", "dev", "formula", "paddleocr", "tests/test_files/doc_with_formula.png"),
    ("formula-v7-dev-02", "dev", "formula", "pix2text", "docs/examples/mixed.jpg"),
    ("formula-v7-final-02", "final", "formula", "pix2text", "docs/examples/page2.png"),
)

WIKIMEDIA_SOURCES = (
    (
        "natural_scene-v7-dev-03",
        "dev",
        "natural_scene",
        "Lake Mountain Landscape.jpg",
        "Bonnie Moreland",
        "CC0-1.0",
        "https://creativecommons.org/publicdomain/zero/1.0/",
        1280,
    ),
    (
        "natural_scene-v7-dev-04",
        "dev",
        "natural_scene",
        "Forest Path (179749337).jpeg",
        "Vtx",
        "CC0-1.0",
        "https://creativecommons.org/publicdomain/zero/1.0/",
        960,
    ),
    (
        "natural_scene-v7-final-03",
        "final",
        "natural_scene",
        "Palm trees on a beach.jpg",
        "www.Pixel.la Free Stock Photos",
        "CC0-1.0",
        "https://creativecommons.org/publicdomain/zero/1.0/",
        1280,
    ),
    (
        "natural_scene-v7-final-04",
        "final",
        "natural_scene",
        "Dog park!.jpg",
        "Sarah Naegels",
        "CC-BY-2.0",
        "https://creativecommons.org/licenses/by/2.0/",
        1280,
    ),
    (
        "formula-v7-final-03",
        "final",
        "formula",
        "Quadratic formula via completing the square.png",
        "Jacob Rus",
        "CC-BY-SA-4.0",
        "https://creativecommons.org/licenses/by-sa/4.0/",
        1280,
    ),
)

DIRECT_DOWNLOAD_FALLBACKS = {
    "https://commons.wikimedia.org/wiki/Special:Redirect/file/Lake%20Mountain%20Landscape.jpg?width=1280": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/75/Lake_Mountain_Landscape.jpg/1280px-Lake_Mountain_Landscape.jpg?utm_source=commons.wikimedia.org&utm_campaign=index&utm_content=thumbnail",
    "https://commons.wikimedia.org/wiki/Special:Redirect/file/Forest%20Path%20%28179749337%29.jpeg?width=960": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Forest_Path_%28179749337%29.jpeg/960px-Forest_Path_%28179749337%29.jpeg?utm_source=commons.wikimedia.org&utm_campaign=index&utm_content=thumbnail",
    "https://commons.wikimedia.org/wiki/Special:Redirect/file/Quadratic%20formula%20via%20completing%20the%20square.png?width=1280": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/39/Quadratic_formula_via_completing_the_square.png/1280px-Quadratic_formula_via_completing_the_square.png?utm_source=commons.wikimedia.org&utm_campaign=index&utm_content=thumbnail",
    "https://commons.wikimedia.org/wiki/Special:Redirect/file/Palm%20trees%20on%20a%20beach.jpg?width=1280": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Palm_trees_on_a_beach.jpg/1280px-Palm_trees_on_a_beach.jpg?utm_source=commons.wikimedia.org&utm_campaign=index&utm_content=thumbnail",
    "https://commons.wikimedia.org/wiki/Special:Redirect/file/Dog%20park%21.jpg?width=1280": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Dog_park%21.jpg/1280px-Dog_park%21.jpg?utm_source=commons.wikimedia.org&utm_campaign=index&utm_content=thumbnail",
}

CHART_SOURCES = (
    ("chart_table-v7-dev-01", "dev", "life-expectancy-vs-health-expenditure"),
    ("chart_table-v7-dev-02", "dev", "co-emissions-per-capita"),
    ("chart_table-v7-final-01", "final", "renewable-share-energy"),
    ("chart_table-v7-final-02", "final", "population-by-age-group"),
)


def _github_spec(image_id: str, split: str, category: str, repository: str, source_path: str) -> dict:
    repo = REPOSITORIES[repository]
    revision = repo["revision"]
    owner_repo = repo["owner_repo"]
    encoded = quote(source_path, safe="/")
    extension = Path(source_path).suffix.lower()
    return {
        "image_id": image_id,
        "split": split,
        "category": category,
        "repository": repository,
        "source_revision": revision,
        "source_revision_kind": "git_commit",
        "source_path": source_path,
        "image_relative_path": f"images_v7/{split}/{image_id}{extension}",
        "source_page_url": f"https://github.com/{owner_repo}/blob/{revision}/{encoded}",
        "download_url": f"https://cdn.jsdelivr.net/gh/{owner_repo}@{revision}/{encoded}",
        "canonical_download_url": (
            f"https://raw.githubusercontent.com/{owner_repo}/{revision}/{encoded}"
        ),
        "author": repo["author"],
        "license": repo["license"],
        "license_url": f"https://github.com/{owner_repo}/blob/{revision}/{repo['license_path']}",
        "provenance_role": "new_v7_heldout_bytes",
    }


def _chart_spec(image_id: str, split: str, slug: str) -> dict:
    return {
        "image_id": image_id,
        "split": split,
        "category": "chart_table",
        "repository": "our-world-in-data-grapher",
        "source_revision": "2026-08-21",
        "source_revision_kind": "retrieval_date",
        "source_path": slug,
        "image_relative_path": f"images_v7/{split}/{image_id}.png",
        "source_page_url": f"https://ourworldindata.org/grapher/{slug}",
        "download_url": f"https://ourworldindata.org/grapher/{slug}.png",
        "author": "Our World in Data / Global Change Data Lab",
        "license": "CC-BY-4.0",
        "license_url": "https://ourworldindata.org/faqs#can-i-reuse-or-republish-your-charts",
        "provenance_role": "new_v7_heldout_bytes",
    }


def _wikimedia_spec(
    image_id: str,
    split: str,
    category: str,
    filename: str,
    author: str,
    license_name: str,
    license_url: str,
    width: int,
) -> dict:
    encoded = quote(filename, safe="")
    extension = Path(filename).suffix.lower()
    return {
        "image_id": image_id,
        "split": split,
        "category": category,
        "repository": "wikimedia-commons",
        "source_revision": "2026-08-21",
        "source_revision_kind": "retrieval_date",
        "source_path": filename,
        "image_relative_path": f"images_v7/{split}/{image_id}{extension}",
        "source_page_url": f"https://commons.wikimedia.org/wiki/File:{encoded}",
        "download_url": (
            "https://commons.wikimedia.org/wiki/Special:Redirect/file/"
            f"{encoded}?width={width}"
        ),
        "author": author,
        "license": license_name,
        "license_url": license_url,
        "provenance_role": "new_v7_heldout_bytes",
    }


def build_source_specs(legacy_manifest: list[dict]) -> list[dict]:
    """Reuse only the 50 original train bytes, then add 20 unseen heldout images."""

    specs = []
    for row in legacy_manifest:
        if row.get("split") != "train":
            continue
        extension = Path(row["image_relative_path"]).suffix.lower()
        specs.append(
            {
                "image_id": row["image_id"],
                "split": "train",
                "category": row["category"],
                "repository": row["repository"],
                "source_revision": row["revision"],
                "source_revision_kind": "git_commit",
                "source_path": row["source_path"],
                "image_relative_path": f"images_v7/train/{row['image_id']}{extension}",
                "legacy_image_relative_path": row["image_relative_path"],
                "source_page_url": row["source_page_url"],
                "download_url": row["download_url"],
                "author": row["author"],
                "license": row["license"],
                "license_url": row["license_url"],
                "expected_sha256": row["sha256"],
                "expected_file_bytes": row["file_bytes"],
                "provenance_role": "reaudited_v6_train_bytes",
            }
        )
    specs.extend(_github_spec(*row) for row in NEW_HELDOUT_SOURCES)
    specs.extend(_wikimedia_spec(*row) for row in WIKIMEDIA_SOURCES)
    specs.extend(_chart_spec(*row) for row in CHART_SOURCES)
    specs.sort(key=lambda row: (row["split"], row["category"], row["image_id"]))
    return specs


def download_bytes(url: str) -> bytes:
    candidates = (url, DIRECT_DOWNLOAD_FALLBACKS[url]) if url in DIRECT_DOWNLOAD_FALLBACKS else (url,)
    last_error = None
    for candidate in candidates:
        command = [
            "curl",
            "-4",
            "-fsSL",
            "--retry",
            "2",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "--connect-timeout",
            "15",
            "--max-time",
            "60",
            "-A",
            "Day26V7DatasetBuilder/1.0",
            candidate,
        ]
        try:
            return subprocess.run(
                command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            ).stdout
        except subprocess.CalledProcessError as error:
            last_error = error
    assert last_error is not None
    raise last_error


def obtain_new_bytes(spec: dict, output_dir: Path) -> bytes:
    target = output_dir / spec["image_relative_path"]
    if target.exists():
        raw = target.read_bytes()
        expected = spec.get("expected_sha256")
        if expected and hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"preseeded bytes do not match expected sha256: {target}")
        return raw
    return download_bytes(spec["download_url"])


def ensure_exact_bytes(path: Path, raw: bytes, expected_sha256: str | None = None) -> str:
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"downloaded bytes do not match expected sha256 for {path.name}")
    if path.exists():
        existing = path.read_bytes()
        if existing != raw:
            raise ValueError(f"refusing to overwrite different frozen bytes: {path}")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return digest


def validate_frozen_manifest(manifest: list[dict], legacy_manifest: list[dict]) -> None:
    if len(manifest) != 70:
        raise ValueError(f"expected 70 manifest rows, got {len(manifest)}")
    old_heldout = {
        row["sha256"] for row in legacy_manifest if row["split"] in {"dev", "final"}
    }
    reused = sorted({row["sha256"] for row in manifest} & old_heldout)
    if reused:
        raise ValueError(f"prior heldout image bytes reused: {reused}")
    owners: dict[str, list[tuple[str, str]]] = {}
    for row in manifest:
        owners.setdefault(row["sha256"], []).append((row["image_id"], row["split"]))
    duplicate = {digest: rows for digest, rows in owners.items() if len(rows) > 1}
    if duplicate:
        raise ValueError(f"duplicate or cross-split image bytes: {duplicate}")


def acquire(
    specs: list[dict], legacy_data_dir: Path, output_dir: Path, legacy_manifest: list[dict]
) -> list[dict]:
    from PIL import Image

    new_specs = [
        spec for spec in specs if spec["provenance_role"] == "new_v7_heldout_bytes"
    ]
    with ThreadPoolExecutor(max_workers=8) as executor:
        downloaded = dict(
            zip(
                (spec["image_id"] for spec in new_specs),
                executor.map(lambda spec: obtain_new_bytes(spec, output_dir), new_specs),
            )
        )

    manifest = []
    for index, spec in enumerate(specs, start=1):
        if spec["provenance_role"] == "reaudited_v6_train_bytes":
            raw = (legacy_data_dir / spec["legacy_image_relative_path"]).read_bytes()
            if len(raw) != spec["expected_file_bytes"]:
                raise ValueError(f"legacy file byte mismatch: {spec['image_id']}")
        else:
            raw = downloaded[spec["image_id"]]
        target = output_dir / spec["image_relative_path"]
        digest = ensure_exact_bytes(target, raw, spec.get("expected_sha256"))
        try:
            with Image.open(target) as image:
                image.verify()
            with Image.open(target) as image:
                width, height = image.size
                image_format = image.format
        except Exception as error:
            raise ValueError(f"undecodable image: {target}") from error
        row = dict(spec)
        row.update(
            {
                "sha256": digest,
                "file_bytes": len(raw),
                "width": width,
                "height": height,
                "image_format": image_format,
                "mime_type": mimetypes.guess_type(target.name)[0]
                or "application/octet-stream",
            }
        )
        manifest.append(row)
        print(f"[{index:02d}/70] {spec['image_id']} {width}x{height}", flush=True)
    validate_frozen_manifest(manifest, legacy_manifest)
    return manifest


def write_outputs(specs: list[dict], manifest: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source_image_specs_v7.json").write_text(
        json.dumps(specs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "source_image_manifest_v7.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "source_image_manifest_v7.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = sorted({field for row in manifest for field in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--legacy-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    legacy = json.loads(args.legacy_manifest.read_text(encoding="utf-8"))
    specs = build_source_specs(legacy)
    manifest = acquire(specs, args.legacy_data_dir, args.output_dir, legacy)
    write_outputs(specs, manifest, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
