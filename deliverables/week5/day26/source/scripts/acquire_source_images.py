#!/usr/bin/env python3
"""Download the 70 pinned, license-traceable source images used by Day26."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import mimetypes
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote


CATEGORIES = ("natural_scene", "ocr", "chart_table", "ui", "formula")

REPOSITORIES = {
    "scikit-image": {
        "owner_repo": "scikit-image/scikit-image",
        "revision": "e8a42ba85aaf5fd9322ef9ca51bc21063b22fcae",
        "author": "scikit-image team and credited sample-data contributors",
        "license": "BSD-3-Clause",
        "license_path": "LICENSE.txt",
    },
    "paddleocr": {
        "owner_repo": "PaddlePaddle/PaddleOCR",
        "revision": "2661c7c0ef5c613e8f93c6e93b2e052399f0f854",
        "author": "PaddleOCR contributors",
        "license": "Apache-2.0",
        "license_path": "LICENSE",
    },
    "pix2text": {
        "owner_repo": "breezedeus/Pix2Text",
        "revision": "f881e9df93d1ee6e90e823a66776ae0bb501f810",
        "author": "BreezeDeus and Pix2Text contributors",
        "license": "MIT",
        "license_path": "LICENSE",
    },
    "omniparser": {
        "owner_repo": "microsoft/OmniParser",
        "revision": "354021201345a96178360b28733573e27269f2de",
        "author": "Microsoft and OmniParser contributors",
        "license": "CC-BY-4.0",
        "license_path": "LICENSE",
    },
}

SOURCE_PATHS = {
    "natural_scene": [
        ("scikit-image", "skimage/data/astronaut.png"),
        ("scikit-image", "skimage/data/camera.png"),
        ("scikit-image", "skimage/data/chelsea.png"),
        ("scikit-image", "skimage/data/coffee.png"),
        ("scikit-image", "skimage/data/coins.png"),
        ("scikit-image", "skimage/data/brick.png"),
        ("scikit-image", "skimage/data/grass.png"),
        ("scikit-image", "skimage/data/gravel.png"),
        ("scikit-image", "skimage/data/horse.png"),
        ("scikit-image", "skimage/data/hubble_deep_field.jpg"),
        ("scikit-image", "skimage/data/moon.png"),
        ("scikit-image", "skimage/data/motorcycle_left.png"),
        ("scikit-image", "skimage/data/retina.jpg"),
        ("scikit-image", "skimage/data/rocket.jpg"),
    ],
    "ocr": [
        ("paddleocr", "deploy/ios_demo/PaddleOCRDemo/Resources/SampleImages/general_ocr_002.jpg"),
        ("paddleocr", "docs/datasets/images/ArT.jpg"),
        ("paddleocr", "docs/datasets/images/CASIA_0.jpg"),
        ("paddleocr", "docs/datasets/images/LSVT_1.jpg"),
        ("paddleocr", "docs/datasets/images/LSVT_2.jpg"),
        ("paddleocr", "docs/datasets/images/captcha_demo.png"),
        ("paddleocr", "docs/datasets/images/ccpd_demo.png"),
        ("paddleocr", "docs/datasets/images/ch_doc1.jpg"),
        ("paddleocr", "docs/datasets/images/ch_doc3.jpg"),
        ("paddleocr", "docs/datasets/images/ch_street_rec_1.png"),
        ("paddleocr", "docs/datasets/images/ch_street_rec_2.png"),
        ("paddleocr", "docs/datasets/images/cmb_demo.jpg"),
        ("paddleocr", "docs/datasets/images/rctw.jpg"),
        ("paddleocr", "docs/datasets/images/wildreceipt_demo/1bbe854b8817dedb8585e0732089fd1f752d2cec.jpeg"),
    ],
    "chart_table": [
        ("paddleocr", "docs/datasets/images/table_PubTabNet_demo/PMC524509_007_00.png"),
        ("paddleocr", "docs/datasets/images/table_PubTabNet_demo/PMC535543_007_01.png"),
        ("paddleocr", "docs/datasets/images/table_tal_demo/1.jpg"),
        ("paddleocr", "docs/datasets/images/publaynet_demo/gt_PMC5086060_00002.jpg"),
        ("paddleocr", "docs/datasets/images/tablebank_demo/004.png"),
        ("paddleocr", "docs/datasets/images/tablebank_demo/005.png"),
        ("paddleocr", "docs/version2.x/ppstructure/images/table_1.png"),
        ("paddleocr", "docs/version2.x/ppstructure/model_train/images/table_ch_result1.jpg"),
        ("paddleocr", "docs/version2.x/ppstructure/model_train/images/table_ch_result2.jpg"),
        ("paddleocr", "docs/version2.x/ppstructure/model_train/images/table_ch_result3.jpg"),
        ("paddleocr", "docs/version2.x/ppstructure/model_train/images/tableocr_pipeline.jpg"),
        ("paddleocr", "tests/test_files/medal_table.png"),
        ("paddleocr", "tests/test_files/table.jpg"),
        ("paddleocr", "docs/datasets/images/publaynet_demo/gt_PMC3724501_00006.jpg"),
    ],
    "ui": [
        ("omniparser", "imgs/excel.png"),
        ("omniparser", "imgs/google_page.png"),
        ("omniparser", "imgs/ios.png"),
        ("omniparser", "imgs/mobile.png"),
        ("omniparser", "imgs/onenote.png"),
        ("omniparser", "imgs/saved_image_demo.png"),
        ("omniparser", "imgs/som_overlaid_omni.png"),
        ("omniparser", "imgs/teams.png"),
        ("omniparser", "imgs/windows.png"),
        ("omniparser", "imgs/windows_home.png"),
        ("omniparser", "imgs/windows_multitab.png"),
        ("omniparser", "imgs/windows_vm.png"),
        ("omniparser", "imgs/word.png"),
        ("omniparser", "imgs/demo_image.jpg"),
    ],
    "formula": [
        ("pix2text", "docs/examples/formula.jpg"),
        ("pix2text", "docs/examples/formula2.png"),
        ("pix2text", "docs/examples/formula3.png"),
        ("pix2text", "docs/examples/formula4.jpg"),
        ("pix2text", "docs/examples/formula5.jpg"),
        ("pix2text", "docs/examples/formula6.jpg"),
        ("pix2text", "docs/examples/hw-formula1.png"),
        ("pix2text", "docs/examples/hw-formula2.png"),
        ("pix2text", "docs/examples/hw-formula3.png"),
        ("pix2text", "docs/examples/math-formula-42.png"),
        ("paddleocr", "tests/test_files/formula.png"),
        ("paddleocr", "docs/version2.x/algorithm/formula_recognition/images/hme_00.jpg"),
        ("paddleocr", "docs/datasets/images/crohme_demo/hme_01.jpg"),
        ("paddleocr", "docs/datasets/images/crohme_demo/hme_02.jpg"),
    ],
}


def _split(index: int) -> str:
    if index <= 10:
        return "train"
    if index <= 12:
        return "dev"
    return "final"


def build_source_specs() -> list[dict]:
    specs = []
    for category in CATEGORIES:
        for index, (repository, source_path) in enumerate(SOURCE_PATHS[category], start=1):
            repo = REPOSITORIES[repository]
            owner_repo = repo["owner_repo"]
            revision = repo["revision"]
            encoded_path = quote(source_path, safe="/")
            extension = Path(source_path).suffix.lower()
            image_id = f"{category}-{index:02d}"
            split = _split(index)
            specs.append(
                {
                    "image_id": image_id,
                    "split": split,
                    "category": category,
                    "repository": repository,
                    "revision": revision,
                    "source_path": source_path,
                    "image_relative_path": f"images/{split}/{image_id}{extension}",
                    "source_page_url": f"https://github.com/{owner_repo}/blob/{revision}/{encoded_path}",
                    "download_url": f"https://raw.githubusercontent.com/{owner_repo}/{revision}/{encoded_path}",
                    "api_url": (
                        f"https://api.github.com/repos/{owner_repo}/contents/{encoded_path}"
                        f"?ref={revision}"
                    ),
                    "author": repo["author"],
                    "license": repo["license"],
                    "license_url": (
                        f"https://github.com/{owner_repo}/blob/{revision}/{repo['license_path']}"
                    ),
                }
            )
    return specs


SOURCE_SPECS = build_source_specs()


def _download(url: str) -> bytes:
    """Use curl because the local urllib TLS path intermittently stalls on GitHub."""
    command = [
        "curl",
        "-4",
        "-fsSL",
        "--retry",
        "4",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--max-time",
        "120",
        "-A",
        "Day26DatasetBuilder/1.0",
        url,
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout


def _decode_github_blob(payload: bytes) -> bytes:
    row = json.loads(payload)
    if row.get("encoding") != "base64" or not row.get("content"):
        raise ValueError("GitHub blob response did not contain base64 content")
    return base64.b64decode(row["content"], validate=False)


def download_source(spec: dict) -> bytes:
    """Prefer raw GitHub, then recover the exact pinned blob through the API."""
    try:
        return _download(spec["download_url"])
    except subprocess.CalledProcessError:
        metadata = json.loads(_download(spec["api_url"]))
        if metadata.get("encoding") == "base64" and metadata.get("content"):
            return _decode_github_blob(json.dumps(metadata).encode())
        git_url = metadata.get("git_url")
        if not git_url:
            raise ValueError(f"GitHub API response has no git_url for {spec['image_id']}")
        return _decode_github_blob(_download(git_url))


def verify_expected_bytes(image_id: str, raw: bytes, expected: dict) -> None:
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected.get("sha256") or len(raw) != expected.get("file_bytes"):
        raise ValueError(
            f"{image_id} does not match the bundled expected manifest: "
            f"sha256={digest}, bytes={len(raw)}"
        )


def read_expected_manifest(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != len(SOURCE_SPECS):
        raise ValueError("expected source manifest must contain exactly 70 rows")
    by_id = {row.get("image_id"): row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != {row["image_id"] for row in SOURCE_SPECS}:
        raise ValueError("expected source manifest image IDs do not match the acquisition spec")
    return by_id


def acquire(output_dir: Path, expected_by_id: dict[str, dict]) -> list[dict]:
    from PIL import Image

    def ensure_downloaded(spec: dict) -> None:
        target = output_dir / spec["image_relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(download_source(spec))

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(ensure_downloaded, SOURCE_SPECS))

    manifest = []
    for spec in SOURCE_SPECS:
        target = output_dir / spec["image_relative_path"]
        raw = target.read_bytes()
        verify_expected_bytes(spec["image_id"], raw, expected_by_id[spec["image_id"]])
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
                "sha256": hashlib.sha256(raw).hexdigest(),
                "file_bytes": len(raw),
                "width": width,
                "height": height,
                "image_format": image_format,
                "mime_type": mimetypes.guess_type(target.name)[0] or "application/octet-stream",
            }
        )
        manifest.append(row)
        print(f"[{len(manifest):02d}/70] {spec['image_id']} {width}x{height}")
    return manifest


def write_manifest(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "source_image_manifest.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = output_dir / "source_image_manifest.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest", type=Path)
    args = parser.parse_args()
    candidates = (
        [args.expected_manifest]
        if args.expected_manifest
        else [
            args.output_dir / "Source_Image_Manifest.json",
            args.output_dir / "source_image_manifest.json",
        ]
    )
    expected_path = next((path for path in candidates if path and path.is_file()), None)
    if expected_path is None:
        raise ValueError(
            "a bundled expected manifest is required; pass --expected-manifest"
        )
    rows = acquire(args.output_dir, read_expected_manifest(expected_path))
    write_manifest(rows, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
