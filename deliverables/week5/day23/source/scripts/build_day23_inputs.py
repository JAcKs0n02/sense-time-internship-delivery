#!/usr/bin/env python3
"""Freeze Day 23 prompts, generation settings, and the 5×5 inference matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_REPOSITORY = "Qwen/Qwen2-VL-7B-Instruct"
MODEL_REVISION = "eed13092ef92e448dd6875b2a00151bd3f7db0ac"
IMAGE_ORDER = (
    "w5-table-01",
    "w5-scene-01",
    "w5-logo-01",
    "w5-formula-01",
    "w5-ui-01",
)
IMAGE_SLUGS = {
    "w5-table-01": "table",
    "w5-scene-01": "scene",
    "w5-logo-01": "logo",
    "w5-formula-01": "formula",
    "w5-ui-01": "ui",
}
PROMPTS = (
    (
        "description",
        "description",
        "请描述图片中可见的主要内容，包括主要对象、属性和位置关系。只陈述图像能够支持的事实；无法确认的信息请明确说明，不要猜测。",
    ),
    (
        "ocr",
        "ocr",
        "请提取图片中所有可识别的文字、数字和数学符号，并尽量保留原有的行列或层级结构。无法辨认的部分请标记为[无法辨认]；如果没有可识别文字，请明确说明。",
    ),
    (
        "structure_explanation",
        "structure",
        "请解释图片中元素的组织结构、层级和相互关系。如果是表格，请解释行列关系；如果是公式，请解释分子、分母和符号关系；如果是UI、Logo或自然场景，请解释界面层级、几何组成或空间关系。不要补充图片中不存在的数值或对象。",
    ),
    (
        "aesthetic_review",
        "aesthetic",
        "请从构图、色彩、对比度、视觉层级、清晰度和可读性方面评价这张图片。请区分可见的视觉事实与主观评价，并简要说明评价依据。",
    ),
    (
        "implicit_reasoning",
        "implicit",
        "请根据图片中可见的证据，推理最多3条可能的隐含信息。每条必须分别写明“图像证据”“合理推断”和“不确定性”。如果图像证据不足，请明确说明，不要把猜测当成事实。",
    ),
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_manifest(rows: list[dict[str, str]]) -> None:
    actual_ids = [row.get("image_id", "") for row in rows]
    if actual_ids != list(IMAGE_ORDER):
        raise ValueError(
            "manifest must contain exactly five frozen image IDs in Day 22 order; "
            f"actual={actual_ids}"
        )
    for row in rows:
        image_id = row["image_id"]
        digest = row.get("sha256", "")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"invalid image sha256 for {image_id}")
        relative_path = Path(row.get("relative_path", ""))
        if relative_path.is_absolute() or ".." in relative_path.parts or not relative_path.name:
            raise ValueError(f"unsafe image path for {image_id}")


def generation_config() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "local_files_only": True,
        "do_sample": False,
        "seed": 42,
        "max_new_tokens": 1024,
        "min_pixels": 200704,
        "max_pixels": 301056,
        "processor_size": {
            "shortest_edge": 200704,
            "longest_edge": 301056,
        },
    }


def prompt_config() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "grounding_policy": (
            "Only state facts supported by the image; expose uncertainty and do not guess."
        ),
        "templates": [
            {
                "prompt_template_id": f"d23-prompt-{index:02d}",
                "question_type": question_type,
                "record_slug": slug,
                "prompt": prompt,
            }
            for index, (question_type, slug, prompt) in enumerate(PROMPTS, start=1)
        ],
    }


def build_artifacts(manifest_path: Path, output_root: Path) -> dict[str, Any]:
    rows = read_manifest(manifest_path)
    validate_manifest(rows)

    config_path = output_root / "configs" / "generation_config.json"
    prompts_path = output_root / "configs" / "prompt_templates.json"
    write_json(config_path, generation_config())
    write_json(prompts_path, prompt_config())
    config_sha = sha256(config_path)
    prompts_sha = sha256(prompts_path)

    manifest_by_id = {row["image_id"]: row for row in rows}
    records: list[dict[str, Any]] = []
    execution_index = 1
    for image_id in IMAGE_ORDER:
        image = manifest_by_id[image_id]
        for prompt_index, (question_type, slug, prompt) in enumerate(PROMPTS, start=1):
            records.append(
                {
                    "record_id": f"d23-{IMAGE_SLUGS[image_id]}-{slug}",
                    "execution_index": execution_index,
                    "image_id": image_id,
                    "image_type": image["image_type"],
                    "image_relative_path": f"images/{image['relative_path']}",
                    "image_sha256": image["sha256"],
                    "question_type": question_type,
                    "prompt_template_id": f"d23-prompt-{prompt_index:02d}",
                    "prompt": prompt,
                    "model_revision": MODEL_REVISION,
                    "generation_config_sha256": config_sha,
                }
            )
            execution_index += 1

    matrix = {
        "schema_version": "1.0",
        "record_count": len(records),
        "image_count": len(IMAGE_ORDER),
        "question_type_count": len(PROMPTS),
        "generation_config_sha256": config_sha,
        "prompt_templates_sha256": prompts_sha,
        "records": records,
    }
    matrix_path = output_root / "source" / "data" / "inference_matrix.json"
    write_json(matrix_path, matrix)
    return {
        "record_count": len(records),
        "generation_config_sha256": config_sha,
        "prompt_templates_sha256": prompts_sha,
        "matrix_sha256": sha256(matrix_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_artifacts(args.manifest, args.output_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
