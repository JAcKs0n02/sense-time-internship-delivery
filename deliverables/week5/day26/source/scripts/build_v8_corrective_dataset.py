#!/usr/bin/env python3
"""Build the bounded Day26 v8 corrective train and development exports."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path


CATEGORIES = ("natural_scene", "ocr", "chart_table", "ui", "formula")

DIRECT_PARAPHRASES = {
    "natural_scene": "仅依据画面可见内容，按主体、背景和相对位置作答；不要猜地点、人物身份或拍摄时间。",
    "ocr": "严格转写题目指定的可见区域；不要补全模糊、遮挡或未显示的文字。",
    "chart_table": "先概括图表主题与视觉编码，再给出两项能够从图中直接核对的关系。",
    "ui": "只根据截图说明应用或页面、主要区域和当前可见状态，不推断未显示的操作。",
    "formula": "只用 LaTeX 完整转写指定公式，逐项保留符号、上下标、括号和正负号，不添加解释。",
}
MODE_PARAPHRASES = {
    "evidence": "观察图片并列出三条互不重复的直接可见证据；不要加入常识或画外推断。",
    "grounding": "分开说明图片中可以确认的事实，以及图片没有显示、因此不能确认的信息。",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verification_paraphrase(row: dict) -> str:
    instruction = row["user_instruction"]
    marker = "待核验陈述："
    if marker not in instruction:
        raise ValueError(f"verification row lacks claim marker: {row.get('id')}")
    claim = instruction.split(marker, 1)[1].strip()
    return (
        "判断下面的陈述是否与图片一致；先写“符合”或“不符合”，再引用一条直接可见依据。"
        f"\n{marker}{claim}"
    )


def paraphrased_instruction(row: dict) -> str:
    mode = row.get("mode")
    category = row.get("category")
    if category not in CATEGORIES:
        raise ValueError(f"unsupported category: {category}")
    if mode == "direct":
        return DIRECT_PARAPHRASES[category]
    if mode == "verification":
        return _verification_paraphrase(row)
    if mode in MODE_PARAPHRASES:
        return MODE_PARAPHRASES[mode]
    raise ValueError(f"unsupported training mode: {mode}")


def _validate_alignment(public_rows: list[dict], internal_rows: list[dict], expected: int) -> None:
    if len(public_rows) != expected or len(internal_rows) != expected:
        raise ValueError(f"expected {expected} aligned rows")
    for public, internal in zip(public_rows, internal_rows):
        if internal.get("split") not in {"train", "final"}:
            raise ValueError("source rows must be v7 train or final")
        if public.get("images") != [internal.get("image_relative_path")]:
            raise ValueError(f"image alignment mismatch: {internal.get('id')}")
        messages = public.get("messages", [])
        if (
            len(messages) != 2
            or messages[0].get("role") != "user"
            or messages[1].get("role") != "assistant"
            or messages[1].get("content") != internal.get("target_answer")
        ):
            raise ValueError(f"message alignment mismatch: {internal.get('id')}")


def build_corrective_train(
    public_rows: list[dict], internal_rows: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Return 200 audited rows plus one deterministic paraphrase per row."""

    _validate_alignment(public_rows, internal_rows, 200)
    if any(row.get("split") != "train" for row in internal_rows):
        raise ValueError("corrective training inputs must be train-only")

    original_public = copy.deepcopy(public_rows)
    original_internal = copy.deepcopy(internal_rows)
    for row in original_internal:
        row["variant"] = "v7-original"

    paraphrase_public: list[dict] = []
    paraphrase_internal: list[dict] = []
    for public, internal in zip(public_rows, internal_rows):
        instruction = paraphrased_instruction(internal)
        prompt = f"<image>\n{instruction}"
        public_copy = copy.deepcopy(public)
        public_copy["messages"][0]["content"] = prompt
        internal_copy = copy.deepcopy(internal)
        internal_copy["id"] = f"{internal['id']}--v8p"
        internal_copy["prompt"] = prompt
        internal_copy["user_instruction"] = instruction
        internal_copy["construction_method"] = "human-double-checked-v8-paraphrase"
        internal_copy["variant"] = "v8-paraphrase"
        paraphrase_public.append(public_copy)
        paraphrase_internal.append(internal_copy)

    combined_public = original_public + paraphrase_public
    combined_internal = original_internal + paraphrase_internal
    counts = Counter(row["category"] for row in combined_internal)
    if counts != Counter({category: 80 for category in CATEGORIES}):
        raise ValueError(f"v8 category balance mismatch: {dict(counts)}")
    if any(
        "/dev/" in row["images"][0] or "/final/" in row["images"][0]
        for row in combined_public
    ):
        raise ValueError("held-out media cannot enter v8 corrective training")
    return combined_public, combined_internal


def assert_v7_final_unopened(result_paths: list[Path]) -> None:
    offenders = [path for path in result_paths if "final" in path.name.lower()]
    if offenders:
        raise ValueError(
            "v7 final has already been opened: " + ", ".join(str(path) for path in offenders)
        )


def promote_unopened_final(
    public_rows: list[dict], internal_rows: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Relabel the never-inferred v7 final cases as the v8 development split."""

    _validate_alignment(public_rows, internal_rows, 20)
    if any(row.get("split") != "final" for row in internal_rows):
        raise ValueError("only the unopened v7 final split may be promoted")

    dev_public = copy.deepcopy(public_rows)
    dev_internal = copy.deepcopy(internal_rows)
    for public, internal in zip(dev_public, dev_internal):
        user_prompt = public["messages"][0]["content"].replace("最终测试图片", "开发测试图片")
        public["messages"][0]["content"] = user_prompt
        internal["id"] = internal["id"].replace("-v7-final-", "-v8-dev-")
        internal["source_image_id"] = internal["source_image_id"].replace(
            "-v7-final-", "-v8-dev-"
        )
        internal["split"] = "dev"
        internal["prompt"] = user_prompt
        internal["user_instruction"] = user_prompt.removeprefix("<image>\n")
        internal["construction_method"] = "promoted-unopened-v7-final-to-v8-dev"
    return dev_public, dev_internal


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    data = root / "source/data"
    results = root / "source/results/v8"

    v7_result_paths = list((root / "source/results/v7").glob("*"))
    assert_v7_final_unopened(v7_result_paths)

    train_public_path = data / "week5_vlm_train_v7.json"
    train_internal_path = data / "week5_vlm_train_v7_internal.json"
    final_public_path = data / "week5_vlm_final_v7.json"
    final_internal_path = data / "week5_vlm_final_v7_internal.json"
    train_public = json.loads(train_public_path.read_text(encoding="utf-8"))
    train_internal = json.loads(train_internal_path.read_text(encoding="utf-8"))
    final_public = json.loads(final_public_path.read_text(encoding="utf-8"))
    final_internal = json.loads(final_internal_path.read_text(encoding="utf-8"))

    corrective_public, corrective_internal = build_corrective_train(
        train_public, train_internal
    )
    dev_public, dev_internal = promote_unopened_final(final_public, final_internal)

    corrective_path = data / "week5_vlm_train_v8_corrective.json"
    corrective_internal_path = data / "week5_vlm_train_v8_corrective_internal.json"
    dev_path = data / "week5_vlm_dev_v8.json"
    dev_internal_path = data / "week5_vlm_dev_v8_internal.json"
    write_json(corrective_path, corrective_public)
    write_json(corrective_internal_path, corrective_internal)
    write_json(dev_path, dev_public)
    write_json(dev_internal_path, dev_internal)

    train_prompts = {row["messages"][0]["content"] for row in corrective_public}
    dev_prompts = {row["messages"][0]["content"] for row in dev_public}
    train_images = {row["image_sha256"] for row in corrective_internal}
    dev_images = {row["image_sha256"] for row in dev_internal}
    manifest = {
        "schema_version": "day26-v8-data-manifest-1",
        "status": "PRE_REGISTERED",
        "v7_final_promotion": "UNOPENED_V7_FINAL_BECOMES_V8_DEV",
        "parent_sha256": {
            "train_public": sha256(train_public_path),
            "train_internal": sha256(train_internal_path),
            "unopened_final_public": sha256(final_public_path),
            "unopened_final_internal": sha256(final_internal_path),
        },
        "train_count": len(corrective_public),
        "dev_count": len(dev_public),
        "train_category_counts": dict(
            sorted(Counter(row["category"] for row in corrective_internal).items())
        ),
        "dev_category_counts": dict(
            sorted(Counter(row["category"] for row in dev_internal).items())
        ),
        "dev_mode_counts": dict(
            sorted(Counter(row["mode"] for row in dev_internal).items())
        ),
        "prompt_intersection": sorted(train_prompts & dev_prompts),
        "image_sha256_intersection": sorted(train_images & dev_images),
        "outputs": {
            "train_public": {"path": str(corrective_path.relative_to(root)), "sha256": sha256(corrective_path)},
            "train_internal": {"path": str(corrective_internal_path.relative_to(root)), "sha256": sha256(corrective_internal_path)},
            "dev_public": {"path": str(dev_path.relative_to(root)), "sha256": sha256(dev_path)},
            "dev_internal": {"path": str(dev_internal_path.relative_to(root)), "sha256": sha256(dev_internal_path)},
        },
    }
    if manifest["prompt_intersection"] or manifest["image_sha256_intersection"]:
        raise ValueError("v8 train/dev leakage detected")
    manifest_path = results / "data_manifest_v8.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
