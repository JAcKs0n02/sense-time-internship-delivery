#!/usr/bin/env python3
"""Build the frozen Day26 multimodal SFT datasets from reviewed image facts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


CATEGORIES = (
    "natural_scene",
    "ocr",
    "chart_table",
    "ui",
    "formula",
)
SPLIT_IMAGE_COUNTS = {"train": 10, "dev": 2, "final": 2}
MODES = {
    "train": ("direct", "structured", "explanation", "correction"),
    "dev": ("direct", "correction"),
    "final": ("direct", "correction"),
}

DIRECT_QUESTIONS = {
    "natural_scene": "请客观描述图中的主体、环境和重要细节。",
    "ocr": "请转写图中清晰可见的文字；无法确认的部分不要猜测。",
    "chart_table": "请提取图表中的关键信息，并概括最重要的结论。",
    "ui": "请说明当前界面的用途、主要区域以及用户可以执行的操作。",
    "formula": "请转写图中的公式，并解释各部分表达的含义。",
}
EXPLANATION_QUESTIONS = {
    "natural_scene": "你是根据哪些可见线索得出这段图像描述的？",
    "ocr": "请解释转写结果，并指出辨认这些文字所依据的视觉位置或排版线索。",
    "chart_table": "请说明你如何从图表的标题、坐标、图例或单元格得到结论。",
    "ui": "请解释这个界面的信息层级，以及完成主要操作的步骤。",
    "formula": "请逐项解释公式中的符号、结构和整体含义。",
}


def _structured_answer(fact: dict) -> str:
    return json.dumps(
        {
            "category": fact["category"],
            "subject": fact["subject"],
            "details": fact["details"],
            "visible_text": fact["visible_text"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _prompt_answer(fact: dict, mode: str) -> tuple[str, str]:
    if mode == "direct":
        return DIRECT_QUESTIONS[fact["category"]], fact["summary"]
    if mode == "structured":
        return (
            "请仅用一个 JSON 对象整理图像信息，字段为 category、subject、details、visible_text。",
            _structured_answer(fact),
        )
    if mode == "explanation":
        return EXPLANATION_QUESTIONS[fact["category"]], fact["explanation"]
    if mode == "correction":
        return (
            f"有人声称：\u201c{fact['false_claim']}\u201d。请先判断该说法是否符合图像，再给出可见依据。",
            fact["correction"],
        )
    raise ValueError(f"unsupported mode: {mode}")


def _validate_fact_counts(facts: list[dict]) -> None:
    counts = Counter((row.get("split"), row.get("category")) for row in facts)
    expected = {
        (split, category): image_count
        for split, image_count in SPLIT_IMAGE_COUNTS.items()
        for category in CATEGORIES
    }
    if counts != Counter(expected):
        raise ValueError(f"image balance mismatch: got={dict(counts)}, expected={expected}")
    image_ids = [row["image_id"] for row in facts]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("duplicate image_id")


def build_internal_records(facts: list[dict]) -> dict[str, list[dict]]:
    """Return auditable records while enforcing the approved 70-image contract."""
    _validate_fact_counts(facts)
    output: dict[str, list[dict]] = defaultdict(list)
    for fact in sorted(facts, key=lambda row: (row["split"], row["category"], row["image_id"])):
        split = fact["split"]
        for mode in MODES[split]:
            question, answer = _prompt_answer(fact, mode)
            output[split].append(
                {
                    "id": f"{fact['image_id']}--{mode}",
                    "split": split,
                    "category": fact["category"],
                    "mode": mode,
                    "source_image_id": fact["image_id"],
                    "image": fact["image_relative_path"],
                    "image_relative_path": fact["image_relative_path"],
                    "image_sha256": fact["sha256"],
                    "task_type": fact["category"],
                    "prompt": f"<image>\n{question}",
                    "user_instruction": question,
                    "reference_answer": answer,
                    "target_answer": answer,
                    "source": fact["source_page_url"],
                    "source_page_url": fact["source_page_url"],
                    "license": fact["license"],
                    "construction_method": "human-reviewed-source-fact+deterministic-template",
                }
            )
    return {split: output[split] for split in ("train", "dev", "final")}


def to_llamafactory_records(records: list[dict]) -> list[dict]:
    return [
        {
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["reference_answer"]},
            ],
            "images": [row["image"]],
        }
        for row in records
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    facts = json.loads(args.facts.read_text(encoding="utf-8"))
    records = build_internal_records(facts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in records.items():
        (args.output_dir / f"week5_vlm_{split}_internal.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / f"week5_vlm_{split}.json").write_text(
            json.dumps(to_llamafactory_records(rows), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
