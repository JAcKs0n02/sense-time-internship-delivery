#!/usr/bin/env python3
"""Build the Day26 grounded-answer training variant and deterministic sampling view."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


CATEGORIES = ("natural_scene", "ocr", "chart_table", "ui", "formula")
MODES = ("direct", "structured", "explanation", "correction")

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

UI_ACTIONS = {
    "ui-01": "选择或编辑单元格、输入数据或公式，并使用顶部功能区",
    "ui-02": "在搜索框输入关键词并提交搜索，或打开可见导航链接",
    "ui-03": "点击应用图标启动应用，或使用底部 Dock 切换常用应用",
    "ui-04": "点击应用或 Dock 图标，并查看可见小组件信息",
    "ui-05": "选择笔记本或页面、编辑笔记，并使用顶部工具栏插入或绘制内容",
    "ui-06": "在页面中输入和编辑文档，并通过功能区调整格式或插入内容",
    "ui-07": "阅读页面说明、查看演示截图，并操作页面中可见的按钮",
    "ui-08": "选择会话、阅读或输入消息，并使用可见搜索或导航区域",
    "ui-09": "双击桌面快捷方式启动应用，或通过任务栏切换和管理窗口",
    "ui-10": "通过任务栏或桌面入口启动应用并管理桌面窗口",
    "ui-11": "在任务管理器查看进程与资源占用，或在浏览器搜索框输入查询",
    "ui-12": "打开桌面快捷方式，或通过任务栏启动和切换应用",
    "ui-13": "在页面中输入和编辑文档，并通过功能区调整格式或插入内容",
    "ui-14": "查看股票价格与走势图，并浏览页面中的市场摘要信息",
}


def _details(fact: dict) -> str:
    return "；".join(fact["details"])


def grounded_direct(fact: dict) -> str:
    category = fact["category"]
    details = _details(fact)
    subject = fact["subject"]
    visible = fact["visible_text"]
    if category == "natural_scene":
        return (
            f"主体与场景：{subject}。可见细节：{details}。"
            "图中未确认的地点、时间、型号或人物身份不作推断。"
        )
    if category == "ocr":
        return (
            f"清晰转写：{visible}。版面依据：{details}。"
            "模糊、遮挡或被裁切的其余内容不作推断。"
        )
    if category == "chart_table":
        return (
            f"图表类型与主题：{subject}。结构与关键信息：{details}。"
            f"可见字段或数值：{visible}。结论仅限上述可见内容，"
            "未显示的年份、事件、排名或数值不作推断。"
        )
    if category == "ui":
        actions = UI_ACTIONS.get(fact["image_id"], "在可见区域中浏览、输入或选择")
        return (
            f"界面用途与场景：{subject}。主要区域：{details}。"
            f"可执行操作：{actions}。未显示的按钮、弹窗或系统状态不作推断。"
        )
    if category == "formula":
        return (
            f"转写：{visible}。结构：{details}。含义：只说明图中可见的符号和运算结构；"
            "具体物理语义、变量取值或计算结果不作推断。"
        )
    raise ValueError(f"unsupported category: {category}")


def grounded_structured(fact: dict) -> str:
    return json.dumps(
        {
            "category": fact["category"],
            "subject": fact["subject"],
            "details": fact["details"],
            "visible_text": fact["visible_text"],
            "grounding": "only_visible_evidence",
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def grounded_explanation(fact: dict) -> str:
    return (
        f"可见依据：{_details(fact)}。可见文字或符号：{fact['visible_text']}。"
        f"这些线索支持将主体判断为“{fact['subject']}”；不使用图外信息。"
    )


def grounded_correction(fact: dict) -> str:
    return (
        f"结论：不符合。可见依据：图中主体是{fact['subject']}；"
        f"{_details(fact)}；可见文字或符号为：{fact['visible_text']}。"
    )


def _prompt_answer(fact: dict, mode: str) -> tuple[str, str]:
    if mode == "direct":
        return DIRECT_QUESTIONS[fact["category"]], grounded_direct(fact)
    if mode == "structured":
        return (
            "请仅用一个 JSON 对象整理图像信息，字段为 category、subject、details、visible_text、grounding。",
            grounded_structured(fact),
        )
    if mode == "explanation":
        return EXPLANATION_QUESTIONS[fact["category"]], grounded_explanation(fact)
    if mode == "correction":
        return (
            f"有人声称：“{fact['false_claim']}”。请先判断该说法是否符合图像，再给出可见依据。",
            grounded_correction(fact),
        )
    raise ValueError(f"unsupported mode: {mode}")


def build_grounded_records(facts: list[dict]) -> tuple[list[dict], list[dict]]:
    train_facts = sorted(
        (row for row in facts if row.get("split") == "train"),
        key=lambda row: (row["category"], row["image_id"]),
    )
    counts = Counter(row["category"] for row in train_facts)
    if counts != Counter({category: 10 for category in CATEGORIES}):
        raise ValueError(f"expected ten training images per category, got {dict(counts)}")

    unique = []
    for fact in train_facts:
        for mode in MODES:
            question, answer = _prompt_answer(fact, mode)
            unique.append(
                {
                    "id": f"{fact['image_id']}--{mode}",
                    "split": "train",
                    "category": fact["category"],
                    "mode": mode,
                    "source_image_id": fact["image_id"],
                    "image_relative_path": fact["image_relative_path"],
                    "image_sha256": fact["sha256"],
                    "task_type": fact["category"],
                    "prompt": f"<image>\n{question}",
                    "user_instruction": question,
                    "reference_answer": answer,
                    "target_answer": answer,
                    "source": fact["source_page_url"],
                    "license": fact["license"],
                    "construction_method": "human-reviewed-source-fact+grounded-template-v2",
                }
            )

    primary = [row for row in unique if row["mode"] in {"direct", "correction"}]
    weighted = unique + [dict(row) for row in primary]
    return unique, weighted


def to_llamafactory_records(records: list[dict]) -> list[dict]:
    return [
        {
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["target_answer"]},
            ],
            "images": [row["image_relative_path"]],
        }
        for row in records
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_grounded_outputs(facts: list[dict], output_dir: Path) -> dict:
    unique, weighted = build_grounded_records(facts)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "week5_vlm_train_grounded_internal.json": unique,
        "week5_vlm_train_grounded.json": to_llamafactory_records(unique),
        "week5_vlm_train_grounded_weighted.json": to_llamafactory_records(weighted),
    }
    for name, payload in outputs.items():
        (output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    manifest = {
        "schema_version": "1.0",
        "status": "SELECTED_V6_TRAINING_INPUT",
        "unique_training_records": len(unique),
        "weighted_training_samples": len(weighted),
        "sampling_policy": {
            "direct": 2,
            "correction": 2,
            "structured": 1,
            "explanation": 1,
        },
        "files": {
            name: {
                "sha256": _sha256(output_dir / name),
                "bytes": (output_dir / name).stat().st_size,
            }
            for name in outputs
        },
    }
    (output_dir / "training_input_manifest_v6.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    facts = json.loads(args.facts.read_text(encoding="utf-8"))
    write_grounded_outputs(facts, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
