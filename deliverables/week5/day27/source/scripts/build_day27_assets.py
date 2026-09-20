#!/usr/bin/env python3
"""Build the traceable Week5 acceptance rows used by Day27 artifacts."""

from __future__ import annotations

import re
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def _status(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def build_acceptance_rows(evidence: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    day22 = evidence["day22"]
    day23 = evidence["day23"]
    day24 = evidence["day24"]
    day25 = evidence["day25"]
    day26 = evidence["day26"]
    day27 = evidence["day27"]

    inference_ok = (
        day22.get("status") == "PASS"
        and day22.get("image_count") == 5
        and day23.get("status") == "PASS"
        and day23.get("record_count") == 25
        and day25.get("status") == "PASS"
        and day25.get("case_count") == 10
    )
    attention_ok = day24.get("status") == "PASS" and day24.get("heatmap_count", 0) >= 3
    quality_ok = (
        day26.get("status") == "PASS"
        and day26.get("stage") == "final"
        and day26.get("case_count") == 20
        and day26.get("mean_gain", float("-inf")) >= 0.35
        and day26.get("direct_gain", float("-inf")) >= 0.0
        and day26.get("opportunity_win_rate", 0.0) >= 0.50
        and day26.get("hallucination_not_worse") is True
    )
    report_ok = day27.get("report_complete") is True and day27.get(
        "model_archive_complete"
    ) is True

    return [
        {
            "requirement_id": "W5-INFERENCE",
            "teacher_requirement": "完成图文推理",
            "status": _status(inference_ok),
            "actual": (
                f"images={day22.get('image_count')}; "
                f"inference_records={day23.get('record_count')}; "
                f"hallucination_cases={day25.get('case_count')}"
            ),
            "evidence": "Day22/Day23/Day25 validated artifacts",
        },
        {
            "requirement_id": "W5-ATTENTION",
            "teacher_requirement": "完成注意力可视化",
            "status": _status(attention_ok),
            "actual": f"heatmaps={day24.get('heatmap_count')}",
            "evidence": "Day24 validation report and heatmaps",
        },
        {
            "requirement_id": "W5-QUALITY",
            "teacher_requirement": "VLM 微调后有明显效果提升",
            "status": _status(quality_ok),
            "actual": (
                f"stage={day26.get('stage')}; case_count={day26.get('case_count')}; "
                f"mean_gain={day26.get('mean_gain', 0.0):.4f}; "
                f"direct_gain={day26.get('direct_gain', 0.0):.4f}; "
                f"opportunity_win_rate={day26.get('opportunity_win_rate', 0.0):.4f}; "
                f"hallucination_not_worse={day26.get('hallucination_not_worse')}"
            ),
            "evidence": "Day26 v8 one-time final blind comparison",
        },
        {
            "requirement_id": "W5-REPORT",
            "teacher_requirement": "第 5 周周报完整",
            "status": _status(report_ok),
            "actual": (
                f"report_complete={day27.get('report_complete')}; "
                f"model_archive_complete={day27.get('model_archive_complete')}"
            ),
            "evidence": "Day27 report, acceptance matrix and final model manifest",
        },
    ]


def validate_model_manifest(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    base = payload.get("base_model", {})
    adapter = payload.get("adapter", {})
    smoke = payload.get("load_smoke", {})
    if payload.get("model_type") != "base_plus_lora_adapter":
        errors.append("model type must be base_plus_lora_adapter")
    if not base.get("repository"):
        errors.append("base repository is missing")
    if not REVISION_RE.fullmatch(str(base.get("revision", ""))):
        errors.append("base revision is missing or invalid")
    if not adapter.get("remote_path"):
        errors.append("adapter remote path is missing")
    if not SHA256_RE.fullmatch(str(adapter.get("weight_sha256", ""))):
        errors.append("adapter weight SHA-256 is missing or invalid")
    if not SHA256_RE.fullmatch(str(adapter.get("config_sha256", ""))):
        errors.append("adapter config SHA-256 is missing or invalid")
    if not isinstance(adapter.get("total_bytes"), int) or adapter.get("total_bytes", 0) <= 0:
        errors.append("adapter byte count is missing or invalid")
    for field in ("training_dataset_sha256", "training_config_sha256"):
        if not SHA256_RE.fullmatch(str(payload.get(field, ""))):
            errors.append(f"{field} is missing or invalid")
    if smoke.get("status") != "PASS" or smoke.get("return_code") != 0:
        errors.append("independent adapter load smoke did not pass")
    return errors
