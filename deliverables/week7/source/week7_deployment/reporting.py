"""Evidence-bound Week 7 status and Markdown-link validation."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path


LOCAL_REQUIRED_PATHS = (
    "docs/week7_execution_plan.md",
    "deliverables/week7/README.md",
    "deliverables/week7/day34/README.md",
    "deliverables/week7/day34/configs/awq_config.json",
    "deliverables/week7/day34/source/scripts/quantize_awq.py",
    "deliverables/week7/day35/README.md",
    "deliverables/week7/day35/configs/week4_dpo_gptq.yaml",
    "deliverables/week7/day35/configs/benchmark_config.json",
    "deliverables/week7/day35/source/scripts/benchmark_model.py",
    "deliverables/week7/day36/README.md",
    "deliverables/week7/day36/source/scripts/start_text_server.sh",
    "deliverables/week7/day36/source/scripts/start_vlm_server.sh",
    "deliverables/week7/day36/source/scripts/chat_client.py",
    "deliverables/week7/day37/README.md",
    "deliverables/week7/day37/app.py",
    "deliverables/week7/day38/README.md",
    "deliverables/week7/day38/OPTIMIZATION_NOTES.md",
    "deliverables/week7/day38/source/scripts/multimodal_client.py",
    "deliverables/week7/day39/README.md",
    "deliverables/week7/day39/REPORT.md",
    "deliverables/week7/day39/LOCAL_DEPLOYMENT_GUIDE.md",
    "deliverables/week7/day39/.env.example",
    "deliverables/week7/day39/source/scripts/validate_week7.py",
)

REMOTE_REQUIRED_RECEIPTS = (
    "deliverables/week7/day34/source/results/llamafactory_awq_compatibility_runtime.json",
    "deliverables/week7/day34/source/results/awq_model_manifest.json",
    "deliverables/week7/day34/source/results/awq_load_smoke.json",
    "deliverables/week7/day35/source/results/bf16.json",
    "deliverables/week7/day35/source/results/awq.json",
    "deliverables/week7/day35/source/results/gptq.json",
    "deliverables/week7/day35/source/results/quantization_acceptance.json",
    "deliverables/week7/day36/source/results/text_service_validation.json",
    "deliverables/week7/day37/source/results/gradio_text_validation.json",
    "deliverables/week7/day38/source/results/multimodal_validation.json",
    "deliverables/week7/day38/source/results/demo_recording_manifest.json",
)


def _remote_status(path: Path) -> str:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "INVALID"
    if not isinstance(value, dict) or value.get("status") not in {"PASS", "FAIL"}:
        return "INVALID"
    return str(value["status"])


def validate_week7(repo_root: Path, *, require_remote: bool) -> dict[str, object]:
    """Return fail-closed local/remote status without creating evidence."""
    local_errors = [
        relative
        for relative in LOCAL_REQUIRED_PATHS
        if not (repo_root / relative).is_file()
    ]
    missing_remote = [
        relative
        for relative in REMOTE_REQUIRED_RECEIPTS
        if not (repo_root / relative).is_file()
    ]
    failed_remote: list[str] = []
    invalid_remote: list[str] = []
    for relative in REMOTE_REQUIRED_RECEIPTS:
        path = repo_root / relative
        if not path.is_file():
            continue
        status = _remote_status(path)
        if status == "FAIL":
            failed_remote.append(relative)
        elif status != "PASS":
            invalid_remote.append(relative)

    if local_errors:
        status = "INCOMPLETE"
    elif failed_remote or invalid_remote:
        status = "FAIL"
    elif missing_remote:
        status = "INCOMPLETE" if require_remote else "READY_FOR_REMOTE"
    else:
        status = "PASS"
    return {
        "schema_version": "1.0",
        "status": status,
        "require_remote": require_remote,
        "local_errors": local_errors,
        "missing_remote_evidence": missing_remote,
        "failed_remote_evidence": failed_remote,
        "invalid_remote_evidence": invalid_remote,
    }


_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def find_broken_markdown_links(
    repo_root: Path, markdown_paths: Sequence[Path]
) -> list[str]:
    """Return broken repository-relative Markdown links from selected files."""
    broken: list[str] = []
    for markdown_path in markdown_paths:
        if not markdown_path.is_file():
            broken.append(f"missing markdown: {markdown_path}")
            continue
        text = markdown_path.read_text(encoding="utf-8")
        for raw_target in _MARKDOWN_LINK.findall(text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            candidate = (markdown_path.parent / target).resolve()
            try:
                candidate.relative_to(repo_root.resolve())
            except ValueError:
                broken.append(f"{markdown_path}: outside repository: {raw_target}")
                continue
            if not candidate.exists():
                broken.append(f"{markdown_path}: missing target: {raw_target}")
    return broken
