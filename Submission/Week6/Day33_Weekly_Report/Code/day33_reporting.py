"""Deterministic helpers for the Week 6 report and acceptance matrix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


ACCEPTANCE_REQUIREMENTS = (
    ("three_tools", "3 个工具可正常调用"),
    ("multistep", "Agent 完成至少 3 步复杂任务"),
    ("error_analysis", "错误分析报告有深度"),
    ("weekly_report", "第 6 周周报完整提交"),
)

V2_READINESS_PATHS = (
    "deliverables/week6/day31/source/data/knowledge_base_v2.json",
    "deliverables/week6/day31/source/data/tool_sft_train_v2.json",
    "deliverables/week6/day31/source/data/tool_sft_dev_v2.json",
    "deliverables/week6/day31/source/data/tool_sft_test_v2.json",
    "deliverables/week6/day31/source/data/dataset_manifest_v2.json",
    "deliverables/week6/day31/source/results/data_validation_v2.json",
    "deliverables/week6/day31/configs/dataset_info_v2.json",
    "deliverables/week6/day31/configs/runtime_versions_v2.json",
    "deliverables/week6/day31/configs/week6_tool_sft_lora_v2.yaml",
    "deliverables/week6/day32/source/configs/prompt_manifest_v2.json",
    "deliverables/week6/day32/source/configs/system_prompt_main.txt",
    "deliverables/week6/day32/source/configs/system_prompt_ablation_input_fidelity.txt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_local_readiness(
    repo_root: Path, *, pytest_count: int, pytest_exit_code: int
) -> dict[str, object]:
    """Build a fail-closed receipt before any formal GPU execution."""
    root = repo_root.resolve()
    artifacts: list[dict[str, object]] = []
    for relative_name in V2_READINESS_PATHS:
        path = root / relative_name
        if not path.is_file():
            raise ValueError(f"required v2 artifact missing: {relative_name}")
        artifacts.append(
            {
                "path": relative_name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest_path = root / "deliverables/week6/day31/source/data/dataset_manifest_v2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(
        (
            root
            / "deliverables/week6/day31/source/results/data_validation_v2.json"
        ).read_text(encoding="utf-8")
    )
    if (
        manifest.get("schema_version") != "2.0"
        or manifest.get("profile") != "week6_agent_formal_v2"
        or validation.get("valid") is not True
    ):
        raise ValueError("v2 dataset manifest is not valid")
    for split in ("train_v2", "dev_v2", "test_v2"):
        record = manifest.get("splits", {}).get(split, {})
        path = manifest_path.parent / str(record.get("file_name", ""))
        if not path.is_file() or record.get("file_sha256") != sha256_file(path):
            raise ValueError(f"v2 dataset split identity mismatch: {split}")
    if pytest_exit_code != 0 or pytest_count < 1:
        raise ValueError("Week6 pytest verification did not pass")
    return {
        "schema_version": "2.0",
        "status": "ready_for_gpu",
        "artifacts": artifacts,
        "pytest": {"passed": pytest_count, "exit_code": pytest_exit_code},
        "metric_namespaces": {
            "raw": "raw_model_metrics",
            "guarded": "guarded_system_metrics",
        },
        "gpu_evidence": {
            "checkpoint_selection": "pending",
            "training": "pending",
            "final_test": "pending",
        },
    }


def build_report_input_manifest(repo_root: Path, paths: Iterable[Path]) -> dict[str, object]:
    root = repo_root.resolve()
    records: list[dict[str, object]] = []
    for path in sorted((Path(value).resolve() for value in paths), key=lambda item: item.as_posix()):
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"report input is outside repository: {path}") from error
        if not path.is_file():
            raise ValueError(f"report input missing: {relative.as_posix()}")
        records.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"schema_version": "1.0", "files": records}


def validate_report_inputs(repo_root: Path, manifest: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    if manifest.get("schema_version") != "1.0":
        failures.append("unsupported manifest schema")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        return failures + ["manifest files must be a non-empty list"]
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            failures.append("invalid manifest row")
            continue
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            failures.append(f"non-portable path: {row['path']}")
            continue
        path = repo_root / relative
        if not path.is_file():
            failures.append(f"missing file: {row['path']}")
        elif row.get("sha256") != sha256_file(path):
            failures.append(f"hash mismatch: {row['path']}")
        elif row.get("bytes") != path.stat().st_size:
            failures.append(f"size mismatch: {row['path']}")
    return failures


def build_acceptance_matrix(
    facts: Mapping[str, Mapping[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for requirement_id, requirement in ACCEPTANCE_REQUIREMENTS:
        if requirement_id not in facts:
            raise ValueError(f"missing acceptance fact: {requirement_id}")
        fact = facts[requirement_id]
        status = fact.get("status")
        if status not in {"PASS", "FAIL"}:
            raise ValueError(f"acceptance status must be PASS or FAIL: {requirement_id}")
        metric = fact.get("metric", "").strip()
        evidence = fact.get("evidence", "").strip()
        if not metric or not evidence:
            raise ValueError(f"acceptance fact incomplete: {requirement_id}")
        rows.append(
            {
                "requirement_id": requirement_id,
                "requirement": requirement,
                "status": status,
                "metric": metric,
                "evidence": evidence,
            }
        )
    return rows
