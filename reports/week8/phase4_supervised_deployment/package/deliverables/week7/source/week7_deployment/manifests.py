"""Strict model-lineage contracts for Week 7 quantization."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    model_dir: str
    manifest_sha256: str
    weight_bytes: int
    weight_file_count: int
    model_type: str


WEEK4_DPO_IDENTITY = ModelIdentity(
    model_id="reward_corrective_40step_merged",
    model_dir=(
        "/root/autodl-tmp/qwen25-week4/best_model/"
        "qwen25-7b-week4-dpo-corrective-merged"
    ),
    manifest_sha256=(
        "aca9a2932ed427f8fc6e9962e70274eb286a09c446d177af2bc00f6df1675e4c"
    ),
    weight_bytes=15_231_271_872,
    weight_file_count=4,
    model_type="qwen2",
)


def validate_model_manifest(
    manifest: Mapping[str, object], expected: ModelIdentity
) -> list[str]:
    """Return every mismatch instead of accepting a partial lineage match."""
    errors: list[str] = []
    required = {
        "valid": True,
        "model_dir": expected.model_dir,
        "model_type": expected.model_type,
        "weight_file_count": expected.weight_file_count,
        "weight_bytes": expected.weight_bytes,
        "adapter_weight_present": False,
        "manifest_sha256": expected.manifest_sha256,
    }
    for field, wanted in required.items():
        if manifest.get(field) != wanted:
            errors.append(
                f"{field}: expected {wanted!r}, received {manifest.get(field)!r}"
            )

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        errors.append("files: non-empty list required")
        return errors
    for index, record in enumerate(files):
        if not isinstance(record, Mapping):
            errors.append(f"files[{index}]: object required")
            continue
        path = record.get("path")
        size = record.get("bytes")
        digest = record.get("sha256")
        if not isinstance(path, str) or not path:
            errors.append(f"files[{index}].path: non-empty string required")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            errors.append(f"files[{index}].bytes: non-negative integer required")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            errors.append(f"files[{index}].sha256: 64 lowercase hex required")
    return errors
