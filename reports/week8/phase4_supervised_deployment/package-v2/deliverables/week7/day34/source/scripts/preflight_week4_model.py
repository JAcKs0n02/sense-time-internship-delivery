#!/usr/bin/env python3
"""Validate the Week 4 merged-model lineage before any quantization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from week7_deployment.hashing import portable_repo_path, sha256_file
from week7_deployment.manifests import WEEK4_DPO_IDENTITY, validate_model_manifest


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "deliverables/week4/day20/source/results/corrective_final/merged_model_manifest.json"
)


def _load_manifest(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("model manifest must be a JSON object")
    return value


def _verify_remote_files(
    manifest: dict[str, object], model_dir: Path
) -> list[str]:
    errors: list[str] = []
    files = manifest.get("files")
    if not isinstance(files, list):
        return ["manifest files list is missing"]
    for index, record in enumerate(files):
        if not isinstance(record, dict):
            errors.append(f"files[{index}] is not an object")
            continue
        target = model_dir / str(record.get("path", ""))
        if not target.is_file():
            errors.append(f"missing file: {target}")
            continue
        if target.stat().st_size != record.get("bytes"):
            errors.append(f"byte mismatch: {target}")
        if sha256_file(target) != record.get("sha256"):
            errors.append(f"sha256 mismatch: {target}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-dir", type=Path, default=Path(WEEK4_DPO_IDENTITY.model_dir))
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    errors = validate_model_manifest(manifest, WEEK4_DPO_IDENTITY)
    if not args.validate_config_only:
        errors.extend(_verify_remote_files(manifest, args.model_dir))
    receipt = {
        "schema_version": "1.0",
        "status": "PASS" if not errors else "FAIL",
        "scope": "manifest_only" if args.validate_config_only else "remote_files",
        "manifest": portable_repo_path(args.manifest, REPO_ROOT),
        "model_dir": args.model_dir.as_posix(),
        "expected_model_id": WEEK4_DPO_IDENTITY.model_id,
        "expected_manifest_sha256": WEEK4_DPO_IDENTITY.manifest_sha256,
        "errors": errors,
    }
    text = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
