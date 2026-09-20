#!/usr/bin/env python3
"""Preflight, download, and audit the frozen Qwen2-VL model on AutoDL."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
REQUIRED_MODEL_FILES = {
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
}
CORE_PACKAGES = (
    "accelerate",
    "huggingface-hub",
    "pillow",
    "qwen-vl-utils",
    "safetensors",
    "torch",
    "torchvision",
    "transformers",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_revision(revision: str) -> str:
    if not REVISION_PATTERN.fullmatch(revision):
        raise ValueError("revision must be a 40-character lowercase hexadecimal commit")
    return revision


def parse_nvidia_smi(row: str) -> dict[str, Any]:
    parts = [part.strip() for part in row.strip().split(",")]
    if len(parts) != 5:
        raise ValueError(f"unexpected nvidia-smi row: {row!r}")
    name, total, free, used, driver = parts
    return {
        "name": name,
        "memory_total_mib": int(total),
        "memory_free_mib": int(free),
        "memory_used_mib": int(used),
        "driver_version": driver,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def model_files(model_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(model_dir.rglob("*")):
        relative = path.relative_to(model_dir)
        if ".cache" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError(f"model directory must not contain symlinks: {relative}")
        if path.is_file():
            files.append(path)
    return files


def build_model_manifest(model_dir: Path, repository: str, revision: str) -> dict[str, Any]:
    revision = validate_revision(revision)
    if not model_dir.is_dir():
        raise ValueError(f"model directory does not exist: {model_dir}")
    missing_required = sorted(name for name in REQUIRED_MODEL_FILES if not (model_dir / name).is_file())
    if missing_required:
        raise ValueError(f"missing required model files: {missing_required}")
    config = read_json(model_dir / "config.json")
    if config.get("model_type") != "qwen2_vl":
        raise ValueError(f"unexpected model_type: {config.get('model_type')!r}")
    if "Qwen2VLForConditionalGeneration" not in config.get("architectures", []):
        raise ValueError("Qwen2VLForConditionalGeneration is missing from config architectures")
    index = read_json(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model weight index must contain a non-empty weight_map")
    shard_names = sorted(set(weight_map.values()))
    if any(not isinstance(name, str) or Path(name).name != name for name in shard_names):
        raise ValueError("weight shard names must be direct relative filenames")
    missing_shards = [name for name in shard_names if not (model_dir / name).is_file()]
    if missing_shards:
        raise ValueError(f"missing weight shards: {missing_shards}")
    files: list[dict[str, Any]] = []
    for path in model_files(model_dir):
        files.append(
            {
                "relative_path": path.relative_to(model_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    files_digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    weight_paths = [model_dir / name for name in shard_names]
    return {
        "schema_version": "1.0",
        "status": "downloaded_verified",
        "repository": repository,
        "revision": revision,
        "canonical_repository_url": f"https://huggingface.co/{repository}/tree/{revision}",
        "download_transport_endpoint": os.environ.get(
            "HF_ENDPOINT", "https://huggingface.co"
        ).rstrip("/"),
        "license": "Apache-2.0",
        "model_type": config["model_type"],
        "architectures": config["architectures"],
        "remote_model_path": str(model_dir),
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "weight_shards": len(shard_names),
        "weight_bytes": sum(path.stat().st_size for path in weight_paths),
        "index_declared_weight_bytes": index.get("metadata", {}).get("total_size"),
        "files_digest_sha256": files_digest,
        "files": files,
        "generated_at": utc_now(),
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in CORE_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def query_gpus() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,memory.used,driver_version",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    return [parse_nvidia_smi(row) for row in output.splitlines() if row.strip()]


def torch_environment() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised on AutoDL
        return {"import_ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "import_ok": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "device_count": torch.cuda.device_count(),
    }


def build_preflight(storage_path: Path, min_vram_mib: int, min_free_gib: int) -> dict[str, Any]:
    gpus = query_gpus()
    disk = shutil.disk_usage(storage_path)
    torch_info = torch_environment()
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        text=True,
        capture_output=True,
        check=False,
    )
    errors: list[str] = []
    if not gpus:
        errors.append("no NVIDIA GPU was reported")
    elif max(gpu["memory_total_mib"] for gpu in gpus) < min_vram_mib:
        errors.append(f"GPU total memory is below {min_vram_mib} MiB")
    if disk.free < min_free_gib * 1024**3:
        errors.append(f"free disk is below {min_free_gib} GiB")
    if not torch_info.get("cuda_available"):
        errors.append("PyTorch CUDA is unavailable")
    versions = package_versions()
    for package in ("accelerate", "huggingface-hub", "qwen-vl-utils", "torch", "transformers"):
        if versions[package] is None:
            errors.append(f"required package is missing: {package}")
    if pip_check.returncode != 0:
        errors.append("pip check reported dependency conflicts")
    return {
        "status": "FAIL" if errors else "PASS",
        "checked_at": utc_now(),
        "host": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
        },
        "gpu": gpus,
        "torch": torch_info,
        "storage": {
            "path": str(storage_path),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "packages": versions,
        "pip_check": {
            "returncode": pip_check.returncode,
            "stdout": pip_check.stdout.strip(),
            "stderr": pip_check.stderr.strip(),
        },
        "thresholds": {
            "min_vram_mib": min_vram_mib,
            "min_free_disk_gib": min_free_gib,
        },
        "errors": errors,
    }


def download_and_manifest(
    repository: str, revision: str, model_dir: Path, manifest_output: Path
) -> dict[str, Any]:
    revision = validate_revision(revision)
    from huggingface_hub import HfApi, snapshot_download

    resolved = HfApi().model_info(repository, revision=revision).sha
    if resolved != revision:
        raise ValueError(f"resolved revision mismatch: actual={resolved}, expected={revision}")
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repository,
        revision=revision,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    manifest = build_model_manifest(model_dir, repository, revision)
    manifest["resolved_revision"] = resolved
    write_json(manifest_output, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--storage-path", type=Path, required=True)
    preflight.add_argument("--min-vram-mib", type=int, default=16000)
    preflight.add_argument("--min-free-gib", type=int, default=22)
    preflight.add_argument("--output", type=Path, required=True)
    download = subparsers.add_parser("download")
    download.add_argument("--repository", required=True)
    download.add_argument("--revision", required=True)
    download.add_argument("--model-dir", type=Path, required=True)
    download.add_argument("--output", type=Path, required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--repository", required=True)
    manifest.add_argument("--revision", required=True)
    manifest.add_argument("--model-dir", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "preflight":
            result = build_preflight(args.storage_path, args.min_vram_mib, args.min_free_gib)
            write_json(args.output, result)
            return 0 if result["status"] == "PASS" else 1
        if args.command == "download":
            download_and_manifest(args.repository, args.revision, args.model_dir, args.output)
            return 0
        result = build_model_manifest(args.model_dir, args.repository, args.revision)
        write_json(args.output, result)
        return 0
    except Exception as exc:
        failure = {
            "status": "FAIL",
            "failed_at": utc_now(),
            "command": args.command,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        write_json(args.output, failure)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
