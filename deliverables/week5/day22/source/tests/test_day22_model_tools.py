from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_day22_remote.py"
RUNNER = Path(__file__).parents[1] / "scripts" / "run_day22_remote.sh"


def load_module():
    assert SCRIPT.is_file(), f"missing remote preparation script: {SCRIPT}"
    spec = importlib.util.spec_from_file_location("prepare_day22_remote", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_fake_model(root: Path) -> None:
    files = {
        "config.json": json.dumps(
            {
                "architectures": ["Qwen2VLForConditionalGeneration"],
                "model_type": "qwen2_vl",
            }
        ).encode(),
        "generation_config.json": b"{}",
        "preprocessor_config.json": b"{}",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
        "model-00001-of-00002.safetensors": b"first shard",
        "model-00002-of-00002.safetensors": b"second shard",
    }
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    index = {
        "metadata": {"total_size": 22},
        "weight_map": {
            "layer.0": "model-00001-of-00002.safetensors",
            "layer.1": "model-00002-of-00002.safetensors",
        },
    }
    (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")


def test_revision_must_be_an_immutable_hugging_face_commit():
    module = load_module()
    revision = "eed13092ef92e448dd6875b2a00151bd3f7db0ac"
    assert module.validate_revision(revision) == revision
    for invalid in ("main", "eed13092", "g" * 40, ""):
        with pytest.raises(ValueError, match="40-character lowercase hexadecimal"):
            module.validate_revision(invalid)


def test_nvidia_smi_parser_preserves_gpu_and_vram_facts():
    module = load_module()
    row = "NVIDIA GeForce RTX 3090, 24576, 23800, 776, 570.124.06"
    parsed = module.parse_nvidia_smi(row)
    assert parsed == {
        "name": "NVIDIA GeForce RTX 3090",
        "memory_total_mib": 24576,
        "memory_free_mib": 23800,
        "memory_used_mib": 776,
        "driver_version": "570.124.06",
    }


def test_model_manifest_recomputes_shards_bytes_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = load_module()
    model_dir = tmp_path / "model"
    write_fake_model(model_dir)
    revision = "eed13092ef92e448dd6875b2a00151bd3f7db0ac"
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")

    manifest = module.build_model_manifest(
        model_dir=model_dir,
        repository="Qwen/Qwen2-VL-7B-Instruct",
        revision=revision,
    )

    assert manifest["status"] == "downloaded_verified"
    assert manifest["repository"] == "Qwen/Qwen2-VL-7B-Instruct"
    assert manifest["revision"] == revision
    assert manifest["download_transport_endpoint"] == "https://hf-mirror.com"
    assert manifest["canonical_repository_url"] == (
        "https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/tree/" + revision
    )
    assert manifest["model_type"] == "qwen2_vl"
    assert manifest["weight_shards"] == 2
    assert manifest["weight_bytes"] == len(b"first shard") + len(b"second shard")
    assert manifest["file_count"] == 8
    assert manifest["total_bytes"] == sum(path.stat().st_size for path in model_dir.iterdir())
    by_path = {row["relative_path"]: row for row in manifest["files"]}
    first = model_dir / "model-00001-of-00002.safetensors"
    assert by_path[first.name]["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert len(manifest["files_digest_sha256"]) == 64


def test_model_manifest_rejects_missing_weight_shard(tmp_path: Path):
    module = load_module()
    model_dir = tmp_path / "model"
    write_fake_model(model_dir)
    (model_dir / "model-00002-of-00002.safetensors").unlink()

    with pytest.raises(ValueError, match="missing weight shards"):
        module.build_model_manifest(
            model_dir=model_dir,
            repository="Qwen/Qwen2-VL-7B-Instruct",
            revision="eed13092ef92e448dd6875b2a00151bd3f7db0ac",
        )


def test_remote_runner_is_pinned_and_syntax_valid():
    assert RUNNER.is_file(), f"missing remote runner: {RUNNER}"
    completed = subprocess.run(
        ["bash", "-n", str(RUNNER)], text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    text = RUNNER.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "Qwen/Qwen2-VL-7B-Instruct" in text
    assert "eed13092ef92e448dd6875b2a00151bd3f7db0ac" in text
    assert "qwen-vl-utils==0.0.14" in text
    assert "export HF_ENDPOINT=https://hf-mirror.com" in text
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in text
    assert "--max-pixels 301056" in text
    assert "--max-new-tokens 32" in text
    assert "--max-pixels 1003520" not in text
    assert "PREFLIGHT_MIN_FREE_GIB=22" in text
    assert "PREFLIGHT_MIN_FREE_GIB=10" in text
    assert '--min-free-gib "${PREFLIGHT_MIN_FREE_GIB}"' in text
    assert re.search(r'prepare_day22_remote\.py"?\s+preflight', text)
    assert re.search(r'prepare_day22_remote\.py"?\s+download', text)
    assert "run_vlm_smoke.py" in text
