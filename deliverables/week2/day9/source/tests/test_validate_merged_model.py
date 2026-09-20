import json
from pathlib import Path
import subprocess
import sys


DAY9_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = DAY9_ROOT / "source" / "scripts" / "validate_merged_model.py"


def run_validation(
    tmp_path: Path,
    *,
    model_files: dict[str, bytes],
    minimum_weight_bytes: int = 10,
) -> subprocess.CompletedProcess[str]:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen2"}),
        encoding="utf-8",
    )
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "qwen"}),
        encoding="utf-8",
    )
    for name, content in model_files.items():
        (model_dir / name).write_bytes(content)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model-dir",
            str(model_dir),
            "--inventory-output",
            str(tmp_path / "inventory.txt"),
            "--sha256-output",
            str(tmp_path / "model.sha256"),
            "--report-output",
            str(tmp_path / "validation.json"),
            "--minimum-weight-bytes",
            str(minimum_weight_bytes),
        ],
        capture_output=True,
        text=True,
    )


def test_accepts_full_model_weight_shards_and_writes_inventory(
    tmp_path: Path,
) -> None:
    result = run_validation(
        tmp_path,
        model_files={
            "model-00001-of-00002.safetensors": b"a" * 8,
            "model-00002-of-00002.safetensors": b"b" * 8,
            "model.safetensors.index.json": b"{}",
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(
        (tmp_path / "validation.json").read_text(encoding="utf-8")
    )
    assert report["valid"] is True
    assert report["model_type"] == "qwen2"
    assert report["weight_file_count"] == 2
    assert report["weight_bytes"] == 16
    inventory = (tmp_path / "inventory.txt").read_text(encoding="utf-8")
    assert "model-00001-of-00002.safetensors" in inventory
    hashes = (tmp_path / "model.sha256").read_text(encoding="utf-8")
    assert "config.json" in hashes


def test_rejects_adapter_only_directory(tmp_path: Path) -> None:
    result = run_validation(
        tmp_path,
        model_files={"adapter_model.safetensors": b"a" * 32},
    )

    assert result.returncode != 0
    assert "no full model weight files" in result.stderr


def test_rejects_implausibly_small_full_model(tmp_path: Path) -> None:
    result = run_validation(
        tmp_path,
        model_files={"model.safetensors": b"a" * 4},
        minimum_weight_bytes=10,
    )

    assert result.returncode != 0
    assert "below minimum" in result.stderr
