import json
from pathlib import Path
import subprocess
import sys


DAY9_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = DAY9_ROOT / "source" / "scripts" / "validate_export_config.py"


def run_validation(
    tmp_path: Path,
    yaml_text: str,
    *,
    adapter_base_model: str | None = None,
) -> subprocess.CompletedProcess[str]:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    (base / "config.json").write_text(
        json.dumps({"model_type": "qwen2"}),
        encoding="utf-8",
    )
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": adapter_base_model or str(base),
                "peft_type": "LORA",
                "r": 8,
                "lora_alpha": 16,
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "export.yaml"
    config.write_text(
        yaml_text.format(
            base=base,
            adapter=adapter,
            export=tmp_path / "merged",
        ),
        encoding="utf-8",
    )
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config),
            "--report-output",
            str(tmp_path / "validation.json"),
        ],
        capture_output=True,
        text=True,
    )


VALID_CONFIG = """\
model_name_or_path: {base}
adapter_name_or_path: {adapter}
template: qwen
trust_remote_code: true
export_dir: {export}
export_size: 4
export_device: cpu
export_legacy_format: false
"""


def test_accepts_unquantized_lora_export_config(tmp_path: Path) -> None:
    result = run_validation(tmp_path, VALID_CONFIG)

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(
        (tmp_path / "validation.json").read_text(encoding="utf-8")
    )
    assert report["valid"] is True
    assert report["model_type"] == "qwen2"
    assert report["adapter_peft_type"] == "LORA"
    assert report["adapter_rank"] == 8
    assert report["adapter_alpha"] == 16


def test_rejects_quantization_bit_during_lora_merge(tmp_path: Path) -> None:
    result = run_validation(
        tmp_path,
        VALID_CONFIG + "quantization_bit: 4\n",
    )

    assert result.returncode != 0
    assert "quantization_bit must not be set" in result.stderr


def test_rejects_adapter_base_model_mismatch(tmp_path: Path) -> None:
    result = run_validation(
        tmp_path,
        VALID_CONFIG,
        adapter_base_model="/wrong/base",
    )

    assert result.returncode != 0
    assert "adapter base model does not match" in result.stderr
