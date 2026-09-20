from __future__ import annotations

import importlib.util
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = SOURCE_ROOT / "scripts" / "validate_week4_dataset.py"


def _module():
    spec = importlib.util.spec_from_file_location("validate_week4_dataset", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_day18_artifacts_pass_combined_validation() -> None:
    result = _module().validate_artifacts(SOURCE_ROOT)
    assert result["valid"] is True
    assert result["errors"] == []
    assert result["record_count"] == 710
    assert result["train_count"] == 639
    assert result["validation_count"] == 71
    assert result["records_over_cutoff"] == 0
    assert result["llamafactory_schema_status"] == "pass"
    assert result["autodl_shutdown_confirmed"] is True
