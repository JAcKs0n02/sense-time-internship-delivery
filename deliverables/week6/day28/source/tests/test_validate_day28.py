from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = SOURCE_ROOT / "scripts" / "validate_day28.py"
RESULT = SOURCE_ROOT / "results" / "day28_validation.json"
DATA = SOURCE_ROOT / "data" / "knowledge_base.json"


def run_validator() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=Path(__file__).resolve().parents[5],
        text=True,
        capture_output=True,
        check=False,
    )


def test_day28_validator_runs_from_repository_root_and_records_real_checks():
    """Catches a validator that only imports in-place or writes a fabricated pass."""
    completed = run_validator()

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    knowledge_check = next(check for check in payload["checks"] if check["name"] == "knowledge_base")
    assert knowledge_check["actual"] == {
        "schema_version": "1.0",
        "product_count": 4,
        "declared_products_sha256": payload["knowledge_base_sha256"],
        "recomputed_products_sha256": payload["knowledge_base_sha256"],
    }
    assert {check["name"] for check in payload["checks"]} == {
        "imports",
        "knowledge_base",
        "calculator_safe_case",
        "calculator_unsafe_case",
        "calculator_limit_case",
        "retrieval_alias_case",
        "retrieval_not_found_case",
    }
    assert all(check["status"] == "PASS" for check in payload["checks"])


def test_day28_validator_fails_closed_for_hash_valid_invalid_product_data():
    """Catches schema validation delegated solely to the production retriever."""
    original = DATA.read_text(encoding="utf-8")
    payload = json.loads(original)
    payload["products"][0]["price_cny"] = -1
    canonical_products = json.dumps(
        payload["products"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    import hashlib

    payload["products_sha256"] = hashlib.sha256(canonical_products).hexdigest()
    DATA.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    try:
        completed = run_validator()
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        assert completed.returncode == 1
        assert result["status"] == "FAIL"
        knowledge_check = next(
            check for check in result["checks"] if check["name"] == "knowledge_base"
        )
        assert knowledge_check["status"] == "FAIL"
        assert "invalid_price_cny" in knowledge_check["error"]
    finally:
        DATA.write_text(original, encoding="utf-8")
        recovery = run_validator()
        assert recovery.returncode == 0, recovery.stderr
