import sys
from pathlib import Path


DAY14_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = DAY14_ROOT / "source" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_code_checks import (  # noqa: E402
    build_test_program,
    check_code_response,
    extract_python_code,
)


def test_extracts_fenced_python_without_rewriting_it() -> None:
    response = "说明如下：\n```python\ndef add(a, b):\n    return a + b\n```"

    assert extract_python_code(response) == "def add(a, b):\n    return a + b"


def test_test_program_contains_candidate_code_and_literal_assertions() -> None:
    program = build_test_program(
        "def add(a, b):\n    return a + b",
        ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
    )

    assert "def add(a, b):" in program
    assert "assert add(1, 2) == 3" in program
    assert "DAY14_TESTS_PASSED" in program


def test_code_is_not_executed_when_sandbox_is_unavailable() -> None:
    result = check_code_response(
        "```python\ndef add(a, b):\n    return a + b\n```",
        ["assert add(1, 2) == 3"],
        sandbox_probe=lambda: {"available": False, "reason": "bwrap missing"},
    )

    assert result["code_execution_status"] == "disabled_no_sandbox"
    assert result["tests_passed"] is None
    assert result["reason"] == "bwrap missing"


def test_missing_code_is_reported_without_execution() -> None:
    result = check_code_response(
        "这里只给出文字说明。",
        ["assert add(1, 2) == 3"],
        sandbox_probe=lambda: {"available": True, "reason": None},
    )

    assert result["code_execution_status"] == "not_run_no_code"
    assert result["tests_passed"] is False
