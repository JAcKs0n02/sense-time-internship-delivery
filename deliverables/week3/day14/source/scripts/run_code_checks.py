#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable


def extract_python_code(response: str) -> str | None:
    fenced = re.search(r"```(?:python|py)?\s*\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    stripped = response.strip()
    if re.search(r"^\s*(?:async\s+)?def\s+\w+\s*\(", stripped, re.MULTILINE):
        return stripped
    return None


def build_test_program(code: str, assertions: list[str]) -> str:
    literal_assertions = "\n".join(assertions)
    return f"{code}\n\n{literal_assertions}\nprint('DAY14_TESTS_PASSED')\n"


def sandbox_preflight() -> dict[str, Any]:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        return {"available": False, "reason": "bwrap is not installed"}
    sandbox_python = Path("/usr/bin/python3")
    if not sandbox_python.is_file():
        return {"available": False, "reason": "/usr/bin/python3 is unavailable"}
    probe = subprocess.run(
        [bwrap, "--unshare-net", "--ro-bind", "/", "/", "--", "/bin/true"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip() or f"exit code {probe.returncode}"
        return {"available": False, "reason": f"bwrap preflight failed: {detail}"}
    return {
        "available": True,
        "reason": None,
        "executable": bwrap,
        "python": str(sandbox_python),
    }


def _limit_resources() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))


def check_code_response(
    response: str,
    assertions: list[str],
    *,
    sandbox_probe: Callable[[], dict[str, Any]] = sandbox_preflight,
) -> dict[str, Any]:
    code = extract_python_code(response)
    if code is None:
        return {
            "code_execution_status": "not_run_no_code",
            "tests_passed": False,
            "reason": "no Python function found",
        }
    preflight = sandbox_probe()
    if not preflight.get("available"):
        return {
            "code_execution_status": "disabled_no_sandbox",
            "tests_passed": None,
            "reason": preflight.get("reason"),
        }
    bwrap = preflight.get("executable") or shutil.which("bwrap")
    if not bwrap:
        return {
            "code_execution_status": "disabled_no_sandbox",
            "tests_passed": None,
            "reason": "bwrap executable unavailable after preflight",
        }
    program = build_test_program(code, assertions)
    with tempfile.TemporaryDirectory(prefix="day14-code-check-") as temporary:
        root = Path(temporary)
        script = root / "candidate.py"
        script.write_text(program, encoding="utf-8")
        command = [
            str(bwrap),
            "--die-with-parent",
            "--unshare-all",
            "--new-session",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/bin",
            "/bin",
            "--ro-bind",
            str(script),
            "/candidate.py",
            "--tmpfs",
            "/tmp",
            "--chdir",
            "/tmp",
            "--",
            str(preflight.get("python") or "/usr/bin/python3"),
            "-I",
            "/candidate.py",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=3,
                preexec_fn=_limit_resources,
            )
        except subprocess.TimeoutExpired:
            return {"code_execution_status": "timeout", "tests_passed": False, "reason": "3 second timeout"}
    return {
        "code_execution_status": "completed" if completed.returncode == 0 else "failed",
        "tests_passed": completed.returncode == 0 and "DAY14_TESTS_PASSED" in completed.stdout,
        "exit_code": completed.returncode,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-2000:],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated checks for Day 14 code responses.")
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--assertions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assertions = json.loads(args.assertions.read_text(encoding="utf-8"))
    result = check_code_response(args.response.read_text(encoding="utf-8"), assertions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
