from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[5]
WEEK6_SOURCE = REPO_ROOT / "deliverables/week6/source"
sys.path.insert(0, str(WEEK6_SOURCE))


def _frozen_case(case_id: str, expected: list[str]) -> dict[str, object]:
    return {
        "id": case_id,
        "success_predicates": {"expected_tool_sequence": expected},
    }


def _case_result(
    case_id: str,
    *,
    actions: list[str],
    checks: dict[str, bool | None],
    observations: list[dict[str, object]] | None = None,
    final_answer: str = "done",
) -> dict[str, object]:
    rows = []
    for index, action in enumerate(actions):
        observation = (
            observations[index]
            if observations is not None
            else {"status": "success", "data": {}, "error_code": None}
        )
        rows.append(
            {
                "action": action,
                "action_input": {"value": index},
                "observation": observation,
            }
        )
    return {
        "case_id": case_id,
        "checks": checks,
        "passed": all(value is not False for value in checks.values()),
        "raw_trace": {"trace": rows, "final_answer": final_answer},
    }


def test_wrong_tool_is_primary_and_completion_is_secondary() -> None:
    """Catches a wrong route being hidden under the downstream completion failure."""
    from week6_agent.day32_analysis import classify_failure

    result = _case_result(
        "route",
        actions=["knowledge_retrieval"],
        checks={
            "first_tool_choice": False,
            "full_tool_sequence": False,
            "argument_correctness": None,
            "completion": False,
            "no_loops": True,
            "no_hallucinated_observations": True,
        },
    )

    classified = classify_failure(_frozen_case("route", ["calculator"]), result)

    assert classified["primary_label"] == "wrong_tool"
    assert classified["actual_tool_sequence"] == ["knowledge_retrieval"]
    assert classified["expected_tool_sequence"] == ["calculator"]
    assert "premature_answer" not in classified["secondary_labels"]


def test_argument_error_preserves_exact_input_failure() -> None:
    """Catches parameter corruption being mislabeled as a successful tool route."""
    from week6_agent.day32_analysis import classify_failure

    result = _case_result(
        "argument",
        actions=["code_executor"],
        checks={
            "first_tool_choice": True,
            "full_tool_sequence": True,
            "argument_correctness": False,
            "completion": True,
            "no_loops": True,
            "no_hallucinated_observations": True,
        },
    )

    classified = classify_failure(_frozen_case("argument", ["code_executor"]), result)

    assert classified["primary_label"] == "argument_extraction_error"


def test_dead_loop_has_priority_over_route_errors() -> None:
    """Catches repeated action/input pairs being diluted by other failed checks."""
    from week6_agent.day32_analysis import classify_failure

    result = _case_result(
        "loop",
        actions=["calculator", "calculator"],
        checks={
            "first_tool_choice": True,
            "full_tool_sequence": False,
            "argument_correctness": False,
            "completion": False,
            "no_loops": False,
            "no_hallucinated_observations": True,
        },
    )
    result["raw_trace"]["trace"][1]["action_input"] = {"value": 0}

    classified = classify_failure(_frozen_case("loop", ["calculator"]), result)

    assert classified["primary_label"] == "dead_loop"
    assert "wrong_tool" in classified["secondary_labels"]
    assert "argument_extraction_error" in classified["secondary_labels"]


def test_rejected_observation_followed_by_false_success_is_unhandled_error() -> None:
    """Catches a rejected tool result being presented as a successful final answer."""
    from week6_agent.day32_analysis import classify_failure

    result = _case_result(
        "rejected",
        actions=["code_executor"],
        observations=[
            {
                "status": "rejected",
                "data": {"risk_nodes": ["Call"]},
                "error_code": "risk_nodes_detected",
            }
        ],
        final_answer="代码没有风险。",
        checks={
            "first_tool_choice": True,
            "full_tool_sequence": True,
            "argument_correctness": None,
            "completion": False,
            "no_loops": True,
            "no_hallucinated_observations": True,
        },
    )

    classified = classify_failure(_frozen_case("rejected", ["code_executor"]), result)

    assert classified["primary_label"] == "tool_error_unhandled"
    assert classified["observation_outcomes"] == [
        {"status": "rejected", "error_code": "risk_nodes_detected"}
    ]


def test_unresolved_tool_call_is_format_parse_error() -> None:
    """Catches truncated tool JSON being reported as a normal final answer."""
    from week6_agent.day32_analysis import classify_failure

    result = _case_result(
        "format",
        actions=[],
        final_answer='<tool_call>\n{"name":"calculator","arguments":',
        checks={
            "first_tool_choice": False,
            "full_tool_sequence": False,
            "argument_correctness": None,
            "completion": False,
            "no_loops": True,
            "no_hallucinated_observations": True,
        },
    )

    classified = classify_failure(_frozen_case("format", ["calculator"]), result)

    assert classified["primary_label"] == "format_parse_error"
    assert "wrong_tool" in classified["secondary_labels"]


def test_successful_tool_with_incomplete_final_response_has_explicit_label() -> None:
    """Catches a response-contract failure being left as an opaque unknown failure."""
    from week6_agent.day32_analysis import classify_failure

    result = _case_result(
        "final",
        actions=["calculator"],
        final_answer="2",
        checks={
            "first_tool_choice": True,
            "full_tool_sequence": True,
            "argument_correctness": None,
            "completion": False,
            "no_loops": True,
            "no_hallucinated_observations": True,
        },
    )

    classified = classify_failure(_frozen_case("final", ["calculator"]), result)

    assert classified["primary_label"] == "final_answer_error"


def test_v2_semantic_and_observation_checks_use_specific_failure_labels() -> None:
    """Catches v2 semantic failures being collapsed into unknown_failure."""
    from week6_agent.day32_analysis import classify_failure

    semantic = _case_result(
        "semantic",
        actions=["calculator"],
        checks={
            "first_tool_choice": True,
            "full_tool_sequence": True,
            "argument_correctness": True,
            "observation_integrity": True,
            "semantic_correctness": False,
            "final_answer_grounded": False,
            "completion": True,
            "no_loops": True,
        },
    )
    semantic["passed"] = False
    observation = copy.deepcopy(semantic)
    observation["case_id"] = "observation"
    observation["checks"]["observation_integrity"] = False

    semantic_label = classify_failure(
        _frozen_case("semantic", ["calculator"]), semantic
    )
    observation_label = classify_failure(
        _frozen_case("observation", ["calculator"]), observation
    )

    assert semantic_label["primary_label"] == "final_answer_error"
    assert observation_label["primary_label"] == "final_answer_error"
    assert "hallucinated_observation" in observation_label["secondary_labels"]


def test_summary_counts_unique_failures_and_zero_occurrence_categories() -> None:
    """Catches multi-label occurrences being mistaken for the number of failed cases."""
    from week6_agent.day32_analysis import analyze_evaluation

    frozen = [_frozen_case("ok", ["calculator"]), _frozen_case("bad", ["calculator"])]
    checks = {
        "first_tool_choice": True,
        "full_tool_sequence": True,
        "argument_correctness": True,
        "completion": True,
        "no_loops": True,
        "no_hallucinated_observations": True,
    }
    report = {
        "status": "completed",
        "case_count": 2,
        "case_results": [
            _case_result("ok", actions=["calculator"], checks=checks),
            _case_result(
                "bad",
                actions=["calculator", "calculator"],
                checks={**checks, "full_tool_sequence": False, "no_loops": False},
            ),
        ],
    }
    report["case_results"][1]["raw_trace"]["trace"][1]["action_input"] = {"value": 0}

    analysis = analyze_evaluation(frozen, report)

    assert analysis["case_count"] == 2
    assert analysis["unique_failed_cases"] == 1
    assert analysis["primary_label_counts"]["dead_loop"] == 1
    assert analysis["primary_label_counts"]["wrong_tool"] == 0
    assert analysis["label_occurrence_counts"]["wrong_tool"] == 1


def test_comparison_rejects_changed_eval_or_generation_protocol() -> None:
    """Catches an apparent optimization produced by changing the evaluation inputs."""
    from week6_agent.day32_analysis import compare_evaluations

    baseline = {
        "case_count": 30,
        "frozen_eval_sha256": "a" * 64,
        "generation_config": {"seed": 42, "do_sample": False},
        "model_identity_sha256": "b" * 64,
        "metrics": {"completion": 0.8},
        "unique_failed_cases": 6,
        "case_analysis": [{"case_id": "x", "passed": False}],
    }
    optimized = copy.deepcopy(baseline)
    optimized["frozen_eval_sha256"] = "c" * 64

    with pytest.raises(ValueError, match="frozen evaluation"):
        compare_evaluations(baseline, optimized)


def test_comparison_reports_improvements_and_regressions() -> None:
    """Catches aggregate gains hiding cases that became newly incorrect."""
    from week6_agent.day32_analysis import compare_evaluations

    common = {
        "case_count": 2,
        "frozen_eval_sha256": "a" * 64,
        "generation_config": {"seed": 42, "do_sample": False},
        "model_identity_sha256": "b" * 64,
    }
    baseline = {
        **common,
        "metrics": {"completion": 0.5},
        "unique_failed_cases": 1,
        "case_analysis": [
            {"case_id": "fixed", "passed": False},
            {"case_id": "regressed", "passed": True},
        ],
    }
    optimized = {
        **common,
        "metrics": {"completion": 0.5},
        "unique_failed_cases": 1,
        "case_analysis": [
            {"case_id": "fixed", "passed": True},
            {"case_id": "regressed", "passed": False},
        ],
    }

    comparison = compare_evaluations(baseline, optimized)

    assert comparison["improved_cases"] == ["fixed"]
    assert comparison["regression_cases"] == ["regressed"]
    assert comparison["metric_deltas"]["completion"] == 0.0


def test_analysis_cli_recomputes_the_six_real_day31_failures(tmp_path: Path) -> None:
    """Catches a report script that substitutes planned counts for raw Day31 traces."""
    script = REPO_ROOT / "deliverables/week6/day32/source/scripts/analyze_failures.py"
    frozen = REPO_ROOT / "deliverables/week6/day31/source/data/tool_sft_eval_frozen.json"
    baseline = REPO_ROOT / "deliverables/week6/day31/source/results/post_eval.json"
    output_dir = tmp_path / "analysis"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--frozen",
            str(frozen),
            "--baseline",
            str(baseline),
            "--output-dir",
            str(output_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    analysis = json.loads((output_dir / "baseline_error_analysis.json").read_text())
    assert analysis["unique_failed_cases"] == 6
    assert analysis["primary_label_counts"] == {
        "wrong_tool": 2,
        "argument_extraction_error": 2,
        "dead_loop": 0,
            "tool_error_unhandled": 1,
            "premature_answer": 0,
            "final_answer_error": 0,
            "hallucinated_observation": 0,
        "format_parse_error": 1,
        "unknown_failure": 0,
    }


def test_analysis_cli_compares_optimized_run_without_hiding_regressions(
    tmp_path: Path,
) -> None:
    """Catches the CLI omitting paired case improvements and regressions."""
    script = REPO_ROOT / "deliverables/week6/day32/source/scripts/analyze_failures.py"
    frozen = REPO_ROOT / "deliverables/week6/day31/source/data/tool_sft_eval_frozen.json"
    baseline_path = REPO_ROOT / "deliverables/week6/day31/source/results/post_eval.json"
    optimized = json.loads(baseline_path.read_text(encoding="utf-8"))
    fixed = next(row for row in optimized["case_results"] if row["case_id"] == "eval-budget-08")
    fixed["checks"]["argument_correctness"] = True
    fixed["passed"] = True
    optimized["metrics"]["argument_correctness"] = 23 / 24
    optimized_path = tmp_path / "optimized.json"
    optimized_path.write_text(json.dumps(optimized), encoding="utf-8")
    output_dir = tmp_path / "analysis"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--frozen",
            str(frozen),
            "--baseline",
            str(baseline_path),
            "--optimized",
            str(optimized_path),
            "--output-dir",
            str(output_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    comparison = json.loads((output_dir / "prompt_comparison.json").read_text())
    assert comparison["improved_cases"] == ["eval-budget-08"]
    assert comparison["regression_cases"] == []
    assert comparison["optimized_unique_failed_cases"] == 5
