from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
WEEK6_SOURCE = REPO_ROOT / "deliverables/week6/source"
sys.path.insert(0, str(WEEK6_SOURCE))


def test_valid_source_reports_ast_metrics_without_execution():
    from week6_agent.tools.code_executor import CodeExecutor

    result = CodeExecutor().inspect("total = 699 + 20\nis_over_budget = total > 1000")

    assert result.status == "success"
    assert result.error_code is None
    assert result.data["syntax_valid"] is True
    assert result.data["risk_nodes"] == []
    assert result.data["node_count"] > 0
    assert result.data["max_depth"] > 0


def test_syntax_error_has_location_and_message():
    from week6_agent.tools.code_executor import CodeExecutor

    result = CodeExecutor().inspect("if True print('x')")

    assert result.status == "rejected"
    assert result.error_code == "syntax_error"
    assert result.data["syntax_valid"] is False
    assert result.data["error_line"] == 1
    assert isinstance(result.data["error_column"], int)
    assert result.data["error_message"]


def test_risk_nodes_are_sorted_and_deduplicated():
    from week6_agent.tools.code_executor import CodeExecutor

    result = CodeExecutor().inspect("import os\nos.path.join('a', 'b')")

    assert result.status == "rejected"
    assert result.error_code == "risk_nodes_detected"
    assert result.data["syntax_valid"] is True
    assert result.data["risk_nodes"] == ["Attribute", "Call", "Import"]


def test_code_executor_reports_syntax_without_execution(tmp_path: Path):
    from week6_agent.tools.code_executor import CodeExecutor

    marker = tmp_path / "must_not_exist"
    result = CodeExecutor().inspect(f"open({str(marker)!r}, 'w').write('x')")

    assert result.status == "rejected"
    assert result.error_code == "risk_nodes_detected"
    assert not marker.exists()


def test_length_and_ast_limits_are_rejected():
    from week6_agent.tools.code_executor import CodeExecutor, MAX_CODE_LENGTH

    too_long = CodeExecutor().inspect("x" * (MAX_CODE_LENGTH + 1))
    too_many_nodes = CodeExecutor().inspect("values = [" + ",".join("1" for _ in range(200)) + "]")

    assert too_long.status == "rejected"
    assert too_long.error_code == "limit_exceeded"
    assert too_long.data["limit"] == "source_length"
    assert too_many_nodes.status == "rejected"
    assert too_many_nodes.error_code == "limit_exceeded"
    assert too_many_nodes.data["limit"] == "ast_node_count"


def test_langchain_wrapper_uses_stable_name_and_serializable_contract():
    from week6_agent.tools.code_executor import CodeExecutor

    tool = CodeExecutor().as_langchain_tool()
    result = tool.invoke({"source": "answer = 6 * 7"})

    assert tool.name == "code_executor"
    assert result["status"] == "success"
    assert result["data"]["syntax_valid"] is True
    assert result["error_code"] is None
