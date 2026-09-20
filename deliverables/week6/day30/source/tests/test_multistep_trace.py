from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


REPO_ROOT = Path(__file__).resolve().parents[5]
WEEK6_SOURCE = REPO_ROOT / "deliverables/week6/source"
CASES_PATH = REPO_ROOT / "deliverables/week6/day30/source/data/multistep_cases.json"
KNOWLEDGE_BASE = REPO_ROOT / "deliverables/week6/day28/source/data/knowledge_base.json"
sys.path.insert(0, str(WEEK6_SOURCE))


class ScriptedChatModel(BaseChatModel):
    _responses: list[AIMessage] = PrivateAttr()
    _index: int = PrivateAttr(default=0)
    _bound_tool_names: tuple[str, ...] = PrivateAttr(default=())

    def __init__(self, responses: Sequence[AIMessage]) -> None:
        super().__init__()
        self._responses = list(responses)

    @property
    def _llm_type(self) -> str:
        return "synthetic-test-only"

    def bind_tools(self, tools: Sequence[Any], **_: Any) -> "ScriptedChatModel":
        self._bound_tool_names = tuple(tool.name for tool in tools)
        return self

    def _generate(self, messages: list[Any], **_: Any) -> ChatResult:
        del messages
        response = self._responses[self._index]
        self._index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


def _call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    raw = json.dumps(
        {"name": name, "arguments": args},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": call_id, "type": "tool_call"}
        ],
        response_metadata={"raw_generated_text": raw},
    )


def _teacher_case() -> dict[str, Any]:
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    return next(case for case in payload["cases"] if case["case_id"] == "teacher_nebula_budget")


def test_day30_factory_binds_exact_three_tools_and_preserves_day29_factory():
    from week6_agent.agent_factory import build_basic_tools
    from week6_agent.day30_agent import build_day30_tools

    day29_names = tuple(tool.name for tool in build_basic_tools(KNOWLEDGE_BASE))
    day30_names = tuple(tool.name for tool in build_day30_tools(KNOWLEDGE_BASE))

    assert day29_names == ("calculator", "knowledge_retrieval")
    assert day30_names == ("calculator", "knowledge_retrieval", "code_executor")


def test_teacher_multistep_trace_has_retrieval_calculation_and_final_reasoning():
    from week6_agent.day30_agent import build_day30_react_agent
    from week6_agent.day30_validation import validate_case_result

    model = ScriptedChatModel(
        [
            _call("knowledge_retrieval", {"query": "星云机械键盘"}, "call-1"),
            _call("calculator", {"expression": "699 + 20"}, "call-2"),
            AIMessage(
                content="总价 719 元，没有超过 1000 元预算，还剩 281 元。",
                response_metadata={
                    "raw_generated_text": "总价 719 元，没有超过 1000 元预算，还剩 281 元。"
                },
            ),
        ]
    )
    agent = build_day30_react_agent(
        model, KNOWLEDGE_BASE, max_steps=6, runtime_limit_seconds=5
    )
    result = agent.invoke(_teacher_case()["prompt"])
    validation = validate_case_result(_teacher_case(), result)

    assert model._bound_tool_names == (
        "calculator",
        "knowledge_retrieval",
        "code_executor",
    )
    assert [row["action"] for row in result["trace"]] == [
        "knowledge_retrieval",
        "calculator",
    ]
    assert result["trace"][1]["observation"]["data"]["value"] == 719
    assert validation["status"] == "PASS"
    assert validation["auditable_stage_count"] == 3


def test_validator_recomputes_price_and_rejects_claimed_success():
    from week6_agent.day30_validation import validate_case_result

    result = {
        "status": "success",
        "error_code": None,
        "tool_names": ["calculator", "knowledge_retrieval", "code_executor"],
        "trace": [
            {
                "action": "knowledge_retrieval",
                "action_input": {"query": "星云机械键盘"},
                "observation": {
                    "status": "success",
                    "data": {
                        "product_id": "nebula-keyboard",
                        "price_cny": 699,
                        "shipping_cny": 20,
                    },
                    "error_code": None,
                },
            },
            {
                "action": "calculator",
                "action_input": {"expression": "699 + 20"},
                "observation": {
                    "status": "success",
                    "data": {"value": 720},
                    "error_code": None,
                },
            },
        ],
        "final_answer": "总价 719 元，没有超过 1000 元预算。",
    }

    validation = validate_case_result(_teacher_case(), result)

    assert validation["status"] == "FAIL"
    assert validation["checks"]["calculator_value"] is False


def test_frozen_case_set_contains_teacher_variants_and_code_routes():
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    ids = {case["case_id"] for case in payload["cases"]}

    assert payload["frozen_before_formal_run"] is True
    assert len(payload["cases"]) == 5
    assert {
        "teacher_nebula_budget",
        "comet_over_budget",
        "orbit_within_budget",
        "code_ast_valid",
        "code_side_effect_rejected",
    } == ids


def _event(
    action: str,
    action_input: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "step_index": 1,
        "action": action,
        "action_input": action_input,
        "tool_call_id": "synthetic-policy-call",
        "raw_generated_text": "synthetic-policy-only",
        "observation": observation,
    }


def test_guarded_policy_requires_calculator_after_product_retrieval():
    from week6_agent.policy import validate_trace_policy

    trace = [
        _event(
            "knowledge_retrieval",
            {"query": "星云机械键盘"},
            {
                "status": "success",
                "data": {"price_cny": 699, "shipping_cny": 20},
                "error_code": None,
            },
        )
    ]

    result = validate_trace_policy("查询星云机械键盘并计算总价。", trace, "总价 719 元。")

    assert "calculator_required_after_retrieval" in result["violations"]


def test_guarded_policy_requires_calculator_arguments_from_observation():
    from week6_agent.policy import validate_trace_policy

    trace = [
        _event(
            "knowledge_retrieval",
            {"query": "星云机械键盘"},
            {
                "status": "success",
                "data": {"price_cny": 699, "shipping_cny": 20},
                "error_code": None,
            },
        ),
        _event(
            "calculator",
            {"expression": "699 + 0"},
            {"status": "success", "data": {"value": 699}, "error_code": None},
        ),
    ]

    result = validate_trace_policy("查询星云机械键盘并计算总价。", trace, "总价 699 元。")

    assert "calculator_arguments_not_grounded" in result["violations"]


def test_guarded_policy_accepts_zero_shipping_as_explicit_calculation():
    from week6_agent.policy import validate_trace_policy

    trace = [
        _event(
            "knowledge_retrieval",
            {"query": "测试商品"},
            {
                "status": "success",
                "data": {"price_cny": 799, "shipping_cny": 0},
                "error_code": None,
            },
        ),
        _event(
            "calculator",
            {"expression": "799 + 0"},
            {"status": "success", "data": {"value": 799}, "error_code": None},
        ),
    ]

    result = validate_trace_policy("查询测试商品并计算总价。", trace, "总价 799 元。")

    assert result["status"] == "PASS"


def test_guarded_policy_preserves_exact_code_source_and_stops_after_error():
    from week6_agent.policy import validate_trace_policy

    prompt = "请检查以下 Python 源码：\nprint('hello')"
    trace = [
        _event(
            "code_executor",
            {"source": "print('changed')"},
            {"status": "failed", "data": None, "error_code": "syntax_error"},
        ),
        _event(
            "calculator",
            {"expression": "1 + 1"},
            {"status": "success", "data": {"value": 2}, "error_code": None},
        ),
    ]

    result = validate_trace_policy(prompt, trace, "检查完成。")

    assert "code_source_not_preserved" in result["violations"]
    assert "continued_after_tool_error" in result["violations"]


def test_guarded_policy_rejects_malformed_tool_text_as_final_answer():
    from week6_agent.policy import validate_trace_policy

    result = validate_trace_policy(
        "计算 1 + 1",
        [],
        '<tool_call>{"name":"calculator","arguments":{"expression":"1 + 1"}}',
    )

    assert "malformed_tool_output" in result["violations"]


def test_guarded_agent_marks_policy_violation_but_raw_mode_preserves_observation():
    from week6_agent.day30_agent import build_day30_react_agent

    responses = [
        _call("knowledge_retrieval", {"query": "星云机械键盘"}, "call-1"),
        AIMessage(content="总价 719 元。"),
    ]
    raw = build_day30_react_agent(
        ScriptedChatModel(responses),
        KNOWLEDGE_BASE,
        max_steps=6,
        runtime_limit_seconds=5,
        policy_mode="raw",
    ).invoke(_teacher_case()["prompt"])
    guarded = build_day30_react_agent(
        ScriptedChatModel(responses),
        KNOWLEDGE_BASE,
        max_steps=6,
        runtime_limit_seconds=5,
        policy_mode="guarded",
    ).invoke(_teacher_case()["prompt"])

    assert raw["status"] == "success"
    assert raw["policy_status"] == "FAIL"
    assert guarded["status"] == "failed"
    assert guarded["error_code"] == "policy_violation"
    assert guarded["policy_violations"] == ["calculator_required_after_retrieval"]
