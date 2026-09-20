from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


REPO_ROOT = Path(__file__).resolve().parents[5]
WEEK6_SOURCE = REPO_ROOT / "deliverables/week6/source"
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
        return "day32-scripted-test"

    def bind_tools(self, tools: Sequence[Any], **_: Any) -> "ScriptedChatModel":
        self._bound_tool_names = tuple(tool.name for tool in tools)
        return self

    def _generate(self, messages: list[Any], **_: Any) -> ChatResult:
        del messages
        response = self._responses[self._index]
        self._index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


def _call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
        response_metadata={"raw_generated_text": json.dumps({"name": name, "arguments": args})},
    )


def test_prompt_manifest_binds_exact_day30_baseline_and_distinct_optimized_prompt(
    tmp_path: Path,
) -> None:
    """Catches prompt comparison evidence that is detached from the actual prompt text."""
    from week6_agent.day30_agent import DAY30_SYSTEM_PROMPT
    from week6_agent.day32_agent import DAY32_SYSTEM_PROMPT, load_prompt_manifest

    baseline = tmp_path / "system_prompt_v1.txt"
    optimized = tmp_path / "system_prompt_v2.txt"
    baseline.write_text(DAY30_SYSTEM_PROMPT + "\n", encoding="utf-8")
    optimized.write_text(DAY32_SYSTEM_PROMPT + "\n", encoding="utf-8")

    manifest = load_prompt_manifest(baseline, optimized)

    assert manifest["baseline"]["version"] == "week6-day30-three-tool-react-v1"
    assert manifest["optimized"]["version"] == "week6-day32-three-tool-react-v2"
    assert manifest["baseline"]["sha256"] == hashlib.sha256(
        DAY30_SYSTEM_PROMPT.encode()
    ).hexdigest()
    assert manifest["optimized"]["sha256"] == hashlib.sha256(
        DAY32_SYSTEM_PROMPT.encode()
    ).hexdigest()
    assert manifest["baseline"]["sha256"] != manifest["optimized"]["sha256"]
    assert manifest["baseline"]["path"] == "system_prompt_v1.txt"
    assert manifest["optimized"]["path"] == "system_prompt_v2.txt"
    assert manifest["change_scope"] == "single_rule_prompt_ablation"
    assert manifest["selection_split"] == "dev_v2"
    assert manifest["final_test_used_for_selection"] is False
    assert len(manifest["added_rules"]) == 1


def test_prompt_manifest_rejects_more_than_one_added_rule(tmp_path: Path) -> None:
    from week6_agent.day30_agent import DAY30_SYSTEM_PROMPT
    from week6_agent.day32_agent import DAY32_SYSTEM_PROMPT, load_prompt_manifest

    baseline = tmp_path / "system_prompt_main.txt"
    ablation = tmp_path / "system_prompt_ablation.txt"
    baseline.write_text(DAY30_SYSTEM_PROMPT + "\n", encoding="utf-8")
    ablation.write_text(
        DAY32_SYSTEM_PROMPT + "\n- Also change routing behavior.\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="exactly one rule"):
        load_prompt_manifest(baseline, ablation)


def test_prompt_selection_gate_accepts_dev_and_rejects_final_test() -> None:
    from week6_agent.day32_agent import validate_prompt_selection_split

    data = REPO_ROOT / "deliverables/week6/day31/source/data"
    manifest = data / "dataset_manifest_v2.json"

    accepted = validate_prompt_selection_split(data / "tool_sft_dev_v2.json", manifest)
    assert accepted["split"] == "dev_v2"
    assert accepted["row_count"] == 40

    with pytest.raises(ValueError, match="development split"):
        validate_prompt_selection_split(data / "tool_sft_test_v2.json", manifest)


def test_prompt_manifest_rejects_a_drifted_baseline(tmp_path: Path) -> None:
    """Catches an optimization comparison whose baseline silently changed."""
    from week6_agent.day32_agent import DAY32_SYSTEM_PROMPT, load_prompt_manifest

    baseline = tmp_path / "system_prompt_v1.txt"
    optimized = tmp_path / "system_prompt_v2.txt"
    baseline.write_text("drifted baseline\n", encoding="utf-8")
    optimized.write_text(DAY32_SYSTEM_PROMPT + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="baseline prompt"):
        load_prompt_manifest(baseline, optimized)


def test_day32_agent_reuses_exact_three_tools_and_executes_real_observation() -> None:
    """Catches the optimized factory dropping or renaming a required tool."""
    from week6_agent.day32_agent import build_day32_react_agent

    model = ScriptedChatModel(
        [
            _call("calculator", {"expression": "2 + 3"}, "call-1"),
            AIMessage(
                content="结果是 5。",
                response_metadata={"raw_generated_text": "结果是 5。"},
            ),
        ]
    )

    agent = build_day32_react_agent(
        model, KNOWLEDGE_BASE, max_steps=4, runtime_limit_seconds=5
    )
    result = agent.invoke("计算 2 + 3")

    assert model._bound_tool_names == (
        "calculator",
        "knowledge_retrieval",
        "code_executor",
    )
    assert result["trace"][0]["observation"]["data"]["value"] == 5
    assert result["final_answer"] == "结果是 5。"
