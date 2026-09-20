"""Deterministic, auditable message and ReAct trace helpers."""

from __future__ import annotations

import json
from typing import Any, Iterable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from week6_agent.model_adapter import parse_model_output


FORMAL_PROMPT = "计算 123 * 456"


def _observation(content: object) -> object:
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return content


def extract_trace(messages: Iterable[BaseMessage]) -> list[dict[str, Any]]:
    """Pair every AI tool request with its real ToolMessage observation."""
    message_list = list(messages)
    observations = {
        message.tool_call_id: _observation(message.content)
        for message in message_list
        if isinstance(message, ToolMessage)
    }
    trace: list[dict[str, Any]] = []
    for message in message_list:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            trace.append(
                {
                    "step_index": len(trace) + 1,
                    "action": call["name"],
                    "action_input": call["args"],
                    "tool_call_id": call["id"],
                    "raw_generated_text": message.response_metadata.get(
                        "raw_generated_text"
                    ),
                    "observation": observations.get(call["id"]),
                }
            )
    return trace


def repeated_action_input(trace: Iterable[dict[str, Any]]) -> bool:
    seen: set[tuple[str, str]] = set()
    for event in trace:
        signature = (
            event["action"],
            json.dumps(
                event["action_input"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        )
        if signature in seen:
            return True
        seen.add(signature)
    return False


def final_answer(messages: Iterable[BaseMessage]) -> str:
    for message in reversed(list(messages)):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return str(message.content)
    return ""


def _tool_call_semantics(message: AIMessage) -> list[tuple[str, dict[str, Any]]]:
    return [(call["name"], call["args"]) for call in message.tool_calls]


def _raw_tool_calls_match(messages: Iterable[BaseMessage]) -> bool:
    for message in messages:
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue
        raw = message.response_metadata.get("raw_generated_text")
        if not isinstance(raw, str) or not raw.strip():
            return False
        reparsed = parse_model_output(raw, call_id_prefix="validator-tool-call")
        if _tool_call_semantics(reparsed) != _tool_call_semantics(message):
            return False
    return True


def _raw_final_answer_matches(messages: Iterable[BaseMessage]) -> bool:
    for message in reversed(list(messages)):
        if isinstance(message, AIMessage) and not message.tool_calls:
            raw = message.response_metadata.get("raw_generated_text")
            return isinstance(raw, str) and bool(raw.strip()) and raw == message.content
    return False


def _formal_calculator_arguments_match(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {"expression"}:
        return False
    expression = value["expression"]
    return isinstance(expression, str) and "".join(expression.split()) == "123*456"


def serialize_message(message: BaseMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": message.type, "content": message.content}
    if isinstance(message, AIMessage):
        payload["tool_calls"] = message.tool_calls
        payload["response_metadata"] = message.response_metadata
    if isinstance(message, ToolMessage):
        payload["tool_call_id"] = message.tool_call_id
        payload["name"] = message.name
    return payload


def deserialize_messages(rows: Iterable[dict[str, Any]]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for row in rows:
        if row.get("type") == "human":
            messages.append(HumanMessage(content=row.get("content", "")))
        elif row.get("type") == "ai":
            messages.append(
                AIMessage(
                    content=row.get("content", ""),
                    tool_calls=row.get("tool_calls", []),
                    response_metadata=row.get("response_metadata", {}),
                )
            )
        elif row.get("type") == "tool":
            messages.append(
                ToolMessage(
                    content=row.get("content", ""),
                    tool_call_id=row.get("tool_call_id", ""),
                    name=row.get("name"),
                )
            )
        else:
            raise ValueError("unsupported_message_type")
    return messages


def validate_formal_messages(messages: Iterable[BaseMessage]) -> dict[str, Any]:
    """Recompute the Day29 case result from raw messages, never a stored status."""
    message_list = list(messages)
    trace = extract_trace(message_list)
    answer = final_answer(message_list)
    checks = {
        "fixed_prompt": bool(message_list)
        and isinstance(message_list[0], HumanMessage)
        and message_list[0].content == FORMAL_PROMPT,
        "exactly_one_tool_call": len(trace) == 1,
        "calculator_action": len(trace) == 1 and trace[0]["action"] == "calculator",
        "exact_arguments": len(trace) == 1
        and _formal_calculator_arguments_match(trace[0]["action_input"]),
        "successful_observation": len(trace) == 1
        and trace[0]["observation"]
        == {"status": "success", "data": {"value": 56088}, "error_code": None},
        "final_answer": "56088" in answer,
        "no_repeated_action_input": not repeated_action_input(trace),
        "raw_tool_call_provenance": _raw_tool_calls_match(message_list),
        "raw_final_answer_provenance": _raw_final_answer_matches(message_list),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
        "trace": trace,
        "final_answer": answer,
    }
