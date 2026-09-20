"""Three-tool Day30 ReAct agent, isolated from the verified Day29 factory."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from week6_agent.agent_factory import CompiledAgent, recursion_limit_for
from week6_agent.agent_factory import (
    validate_formal_adapter_path,
    validate_formal_model_path,
    verify_runtime_dpo_identity,
)
from week6_agent.tools.calculator import CalculatorTool
from week6_agent.tools.code_executor import CodeExecutor
from week6_agent.tools.knowledge_retrieval import KnowledgeRetrievalTool


DAY30_SYSTEM_PROMPT_VERSION = "week6-day30-three-tool-react-v1"
DAY30_SYSTEM_PROMPT = """You are a bounded three-tool assistant.

Tool policy:
- For a product-price or budget question, first call knowledge_retrieval using the product name. Read price_cny and shipping_cny from its observation. Then call calculator with exactly price_cny + shipping_cny. Only after both observations, compare the returned total with the stated budget and answer with product price, shipping, total, whether it exceeds the budget, and the difference.
- For Python source inspection, call code_executor exactly once. It parses AST only and never runs the source. Report syntax_valid, risk_nodes, and rejection status from the observation.
- Use calculator only for arithmetic, knowledge_retrieval only for the fixed local catalog, and code_executor only for non-executing AST inspection.

Never invent an observation, never execute code, never skip a required tool, and never repeat an identical tool call."""
DAY30_TOOL_NAMES = ("calculator", "knowledge_retrieval", "code_executor")
DAY30_CODE_RELATIVE_PATHS = (
    "deliverables/week6/source/week6_agent/day30_agent.py",
    "deliverables/week6/source/week6_agent/day30_validation.py",
    "deliverables/week6/source/week6_agent/tools/calculator.py",
    "deliverables/week6/source/week6_agent/tools/knowledge_retrieval.py",
    "deliverables/week6/source/week6_agent/tools/code_executor.py",
    "deliverables/week6/day30/source/scripts/initialize_day30.py",
    "deliverables/week6/day30/source/scripts/run_multistep.py",
    "deliverables/week6/day30/source/scripts/validate_day30.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_day30_tools(knowledge_base_path: Path) -> list[BaseTool]:
    return [
        CalculatorTool().as_langchain_tool(),
        KnowledgeRetrievalTool(knowledge_base_path).as_langchain_tool(),
        CodeExecutor().as_langchain_tool(),
    ]


def build_day30_tool_manifest(tools: Sequence[BaseTool]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for tool in tools:
        source_path_value = inspect.getsourcefile(tool.func)
        if source_path_value is None:
            raise ValueError(f"tool source unavailable: {tool.name}")
        source_path = Path(source_path_value).resolve()
        schema = tool.args_schema.model_json_schema() if tool.args_schema else {}
        entries.append(
            {
                "name": tool.name,
                "description": tool.description,
                "source_path": str(source_path),
                "source_sha256": _sha256(source_path),
                "schema": schema,
                "schema_sha256": _canonical_sha256(schema),
            }
        )
    names = tuple(entry["name"] for entry in entries)
    if names != DAY30_TOOL_NAMES:
        raise ValueError(f"unexpected Day30 tool names: {names}")
    return {
        "tool_names": list(names),
        "tools": entries,
        "tools_sha256": _canonical_sha256(entries),
    }


class Day30CompiledAgent(CompiledAgent):
    """Use the proven bounded invoke loop with a three-tool manifest."""

    def __init__(
        self,
        compiled_graph: Any,
        tools: Sequence[BaseTool],
        max_steps: int,
        runtime_limit_seconds: float,
        policy_mode: str = "raw",
    ) -> None:
        if max_steps < 1 or runtime_limit_seconds <= 0:
            raise ValueError("agent limits must be positive")
        self.compiled_graph = compiled_graph
        self.tool_names = tuple(tool.name for tool in tools)
        if self.tool_names != DAY30_TOOL_NAMES:
            raise ValueError(f"unexpected Day30 tool names: {self.tool_names}")
        self.max_steps = max_steps
        self.runtime_limit_seconds = runtime_limit_seconds
        if policy_mode not in ("raw", "guarded"):
            raise ValueError(f"unsupported policy_mode: {policy_mode}")
        self.policy_mode = policy_mode
        self.recursion_limit = recursion_limit_for(max_steps)
        self.tool_manifest = build_day30_tool_manifest(tools)


def build_day30_react_agent(
    model: Any,
    knowledge_base_path: Path,
    max_steps: int = 6,
    runtime_limit_seconds: float = 60,
    policy_mode: str = "raw",
) -> Day30CompiledAgent:
    tools = build_day30_tools(Path(knowledge_base_path))
    graph = create_react_agent(
        model,
        tools,
        prompt=DAY30_SYSTEM_PROMPT,
        version="v2",
        name="week6_day30_three_tool_react_agent",
    )
    return Day30CompiledAgent(graph, tools, max_steps, runtime_limit_seconds, policy_mode)


def day30_system_prompt_manifest() -> dict[str, str]:
    return {
        "system_prompt_version": DAY30_SYSTEM_PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(
            DAY30_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
    }


def build_day30_code_prompt_manifest(
    repo_root: Path, knowledge_base_path: Path, cases_path: Path
) -> dict[str, Any]:
    files = [
        {"path": relative_path, "sha256": _sha256(repo_root / relative_path)}
        for relative_path in DAY30_CODE_RELATIVE_PATHS
    ]
    return {
        "files": files,
        "cases_path": str(cases_path.resolve()),
        "cases_sha256": _sha256(cases_path),
        **build_day30_tool_manifest(build_day30_tools(knowledge_base_path)),
        **day30_system_prompt_manifest(),
    }


def build_day30_evidence_bundle(
    *,
    config: dict[str, Any],
    config_path: Path,
    model_path: Path,
    adapter_path: Path,
    source_manifest_path: Path,
    repo_root: Path,
    knowledge_base_path: Path,
    cases_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    validate_formal_model_path(model_path, config)
    validate_formal_adapter_path(adapter_path, config)
    preflight = verify_runtime_dpo_identity(
        model_path, adapter_path, source_manifest_path, config
    )
    code_manifest = build_day30_code_prompt_manifest(
        repo_root, knowledge_base_path, cases_path
    )
    environment = collect_day30_environment_evidence()
    effective_config = {
        **config,
        "remote_model_path": str(model_path.resolve()),
        "remote_adapter_path": str(adapter_path.resolve()),
        "recursion_limit": recursion_limit_for(config["max_steps"]),
        "config_file_sha256": _sha256(config_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
    }
    artifacts = {
        "model_preflight": preflight,
        "effective_config": effective_config,
        "code_tool_prompt_manifest": code_manifest,
        "environment": environment,
    }
    payload = {
        "schema_version": "1.0",
        "model_id": config["model_id"],
        "runtime_load_mode": config["runtime_load_mode"],
        "preflight_manifest_sha256": preflight["manifest_sha256"],
        "source_manifest_sha256": _sha256(source_manifest_path),
        "config_file_sha256": _sha256(config_path),
        "cases_sha256": code_manifest["cases_sha256"],
        "tools_sha256": code_manifest["tools_sha256"],
        "system_prompt_sha256": code_manifest["system_prompt_sha256"],
        "system_prompt_version": code_manifest["system_prompt_version"],
        "code_files": code_manifest["files"],
        "effective_config": effective_config,
        "environment_api_evidence_sha256": environment["evidence_sha256"],
    }
    document = {"run_spec": payload, "run_spec_sha256": _canonical_sha256(payload)}
    return artifacts, payload, document


def collect_day30_environment_evidence() -> dict[str, Any]:
    """Record only packages imported by the Day30 runtime."""
    packages = {
        "langchain": "langchain",
        "langchain_core": "langchain-core",
        "langgraph": "langgraph",
        "langgraph_prebuilt": "langgraph-prebuilt",
        "pydantic": "pydantic",
        "peft": "peft",
        "transformers": "transformers",
        "torch": "torch",
    }
    evidence = {
        "python": sys.version.split()[0],
        "python_executable": str(Path(sys.executable).resolve()),
        "packages": {
            key: importlib.metadata.version(distribution)
            for key, distribution in packages.items()
        },
        "create_react_agent": {
            "module": create_react_agent.__module__,
            "name": create_react_agent.__name__,
            "callable": callable(create_react_agent),
        },
    }
    return {**evidence, "evidence_sha256": _canonical_sha256(evidence)}


def verify_day30_run_spec(
    document: dict[str, Any], current_payload: dict[str, Any]
) -> str:
    if set(document) != {"run_spec", "run_spec_sha256"}:
        raise ValueError("invalid Day30 run_spec schema")
    if document["run_spec_sha256"] != _canonical_sha256(document["run_spec"]):
        raise ValueError("stored Day30 run_spec hash mismatch")
    if document["run_spec"] != current_payload:
        raise ValueError("current inputs mismatch frozen Day30 run_spec")
    return document["run_spec_sha256"]
