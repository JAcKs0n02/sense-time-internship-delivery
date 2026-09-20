"""Factory and bounded wrapper for the required LangGraph ReAct agent."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from week6_agent.tools.calculator import CalculatorTool
from week6_agent.tools.knowledge_retrieval import KnowledgeRetrievalTool
from week6_agent.policy import POLICY_MODES, validate_trace_policy
from week6_agent.trace_schema import extract_trace, final_answer, repeated_action_input


SYSTEM_PROMPT_VERSION = "week6-day29-react-v1"
SYSTEM_PROMPT = """You are a bounded tool-using assistant. Use calculator for arithmetic and knowledge_retrieval only for facts in the fixed local catalog. Emit a structured tool call when a tool is needed. After a tool observation, answer concisely from that observation. Never invent an observation or repeat an identical tool call."""
TOOL_NAMES = ("calculator", "knowledge_retrieval")
CODE_RELATIVE_PATHS = (
    "deliverables/week6/source/week6_agent/model_adapter.py",
    "deliverables/week6/source/week6_agent/agent_factory.py",
    "deliverables/week6/source/week6_agent/trace_schema.py",
    "deliverables/week6/day29/source/scripts/initialize_agent.py",
    "deliverables/week6/day29/source/scripts/run_single_turn.py",
    "deliverables/week6/day29/source/scripts/validate_day29.py",
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


def build_basic_tools(knowledge_base_path: Path) -> list[BaseTool]:
    """Import and wrap exactly the two approved Day28 implementations."""
    return [
        CalculatorTool().as_langchain_tool(),
        KnowledgeRetrievalTool(knowledge_base_path).as_langchain_tool(),
    ]


def build_tool_manifest(tools: Sequence[BaseTool]) -> dict[str, Any]:
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
    names = [entry["name"] for entry in entries]
    if tuple(names) != TOOL_NAMES:
        raise ValueError(f"unexpected tool names: {names}")
    return {
        "tool_names": names,
        "tools": entries,
        "tools_sha256": _canonical_sha256(entries),
    }


def verify_model_manifest(
    model_dir: Path, expected_manifest_path: Path, expected_manifest_sha256: str
) -> dict[str, Any]:
    """Hash every archived path and reject any missing, changed, or extra file."""
    expected = json.loads(expected_manifest_path.read_text(encoding="utf-8"))
    if expected.get("valid") is not True:
        raise ValueError("expected manifest valid must be true")
    if expected.get("model_type") != "qwen2":
        raise ValueError("expected manifest model_type must be qwen2")
    if Path(str(expected.get("model_dir", ""))).resolve() != model_dir.resolve():
        raise ValueError("expected manifest model_dir mismatch")
    if expected.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("expected manifest identity mismatch")
    entries = expected.get("files")
    if not isinstance(entries, list) or expected.get("file_count") != len(entries):
        raise ValueError("expected manifest file_count mismatch")
    expected_paths = [entry.get("path") for entry in entries]
    if any(
        not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts
        for path in expected_paths
    ):
        raise ValueError("invalid expected manifest path")
    actual_paths = sorted(
        str(path.relative_to(model_dir)) for path in model_dir.rglob("*") if path.is_file()
    )
    if actual_paths != sorted(expected_paths):
        raise ValueError("model file set mismatch")

    actual_entries: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for expected_entry in entries:
        path = model_dir / expected_entry["path"]
        actual_entry = {
            "path": expected_entry["path"],
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if actual_entry != {
            "path": expected_entry.get("path"),
            "bytes": expected_entry.get("bytes"),
            "sha256": expected_entry.get("sha256"),
        }:
            raise ValueError(f"model file mismatch: {expected_entry.get('path')}")
        aggregate.update(
            f"{actual_entry['sha256']}  {actual_entry['path']}\n".encode("utf-8")
        )
        actual_entries.append(actual_entry)
    actual_manifest_sha = aggregate.hexdigest()
    if actual_manifest_sha != expected_manifest_sha256:
        raise ValueError("aggregate manifest identity mismatch")
    return {
        "model_dir": str(model_dir.resolve()),
        "file_count": len(actual_entries),
        "files": actual_entries,
        "manifest_sha256": actual_manifest_sha,
    }


def _fingerprint_directory(directory: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(directory)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def verify_runtime_dpo_identity(
    base_dir: Path,
    adapter_dir: Path,
    adapter_manifest_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Bind the formal runtime to the archived SFT base plus exact DPO adapter."""
    if config.get("runtime_load_mode") != "base_plus_peft_merge":
        raise ValueError("runtime_load_mode must be base_plus_peft_merge")
    if base_dir.resolve() != Path(config["expected_remote_model_path"]).resolve():
        raise ValueError("remote model path mismatch")
    if adapter_dir.resolve() != Path(config["expected_remote_adapter_path"]).resolve():
        raise ValueError("remote adapter path mismatch")
    for required in (base_dir / "config.json", base_dir / "tokenizer_config.json"):
        if not required.is_file():
            raise ValueError(f"base model file missing: {required.name}")
    if not list(base_dir.glob("*.safetensors")):
        raise ValueError("base model weights missing")

    archived = json.loads(adapter_manifest_path.read_text(encoding="utf-8"))
    if archived.get("selected_run") != config["expected_adapter_run"]:
        raise ValueError("adapter selected_run mismatch")
    archived_adapter = archived.get("adapter", {})
    adapter_model = adapter_dir / "adapter_model.safetensors"
    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_model.is_file() or not adapter_config.is_file():
        raise ValueError("adapter files missing")
    adapter_evidence = {
        "path": str(adapter_model.resolve()),
        "bytes": adapter_model.stat().st_size,
        "sha256": _sha256(adapter_model),
        "config_sha256": _sha256(adapter_config),
    }
    expected_adapter = {
        "bytes": config["expected_adapter_bytes"],
        "sha256": config["expected_adapter_sha256"],
    }
    if {key: adapter_evidence[key] for key in expected_adapter} != expected_adapter:
        raise ValueError("adapter bytes or sha256 mismatch")
    if {
        "bytes": archived_adapter.get("bytes"),
        "sha256": archived_adapter.get("sha256"),
    } != expected_adapter:
        raise ValueError("adapter source manifest mismatch")

    base_files = _fingerprint_directory(base_dir)
    identity = {
        "load_mode": "base_plus_peft_merge",
        "base_model_dir": str(base_dir.resolve()),
        "base_archive_manifest_sha256": config["base_archive_manifest_sha256"],
        "base_files": base_files,
        "base_files_sha256": _canonical_sha256(base_files),
        "adapter_dir": str(adapter_dir.resolve()),
        "adapter": adapter_evidence,
        "adapter_source_manifest_sha256": _sha256(adapter_manifest_path),
        "archived_merged_model_path": config["archived_merged_model_path"],
        "archived_merged_manifest_sha256": config[
            "archived_merged_manifest_sha256"
        ],
    }
    runtime_identity_sha256 = _canonical_sha256(identity)
    return {
        **identity,
        "runtime_identity_sha256": runtime_identity_sha256,
        # Retained for the existing trace-binding schema.
        "manifest_sha256": runtime_identity_sha256,
    }


def validate_formal_model_path(model_path: Path, config: dict[str, Any]) -> str:
    expected = Path(config["expected_remote_model_path"]).resolve()
    actual = model_path.resolve()
    if actual != expected:
        raise ValueError(f"remote model path mismatch: {actual} != {expected}")
    return str(actual)


def validate_formal_adapter_path(adapter_path: Path, config: dict[str, Any]) -> str:
    expected = Path(config["expected_remote_adapter_path"]).resolve()
    actual = adapter_path.resolve()
    if actual != expected:
        raise ValueError(f"remote adapter path mismatch: {actual} != {expected}")
    return str(actual)


def collect_environment_evidence() -> dict[str, Any]:
    packages = {
        "langchain": "langchain",
        "langchain_community": "langchain-community",
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


def build_code_tool_prompt_manifest(
    repo_root: Path, knowledge_base_path: Path
) -> dict[str, Any]:
    files = [
        {
            "path": relative_path,
            "sha256": _sha256(repo_root / relative_path),
        }
        for relative_path in CODE_RELATIVE_PATHS
    ]
    return {
        "files": files,
        **build_tool_manifest(build_basic_tools(knowledge_base_path)),
        **system_prompt_manifest(),
    }


def build_effective_config(
    config: dict[str, Any], model_path: Path, adapter_path: Path, recursion_limit: int
) -> dict[str, Any]:
    return {
        **config,
        "remote_model_path": validate_formal_model_path(model_path, config),
        "remote_adapter_path": validate_formal_adapter_path(adapter_path, config),
        "recursion_limit": recursion_limit,
    }


def build_run_spec_payload(
    *,
    config: dict[str, Any],
    config_path: Path,
    source_manifest_path: Path,
    preflight: dict[str, Any],
    effective_config: dict[str, Any],
    code_manifest: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "model_id": config["model_id"],
        "runtime_load_mode": config["runtime_load_mode"],
        "expected_remote_model_path": config["expected_remote_model_path"],
        "expected_remote_adapter_path": config["expected_remote_adapter_path"],
        "expected_adapter_sha256": config["expected_adapter_sha256"],
        "archived_merged_manifest_sha256": config[
            "archived_merged_manifest_sha256"
        ],
        "preflight_manifest_sha256": preflight["manifest_sha256"],
        "source_manifest_sha256": _sha256(source_manifest_path),
        "config_file_sha256": _sha256(config_path),
        "effective_config": effective_config,
        "code_files": code_manifest["files"],
        "tools_sha256": code_manifest["tools_sha256"],
        "system_prompt_sha256": code_manifest["system_prompt_sha256"],
        "system_prompt_version": code_manifest["system_prompt_version"],
        "formal_prompt": config["formal_prompt"],
        "environment_api_evidence_sha256": environment["evidence_sha256"],
    }


def build_run_spec_document(payload: dict[str, Any]) -> dict[str, Any]:
    return {"run_spec": payload, "run_spec_sha256": _canonical_sha256(payload)}


def verify_run_spec_document(
    document: dict[str, Any], current_payload: dict[str, Any]
) -> str:
    if set(document) != {"run_spec", "run_spec_sha256"}:
        raise ValueError("invalid run_spec document schema")
    stored_payload = document["run_spec"]
    stored_sha = document["run_spec_sha256"]
    if not isinstance(stored_payload, dict) or stored_sha != _canonical_sha256(
        stored_payload
    ):
        raise ValueError("stored run_spec hash mismatch")
    if stored_payload != current_payload or stored_sha != _canonical_sha256(current_payload):
        raise ValueError("current critical inputs mismatch run_spec")
    return stored_sha


def evidence_binding_checks(
    trace_artifact: dict[str, Any],
    run_spec_document: dict[str, Any],
    current_payload: dict[str, Any],
    recorded_artifacts: dict[str, Any],
    current_artifacts: dict[str, Any],
) -> dict[str, bool]:
    try:
        run_spec_sha = verify_run_spec_document(run_spec_document, current_payload)
        integrity = True
    except (KeyError, TypeError, ValueError):
        run_spec_sha = None
        integrity = False
    payload = run_spec_document.get("run_spec", {})
    return {
        "run_spec_integrity": integrity,
        "trace_run_spec_sha256": trace_artifact.get("run_spec_sha256") == run_spec_sha,
        "trace_model_manifest_sha256": trace_artifact.get("model_manifest_sha256")
        == payload.get("preflight_manifest_sha256"),
        "runner_evidence": trace_artifact.get("runner_evidence") == current_artifacts,
        "initialization_artifacts": recorded_artifacts == current_artifacts,
    }


def recursion_limit_for(max_steps: int) -> int:
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    return max_steps * 2 + 4


def build_formal_evidence_bundle(
    *,
    config: dict[str, Any],
    config_path: Path,
    model_path: Path,
    adapter_path: Path,
    source_manifest_path: Path,
    repo_root: Path,
    knowledge_base_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    validate_formal_model_path(model_path, config)
    validate_formal_adapter_path(adapter_path, config)
    preflight = verify_runtime_dpo_identity(
        model_path, adapter_path, source_manifest_path, config
    )
    code_manifest = build_code_tool_prompt_manifest(repo_root, knowledge_base_path)
    environment = collect_environment_evidence()
    effective_config = build_effective_config(
        config, model_path, adapter_path, recursion_limit_for(config["max_steps"])
    )
    effective_config = {
        **effective_config,
        "config_file_sha256": _sha256(config_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
    }
    artifacts = {
        "model_preflight": preflight,
        "effective_config": effective_config,
        "code_tool_prompt_manifest": code_manifest,
        "environment": environment,
    }
    payload = build_run_spec_payload(
        config=config,
        config_path=config_path,
        source_manifest_path=source_manifest_path,
        preflight=preflight,
        effective_config=effective_config,
        code_manifest=code_manifest,
        environment=environment,
    )
    return artifacts, payload, build_run_spec_document(payload)


class _RuntimeExceeded(Exception):
    pass


class CompiledAgent:
    """Bound graph with enforced tool-step, recursion, runtime, and loop limits."""

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
        if policy_mode not in POLICY_MODES:
            raise ValueError(f"unsupported policy_mode: {policy_mode}")
        self.compiled_graph = compiled_graph
        self.tool_names = tuple(tool.name for tool in tools)
        self.max_steps = max_steps
        self.runtime_limit_seconds = runtime_limit_seconds
        self.policy_mode = policy_mode
        # One extra model decision is allowed so this wrapper can classify the
        # attempted (max_steps + 1)th call instead of accepting LangGraph's
        # generic "need more steps" fallback as a final answer.
        self.recursion_limit = recursion_limit_for(max_steps)
        self.tool_manifest = build_tool_manifest(tools)

    def invoke(self, prompt: str) -> dict[str, Any]:
        started = time.monotonic()
        messages: list[BaseMessage] = [HumanMessage(content=prompt)]
        error_code: str | None = None
        seen_call_ids: set[str] = set()
        seen_signatures: set[tuple[str, str]] = set()
        previous_handler: Any | None = None

        def alarm_handler(_: int, __: Any) -> None:
            raise _RuntimeExceeded

        use_alarm = hasattr(signal, "setitimer")
        if use_alarm:
            try:
                previous_handler = signal.signal(signal.SIGALRM, alarm_handler)
                signal.setitimer(signal.ITIMER_REAL, self.runtime_limit_seconds)
            except ValueError:
                use_alarm = False
        try:
            stream = self.compiled_graph.stream(
                {"messages": messages},
                config={"recursion_limit": self.recursion_limit},
                stream_mode="values",
            )
            for state in stream:
                messages = list(state.get("messages", messages))
                for message in messages:
                    if not isinstance(message, AIMessage):
                        continue
                    for call in message.tool_calls:
                        if call["id"] in seen_call_ids:
                            continue
                        seen_call_ids.add(call["id"])
                        signature = (
                            call["name"],
                            json.dumps(
                                call["args"],
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        )
                        if signature in seen_signatures:
                            error_code = "repeated_action_input"
                            break
                        seen_signatures.add(signature)
                        if len(seen_call_ids) > self.max_steps:
                            error_code = "max_steps_exceeded"
                            break
                    if error_code:
                        break
                if error_code:
                    break
                if time.monotonic() - started > self.runtime_limit_seconds:
                    error_code = "runtime_limit_exceeded"
                    break
        except _RuntimeExceeded:
            error_code = "runtime_limit_exceeded"
        except GraphRecursionError:
            error_code = "max_steps_exceeded"
        finally:
            if use_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_handler)

        trace = extract_trace(messages)
        if error_code is None and len(trace) >= self.max_steps and not final_answer(messages):
            error_code = "max_steps_exceeded"
        if error_code is None and repeated_action_input(trace):
            error_code = "repeated_action_input"
        answer = final_answer(messages)
        policy = validate_trace_policy(prompt, trace, answer)
        if error_code is None and self.policy_mode == "guarded" and policy["status"] == "FAIL":
            error_code = "policy_violation"
        return {
            "status": "failed" if error_code else "success",
            "error_code": error_code,
            "messages": messages,
            "trace": trace,
            "final_answer": answer,
            "policy_mode": self.policy_mode,
            "policy_status": policy["status"],
            "policy_checks": policy["checks"],
            "policy_violations": policy["violations"],
            "tool_names": list(self.tool_names),
            "max_steps": self.max_steps,
            "recursion_limit": self.recursion_limit,
            "runtime_limit_seconds": self.runtime_limit_seconds,
            "latency_seconds": time.monotonic() - started,
        }


def build_react_agent(
    model: Any,
    knowledge_base_path: Path,
    max_steps: int = 4,
    runtime_limit_seconds: float = 60,
    policy_mode: str = "raw",
) -> CompiledAgent:
    tools = build_basic_tools(Path(knowledge_base_path))
    graph = create_react_agent(
        model,
        tools,
        prompt=SYSTEM_PROMPT,
        version="v2",
        name="week6_day29_react_agent",
    )
    return CompiledAgent(graph, tools, max_steps, runtime_limit_seconds, policy_mode)


def system_prompt_manifest() -> dict[str, str]:
    return {
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
    }
