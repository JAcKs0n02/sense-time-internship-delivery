from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Sequence

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


DAY29_SOURCE = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
WEEK6_SOURCE = REPO_ROOT / "deliverables/week6/source"
KNOWLEDGE_BASE = REPO_ROOT / "deliverables/week6/day28/source/data/knowledge_base.json"
sys.path.insert(0, str(WEEK6_SOURCE))


class ScriptedChatModel(BaseChatModel):
    """Synthetic-test-only model; it never represents formal GPU evidence."""

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


def tool_call(
    expression: str, call_id: str, raw_generated_text: str | None = None
) -> AIMessage:
    raw = raw_generated_text or json.dumps(
        {"name": "calculator", "arguments": {"expression": expression}},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculator",
                "args": {"expression": expression},
                "id": call_id,
                "type": "tool_call",
            }
        ],
        response_metadata={"raw_generated_text": raw},
    )


def test_parser_accepts_only_actual_structured_model_output():
    """Catches prompt-text heuristics fabricating a calculator tool call."""
    from week6_agent.model_adapter import parse_model_output

    raw = '<tool_call>{"name":"calculator","arguments":{"expression":"123 * 456"}}</tool_call>'
    parsed = parse_model_output(raw)

    assert parsed.tool_calls == [
        {
            "name": "calculator",
            "args": {"expression": "123 * 456"},
            "id": "model-tool-call-0",
            "type": "tool_call",
        }
    ]
    assert parsed.response_metadata["raw_generated_text"] == raw
    assert parse_model_output("Please calculate 123 * 456 with calculator").tool_calls == []
    assert parse_model_output('{"name":"calculator","arguments":{}} trailing').tool_calls == []


def test_adapter_passes_bound_tool_schema_to_qwen_chat_template():
    """Catches an adapter that binds tools in LangGraph but hides them from Qwen."""
    from week6_agent.model_adapter import HuggingFaceLocalChatModel
    from week6_agent.tools.calculator import CalculatorTool

    class FakeTensor:
        shape = (1, 3)

        def to(self, _: Any) -> "FakeTensor":
            return self

    class FakeTokenizer:
        eos_token_id = 2
        pad_token_id = 2

        def __init__(self) -> None:
            self.tools: list[dict[str, Any]] | None = None

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> FakeTensor:
            assert messages[-1] == {"role": "user", "content": "synthetic test only"}
            self.tools = kwargs["tools"]
            return FakeTensor()

        def decode(self, tokens: Any, **_: Any) -> str:
            assert tokens == [9]
            return '{"name":"calculator","arguments":{"expression":"123 * 456"}}'

    class FakeModel:
        device = "cpu"

        def generate(self, inputs: FakeTensor, **_: Any) -> list[list[int]]:
            assert inputs.shape == (1, 3)
            return [[1, 2, 3, 9]]

    tokenizer = FakeTokenizer()
    model = HuggingFaceLocalChatModel.from_loaded(
        tokenizer=tokenizer, model=FakeModel(), model_path="synthetic-test-only"
    )
    model.bind_tools([CalculatorTool().as_langchain_tool()])

    response = model.invoke([HumanMessage(content="synthetic test only")])
    repeated_response = model.invoke([HumanMessage(content="synthetic test only")])

    assert tokenizer.tools is not None
    assert tokenizer.tools[0]["function"]["name"] == "calculator"
    assert response.tool_calls[0]["args"] == {"expression": "123 * 456"}
    assert repeated_response.tool_calls[0]["id"] != response.tool_calls[0]["id"]
    assert response.response_metadata["raw_generated_text"].startswith("{")


def test_factory_calls_real_langgraph_agent_and_binds_exact_day28_tools():
    """Catches copied tools or a factory that never builds a LangGraph agent."""
    from week6_agent.agent_factory import build_react_agent

    model = ScriptedChatModel([tool_call("123 * 456", "call-1"), AIMessage(content="56088")])
    agent = build_react_agent(
        model,
        knowledge_base_path=KNOWLEDGE_BASE,
        max_steps=2,
        runtime_limit_seconds=5,
    )
    result = agent.invoke("synthetic test only")

    assert agent.tool_names == ("calculator", "knowledge_retrieval")
    assert model._bound_tool_names == agent.tool_names
    assert agent.compiled_graph.__class__.__name__ == "CompiledStateGraph"
    assert result["status"] == "success"
    assert result["trace"][0]["action_input"] == {"expression": "123 * 456"}
    assert result["trace"][0]["observation"]["data"]["value"] == 56088
    assert result["final_answer"] == "56088"


def test_tool_manifest_records_exact_source_and_schema_hashes():
    """Catches source/schema evidence being copied from a stale status artifact."""
    from week6_agent.agent_factory import build_basic_tools, build_tool_manifest

    tools = build_basic_tools(KNOWLEDGE_BASE)
    manifest = build_tool_manifest(tools)
    by_name = {entry["name"]: entry for entry in manifest["tools"]}
    calculator_source = WEEK6_SOURCE / "week6_agent/tools/calculator.py"

    assert by_name["calculator"]["source_sha256"] == hashlib.sha256(
        calculator_source.read_bytes()
    ).hexdigest()
    assert len(by_name["calculator"]["schema_sha256"]) == 64
    assert manifest["tool_names"] == ["calculator", "knowledge_retrieval"]


def test_bounded_wrapper_stops_repeated_action_input_structurally():
    """Catches an identical action/input loop continuing until graph recursion fails."""
    from week6_agent.agent_factory import build_react_agent

    model = ScriptedChatModel(
        [tool_call("1 + 1", "call-1"), tool_call("1 + 1", "call-2")]
    )
    agent = build_react_agent(model, KNOWLEDGE_BASE, max_steps=4, runtime_limit_seconds=5)

    result = agent.invoke("synthetic test only")

    assert result["status"] == "failed"
    assert result["error_code"] == "repeated_action_input"
    assert len(result["trace"]) == 2


def test_bounded_wrapper_stops_after_configured_tool_step_limit():
    """Catches a max_steps setting that is recorded but never enforced."""
    from week6_agent.agent_factory import build_react_agent

    model = ScriptedChatModel(
        [tool_call("1 + 1", "call-1"), tool_call("2 + 2", "call-2")]
    )
    agent = build_react_agent(model, KNOWLEDGE_BASE, max_steps=1, runtime_limit_seconds=5)

    result = agent.invoke("synthetic test only")

    assert result["status"] == "failed"
    assert result["error_code"] == "max_steps_exceeded"
    assert result["max_steps"] == 1
    assert result["recursion_limit"] > result["max_steps"]


def test_formal_validator_recomputes_messages_and_rejects_claimed_pass():
    """Catches acceptance based on a stored PASS field instead of raw messages."""
    from week6_agent.trace_schema import validate_formal_messages

    valid_messages = [
        HumanMessage(content="计算 123 * 456"),
        tool_call("123 * 456", "call-1"),
        ToolMessage(
            content=json.dumps(
                {"status": "success", "data": {"value": 56088}, "error_code": None}
            ),
            tool_call_id="call-1",
            name="calculator",
        ),
        AIMessage(
            content="计算结果是 56088。",
            response_metadata={"raw_generated_text": "计算结果是 56088。"},
        ),
    ]
    assert validate_formal_messages(valid_messages)["status"] == "PASS"

    wrong_messages = list(valid_messages)
    wrong_messages[-1] = AIMessage(content="claimed PASS but wrong answer")
    result = validate_formal_messages(wrong_messages)
    assert result["status"] == "FAIL"
    assert "final_answer" in result["failed_checks"]


def test_formal_validator_accepts_equivalent_calculator_whitespace():
    """Catches rejecting a genuine calculator call solely for formatting spaces."""
    from week6_agent.trace_schema import validate_formal_messages

    messages = [
        HumanMessage(content="计算 123 * 456"),
        tool_call("123*456", "call-1"),
        ToolMessage(
            content=json.dumps(
                {"status": "success", "data": {"value": 56088}, "error_code": None}
            ),
            tool_call_id="call-1",
            name="calculator",
        ),
        AIMessage(content="56088", response_metadata={"raw_generated_text": "56088"}),
    ]

    result = validate_formal_messages(messages)

    assert result["status"] == "PASS"
    assert result["checks"]["exact_arguments"] is True


@pytest.mark.parametrize(
    "bad_raw",
    [
        "I should use the calculator for this.",
        '{"name":"calculator","arguments":{"expression":"123 + 456"}}',
    ],
)
def test_formal_validator_rejects_normalized_tool_call_without_matching_raw_provenance(
    bad_raw: str,
):
    """Catches hand-authored normalized calls being passed off as model output."""
    from week6_agent.trace_schema import validate_formal_messages

    messages = [
        HumanMessage(content="计算 123 * 456"),
        tool_call("123 * 456", "call-1", raw_generated_text=bad_raw),
        ToolMessage(
            content=json.dumps(
                {"status": "success", "data": {"value": 56088}, "error_code": None}
            ),
            tool_call_id="call-1",
            name="calculator",
        ),
        AIMessage(
            content="56088", response_metadata={"raw_generated_text": "56088"}
        ),
    ]

    result = validate_formal_messages(messages)

    assert result["status"] == "FAIL"
    assert "raw_tool_call_provenance" in result["failed_checks"]


def test_formal_validator_rejects_final_answer_without_matching_raw_provenance():
    """Catches a hand-authored final answer added after genuine tool evidence."""
    from week6_agent.trace_schema import validate_formal_messages

    messages = [
        HumanMessage(content="计算 123 * 456"),
        tool_call("123 * 456", "call-1"),
        ToolMessage(
            content=json.dumps(
                {"status": "success", "data": {"value": 56088}, "error_code": None}
            ),
            tool_call_id="call-1",
            name="calculator",
        ),
        AIMessage(
            content="56088", response_metadata={"raw_generated_text": "not the answer"}
        ),
    ]

    result = validate_formal_messages(messages)

    assert result["status"] == "FAIL"
    assert "raw_final_answer_provenance" in result["failed_checks"]


def _write_fake_manifest(model_dir: Path, manifest_path: Path) -> str:
    files = {
        "config.json": b"{}",
        "generation_config.json": b"{}",
        "model.safetensors": b"synthetic test only",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b'{"chat_template":"x"}',
    }
    entries = []
    for name, data in files.items():
        (model_dir / name).write_bytes(data)
        entries.append(
            {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(f"{entry['sha256']}  {entry['path']}\n".encode())
    manifest_sha = digest.hexdigest()
    manifest_path.write_text(
        json.dumps(
            {
                "valid": True,
                "model_dir": str(model_dir.resolve()),
                "model_type": "qwen2",
                "file_count": len(entries),
                "files": entries,
                "manifest_sha256": manifest_sha,
            }
        )
    )
    return manifest_sha


def test_model_preflight_recomputes_every_file_and_fails_on_tamper(tmp_path: Path):
    """Catches trusting archived file hashes without reading the actual model directory."""
    from week6_agent.agent_factory import verify_model_manifest

    model_dir = tmp_path / "synthetic-model"
    model_dir.mkdir()
    manifest_path = tmp_path / "expected.json"
    manifest_sha = _write_fake_manifest(model_dir, manifest_path)

    evidence = verify_model_manifest(model_dir, manifest_path, manifest_sha)
    assert evidence["file_count"] == 5
    assert evidence["manifest_sha256"] == manifest_sha

    (model_dir / "config.json").write_text('{"tampered":true}')
    with pytest.raises(ValueError, match="mismatch"):
        verify_model_manifest(model_dir, manifest_path, manifest_sha)


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("valid", False, "valid"),
        ("model_type", "llama", "model_type"),
        ("model_dir", "/wrong/model", "model_dir"),
    ],
)
def test_model_preflight_binds_archived_model_identity(
    tmp_path: Path, field: str, value: object, expected_error: str
):
    """Catches valid file hashes being paired with the wrong archived model identity."""
    from week6_agent.agent_factory import verify_model_manifest

    model_dir = tmp_path / "synthetic-model"
    model_dir.mkdir()
    manifest_path = tmp_path / "expected.json"
    manifest_sha = _write_fake_manifest(model_dir, manifest_path)
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=expected_error):
        verify_model_manifest(model_dir, manifest_path, manifest_sha)


def test_formal_model_path_is_frozen_and_cli_mismatch_is_rejected():
    """Catches a caller substituting another local model with a separate manifest."""
    from week6_agent.agent_factory import validate_formal_model_path

    config = json.loads(
        (DAY29_SOURCE / "configs/agent_config.json").read_text(encoding="utf-8")
    )
    expected = "/root/autodl-tmp/qwen25-week3/best_model/qwen25-7b-week3-best-merged"

    assert config["expected_remote_model_path"] == expected
    assert validate_formal_model_path(Path(expected), config) == expected
    with pytest.raises(ValueError, match="remote model path"):
        validate_formal_model_path(Path("/root/autodl-tmp/another-model"), config)


def test_run_spec_hash_and_evidence_bindings_reject_mixed_trace():
    """Catches a valid trace being mixed with a different initialization spec."""
    from week6_agent.agent_factory import (
        build_run_spec_document,
        evidence_binding_checks,
        verify_run_spec_document,
    )

    payload = {
        "schema_version": "1.0",
        "model_id": "reward_corrective_40step_merged",
        "expected_remote_model_path": "/synthetic/model",
        "expected_manifest_sha256": "a" * 64,
        "preflight_manifest_sha256": "a" * 64,
        "source_manifest_sha256": "b" * 64,
        "config_file_sha256": "c" * 64,
        "effective_config": {"formal_prompt": "计算 123 * 456"},
        "code_files": [{"path": "agent_factory.py", "sha256": "d" * 64}],
        "tools_sha256": "e" * 64,
        "system_prompt_sha256": "f" * 64,
        "system_prompt_version": "week6-day29-react-v1",
        "formal_prompt": "计算 123 * 456",
        "environment_api_evidence_sha256": "1" * 64,
    }
    document = build_run_spec_document(payload)
    assert verify_run_spec_document(document, payload) == document["run_spec_sha256"]

    artifacts = {
        "model_preflight": {"manifest_sha256": "a" * 64},
        "effective_config": payload["effective_config"],
        "code_tool_prompt_manifest": {
            "files": payload["code_files"],
            "tools_sha256": "e" * 64,
            "system_prompt_sha256": "f" * 64,
            "system_prompt_version": "week6-day29-react-v1",
        },
        "environment": {"evidence_sha256": "1" * 64},
    }
    trace = {
        "run_spec_sha256": document["run_spec_sha256"],
        "model_manifest_sha256": "a" * 64,
        "runner_evidence": artifacts,
    }
    assert all(
        evidence_binding_checks(trace, document, payload, artifacts, artifacts).values()
    )

    mixed_trace = {**trace, "run_spec_sha256": "9" * 64}
    checks = evidence_binding_checks(
        mixed_trace, document, payload, artifacts, artifacts
    )
    assert checks["trace_run_spec_sha256"] is False

    tampered_artifacts = {**artifacts, "effective_config": {"max_steps": 999}}
    checks = evidence_binding_checks(
        trace, document, payload, tampered_artifacts, artifacts
    )
    assert checks["initialization_artifacts"] is False


@pytest.mark.parametrize(
    "script_name", ["initialize_agent.py", "run_single_turn.py", "validate_day29.py"]
)
def test_remote_scripts_require_lineage_and_output_cli_arguments(script_name: str):
    """Catches a teacher script with hidden machine-specific lineage settings."""
    script = DAY29_SOURCE / "scripts" / script_name
    completed = subprocess.run(
        [sys.executable, str(script), "--help"], text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert "--remote-model-path" in completed.stdout
    assert "--remote-adapter-path" in completed.stdout
    assert "--expected-source-manifest" in completed.stdout
    assert "--output-directory" in completed.stdout
    if script_name != "initialize_agent.py":
        assert "--run-spec" in completed.stdout


def test_runtime_modules_do_not_import_transformers_or_torch_eagerly():
    """Catches unit-test imports accidentally requiring the GPU environment."""
    code = f"""
import sys
sys.path.insert(0, {str(WEEK6_SOURCE)!r})
import week6_agent.model_adapter
assert 'transformers' not in sys.modules
assert 'torch' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_local_loader_uses_offline_files_and_applies_frozen_seed(
    monkeypatch: pytest.MonkeyPatch,
):
    """Catches formal loading that reaches the network or ignores deterministic seed."""
    from week6_agent.model_adapter import HuggingFaceLocalChatModel

    calls: dict[str, Any] = {}

    class Loader:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> Any:
            calls.setdefault("loads", []).append((cls.__name__, path, kwargs))
            if cls.__name__ == "FakeTokenizerLoader":
                return object()
            return types.SimpleNamespace(eval=lambda: calls.update(eval_called=True))

    class FakeTokenizerLoader(Loader):
        pass

    class FakeModelLoader(Loader):
        pass

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModelForCausalLM=FakeModelLoader,
            AutoTokenizer=FakeTokenizerLoader,
            set_seed=lambda seed: calls.update(seed=seed),
        ),
    )

    HuggingFaceLocalChatModel.from_local_model("/synthetic/model", seed=42)

    assert calls["seed"] == 42
    assert all(load[2]["local_files_only"] is True for load in calls["loads"])
    assert calls["eval_called"] is True


def test_local_loader_merges_the_supplied_peft_adapter_offline(
    monkeypatch: pytest.MonkeyPatch,
):
    """Catches a runtime that records DPO lineage but actually loads SFT-only."""
    from week6_agent.model_adapter import HuggingFaceLocalChatModel

    calls: dict[str, Any] = {}
    base_model = types.SimpleNamespace()
    merged_model = types.SimpleNamespace(eval=lambda: calls.update(eval_called=True))

    class FakeTokenizerLoader:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> object:
            calls["tokenizer"] = (path, kwargs)
            return object()

    class FakeModelLoader:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> object:
            calls["base"] = (path, kwargs)
            return base_model

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, model: object, path: str, **kwargs: Any) -> Any:
            calls["adapter"] = (model, path, kwargs)
            return types.SimpleNamespace(
                merge_and_unload=lambda: calls.update(merged=True) or merged_model
            )

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModelForCausalLM=FakeModelLoader,
            AutoTokenizer=FakeTokenizerLoader,
            set_seed=lambda seed: calls.update(seed=seed),
        ),
    )
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=FakePeftModel))

    HuggingFaceLocalChatModel.from_local_model(
        "/synthetic/base", adapter_path="/synthetic/dpo-adapter", seed=42
    )

    assert calls["adapter"] == (
        base_model,
        "/synthetic/dpo-adapter",
        {"local_files_only": True},
    )
    assert calls["merged"] is True
    assert calls["eval_called"] is True


def test_runtime_dpo_preflight_binds_base_and_adapter_bytes(tmp_path: Path):
    """Catches SFT-only substitution or a copied adapter with changed bytes."""
    from week6_agent.agent_factory import verify_runtime_dpo_identity

    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    (base / "config.json").write_text('{"model_type":"qwen2"}')
    (base / "model.safetensors").write_bytes(b"synthetic base")
    (base / "tokenizer_config.json").write_text("{}")
    adapter_bytes = b"synthetic dpo adapter"
    (adapter / "adapter_model.safetensors").write_bytes(adapter_bytes)
    (adapter / "adapter_config.json").write_text("{}")
    manifest = tmp_path / "adapter_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "selected_run": "reward-corrective-40step/attempt_001",
                "adapter": {
                    "path": "/archived/adapter_model.safetensors",
                    "bytes": len(adapter_bytes),
                    "sha256": hashlib.sha256(adapter_bytes).hexdigest(),
                },
            }
        )
    )
    config = {
        "runtime_load_mode": "base_plus_peft_merge",
        "expected_remote_model_path": str(base),
        "expected_remote_adapter_path": str(adapter),
        "expected_adapter_sha256": hashlib.sha256(adapter_bytes).hexdigest(),
        "expected_adapter_bytes": len(adapter_bytes),
        "expected_adapter_run": "reward-corrective-40step/attempt_001",
        "archived_merged_model_path": "/archive/week4-merged",
        "archived_merged_manifest_sha256": "a" * 64,
        "base_archive_manifest_sha256": "b" * 64,
    }

    evidence = verify_runtime_dpo_identity(base, adapter, manifest, config)
    assert evidence["load_mode"] == "base_plus_peft_merge"
    assert evidence["adapter"]["sha256"] == config["expected_adapter_sha256"]
    assert evidence["manifest_sha256"] == evidence["runtime_identity_sha256"]

    (adapter / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="adapter"):
        verify_runtime_dpo_identity(base, adapter, manifest, config)


@pytest.mark.parametrize("script_name", ["initialize_agent.py", "validate_day29.py"])
def test_audit_scripts_resolve_the_worktree_as_repository_root(script_name: str):
    """Catches code evidence paths being recorded relative to the workspace parent."""
    script = DAY29_SOURCE / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"test_{script.stem}", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.REPO_ROOT == REPO_ROOT
