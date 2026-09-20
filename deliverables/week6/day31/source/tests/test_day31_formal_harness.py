from __future__ import annotations

import subprocess
import sys
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[5]
PREFLIGHT = REPO_ROOT / "deliverables/week6/day31/source/scripts/formal_preflight.py"
COLLECTOR = PREFLIGHT.parent / "collect_training_evidence.py"
SUMMARY_BUILDER = PREFLIGHT.parent / "build_pre_post_eval_summary.py"
EVALUATOR = PREFLIGHT.parent / "evaluate_frozen_cases.py"
CHECKPOINT_SELECTOR = PREFLIGHT.parent / "select_v2_checkpoint.py"
sys.path.insert(0, str(PREFLIGHT.parent))

from day31_harness import (  # noqa: E402
    WEEK4_MERGED_MANIFEST_SHA256,
    build_preflight_report,
    collect_training_evidence,
    evaluate_frozen_cases,
    resolve_scheme_a_lineage,
    score_frozen_case,
    score_case_v2,
    verify_official_parser_receipt,
    verify_ready_merged_preflight,
)


V2_DATA = REPO_ROOT / "deliverables/week6/day31/source/data"
V2_KNOWLEDGE_BASE = V2_DATA / "knowledge_base_v2.json"


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return list(range(sum(len(message["content"]) for message in messages) // 3 + 1))


class FakeModel:
    def __init__(self, response):
        self.response = response

    def invoke(self, _row):
        return self.response


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _manifest(directory: Path, digest: str | None = None) -> Path:
    payload = directory / "config.json"
    _write(payload, "{}")
    manifest = directory.parent / "manifest.json"
    entries = [{"path": "config.json", "bytes": payload.stat().st_size, "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}]
    computed = hashlib.sha256(
        "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in entries).encode("utf-8")
    ).hexdigest()
    _write(manifest, __import__("json").dumps({
        "valid": True, "model_type": "qwen2", "model_dir": str(directory.resolve()),
        "file_count": len(entries), "files": entries, "manifest_sha256": digest or computed,
    }))
    return manifest


def _manifest_digest(path: Path) -> str:
    return json.loads(path.read_text(encoding="utf-8"))["manifest_sha256"]


def _valid_safetensors() -> bytes:
    header = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": {
            "dtype": "F32", "shape": [1], "data_offsets": [0, 4],
        }
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return len(encoded).to_bytes(8, "little") + encoded + b"\0\0\0\0"


def _signed_receipt(value: dict[str, object]) -> dict[str, object]:
    value["receipt_sha256"] = hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return value


def test_official_runner_passes_v093_alignment_namespaces() -> None:
    """The smoke runner invokes registry and alignment APIs with v0.9.3 fields."""
    runner = importlib.import_module("run_official_parser_smoke")
    calls: dict[str, object] = {}
    attrs = [SimpleNamespace(dataset_name="tool_sft_train_100.json"), SimpleNamespace(dataset_name="tool_sft_eval_frozen.json")]

    def get_dataset_list(names, data_dir):
        calls["registry"] = (names, data_dir)
        return attrs

    def load_dataset(kind, *, data_files, split):
        calls.setdefault("load", []).append((kind, data_files, split))
        return [{"source": data_files}]

    def align_dataset(dataset, attr, data_args, training_args):
        calls.setdefault("align", []).append((dataset, attr, data_args, training_args))
        return [{"_prompt": "p", "_response": "r", "_tools": "[]"}]

    projections = runner.align_registered_datasets(
        Path("/tmp/official-parser-data"), get_dataset_list, align_dataset, load_dataset
    )
    assert [count for count, _ in projections] == [1, 1]
    assert calls["registry"][0] == ["week6_tool_sft_train", "week6_tool_sft_eval_frozen"]
    assert len(calls["align"]) == 2
    data_args = calls["align"][0][2]
    training_args = calls["align"][0][3]
    assert data_args.media_dir == "/tmp/official-parser-data"
    assert data_args.streaming is False
    assert data_args.preprocessing_num_workers == 1
    assert data_args.overwrite_cache is True
    assert training_args.local_process_index == 0


def test_official_runner_v2_profile_uses_train_and_dev_registry_names() -> None:
    runner = importlib.import_module("run_official_parser_smoke")
    calls: dict[str, object] = {}
    attrs = [
        SimpleNamespace(dataset_name="tool_sft_train_v2.json"),
        SimpleNamespace(dataset_name="tool_sft_dev_v2.json"),
    ]

    def get_dataset_list(names, data_dir):
        calls["registry"] = (names, data_dir)
        return attrs

    projections = runner.align_registered_datasets(
        Path("/tmp/official-parser-v2"),
        get_dataset_list,
        lambda dataset, attr, data_args, training_args: [
            {"_prompt": "p", "_response": "r", "_tools": "[]"}
        ],
        lambda kind, *, data_files, split: [{"source": data_files}],
        dataset_names=["week6_tool_sft_train_v2", "week6_tool_sft_dev_v2"],
    )

    assert [count for count, _ in projections] == [1, 1]
    assert calls["registry"][0] == [
        "week6_tool_sft_train_v2",
        "week6_tool_sft_dev_v2",
    ]


def test_official_v2_profile_spec_binds_300_train_and_40_dev_rows() -> None:
    import day31_harness

    spec = day31_harness.official_profile_spec("v2")

    assert spec["registry_names"] == [
        "week6_tool_sft_train_v2",
        "week6_tool_sft_dev_v2",
    ]
    assert spec["file_names"] == ["tool_sft_train_v2.json", "tool_sft_dev_v2.json"]
    assert spec["row_counts"] == [300, 40]
    assert spec["dataset_info_name"] == "dataset_info_v2.json"


def test_official_runner_source_prevents_empty_alignment_namespaces() -> None:
    source = (PREFLIGHT.parent / "run_official_parser_smoke.py").read_text(encoding="utf-8")
    assert "align_dataset(dataset, attr, data_args, training_args)" in source
    assert "SimpleNamespace()" not in source
    assert 'spec["dataset_info_name"]' in source


def test_official_receipt_recomputes_runner_and_provenance(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    _write(runner, "# pinned runner\n")
    data_dir = REPO_ROOT / "deliverables/week6/day31/source/data"
    train_path = data_dir / "tool_sft_train_100.json"
    evaluation_path = data_dir / "tool_sft_eval_frozen.json"
    info_path = REPO_ROOT / "deliverables/week6/day31/configs/dataset_info.json"
    train = json.loads(train_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    provenance = {
        "version": "0.9.3", "git_revision": "ca75f1edf3cb50343ed1c98605141c3e22075b5f",
        "official_module_hashes": {
            "llamafactory.data.parser": {"path": "/official/parser.py", "sha256": "a" * 64},
            "llamafactory.data.converter": {"path": "/official/converter.py", "sha256": "b" * 64},
            "llamafactory.data.loader": {"path": "/official/loader.py", "sha256": "c" * 64},
        },
    }
    receipt = _signed_receipt({
        "command": ["runner"], "exit_code": 0, "version": provenance["version"], "git_revision": provenance["git_revision"],
        "train_aligned_rows": 100, "eval_aligned_rows": 30,
        "aligned_projection_sha256": {"train": "d" * 64, "eval_frozen": "e" * 64},
        "dataset_sha256": {
            "train": hashlib.sha256(json.dumps(train, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "eval_frozen": hashlib.sha256(json.dumps(evaluation, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        },
        "runner_script_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "dataset_file_sha256": {
            "train": hashlib.sha256(train_path.read_bytes()).hexdigest(),
            "eval_frozen": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
            "dataset_info": hashlib.sha256(info_path.read_bytes()).hexdigest(),
        },
        "official_module_hashes": provenance["official_module_hashes"],
    })
    alignment = lambda: {"train": "d" * 64, "eval_frozen": "e" * 64}
    assert verify_official_parser_receipt(
        receipt, train, evaluation, runner, provenance_provider=lambda: provenance, alignment_provider=alignment
    )
    forged = dict(receipt)
    forged.pop("receipt_sha256")
    forged["runner_script_sha256"] = "f" * 64
    forged = _signed_receipt(forged)
    assert not verify_official_parser_receipt(
        forged, train, evaluation, runner, provenance_provider=lambda: provenance, alignment_provider=alignment
    )


def test_ready_preflight_rechecks_manifest_instead_of_trusting_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import day31_harness
    model = tmp_path / "merged"
    model.mkdir()
    preflight = tmp_path / "preflight.json"
    _write(preflight, json.dumps({
        "status": "ready", "lineage": {
            "scheme": "A", "load_mode": "verified_week4_merged", "model_dir": str(model),
            "merged_manifest_path": str(tmp_path / "missing-manifest.json"),
            "merged_manifest_sha256": WEEK4_MERGED_MANIFEST_SHA256,
        }, "official_parser_smoke": {"receipt_sha256": "self-asserted"},
    }))
    monkeypatch.setattr(day31_harness, "verify_official_parser_receipt", lambda *_args, **_kwargs: True)
    with pytest.raises(ValueError, match="merged manifest"):
        verify_ready_merged_preflight(preflight, model)


def test_preflight_cli_rejects_missing_lineage_inputs(tmp_path: Path) -> None:
    """Formal execution cannot begin without a verified Scheme A lineage."""
    completed = subprocess.run(
        [sys.executable, str(PREFLIGHT), "--output", str(tmp_path / "preflight.json")],
        cwd=REPO_ROOT, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 2
    assert "official-parser-receipt" in completed.stderr.lower()


def test_evaluator_cli_exposes_versioned_data_and_knowledge_base_inputs() -> None:
    """Formal v2 evaluation must bind explicit split and catalog bytes."""
    completed = subprocess.run(
        [sys.executable, str(EVALUATOR), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--eval-file" in completed.stdout
    assert "--knowledge-base" in completed.stdout
    assert "--evaluation-profile" in completed.stdout
    assert "--policy-mode" in completed.stdout
    assert "--split-role" in completed.stdout


def test_checkpoint_selector_uses_raw_development_strict_success_first(
    tmp_path: Path,
) -> None:
    """Checkpoint selection must never rank guarded or final-test results."""
    protocol = {
        "evaluation_profile": "v2",
        "split_role": "development",
        "policy_mode": "raw",
        "frozen_eval_file_sha256": "a" * 64,
        "knowledge_base_sha256": "b" * 64,
    }
    common = {
        "schema_version": "2.0",
        "status": "completed",
        "case_count": 40,
        "generation_config": {"do_sample": False, "seed": 42},
        "protocol": protocol,
    }
    candidates = []
    for name, passed, argument, sequence in (
        ("checkpoint-19", 34, 0.98, 0.95),
        ("checkpoint-38", 37, 0.96, 0.93),
        ("final", 37, 0.95, 0.96),
    ):
        path = tmp_path / f"{name}.json"
        payload = {
            **common,
            "model_identity": {
                "mode": "post",
                "adapter_path": f"/runs/{name}",
                "adapter_model_sha256": (name[0] * 64),
            },
            "metrics": {
                "first_tool_choice": 0.98,
                "argument_correctness": argument,
                "full_tool_sequence": sequence,
                "semantic_correctness": 0.95,
                "observation_integrity": 1.0,
                "no_loops": 1.0,
                "final_answer_grounded": 0.98,
            },
            "case_results": [
                {"case_id": f"case-{index}", "passed": index < passed}
                for index in range(40)
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        candidates.append(path)
    output = tmp_path / "selection.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(CHECKPOINT_SELECTOR),
            "--output",
            str(output),
            *sum((["--candidate", str(path)] for path in candidates), []),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["selected_checkpoint"] == "/runs/checkpoint-38"
    assert report["selection_metrics"]["strict_success"] == 37 / 40
    assert report["acceptance_gate"]["passed"] is True


def test_lora_config_declares_v093_qlora_memory_contract() -> None:
    import yaml

    config = yaml.safe_load(
        (REPO_ROOT / "deliverables/week6/day31/configs/week6_tool_sft_lora.yaml").read_text(encoding="utf-8")
    )
    assert config["quantization_bit"] == 4
    assert config["quantization_type"] == "nf4"
    assert config["double_quantization"] is True
    assert config["gradient_checkpointing"] is True
    assert config["save_total_limit"] == 2


def test_v2_lora_config_binds_formal_data_and_selection_cadence() -> None:
    import yaml

    config_path = (
        REPO_ROOT / "deliverables/week6/day31/configs/week6_tool_sft_lora_v2.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["dataset"] == "week6_tool_sft_train_v2"
    assert config["eval_dataset"] == "week6_tool_sft_dev_v2"
    assert config["learning_rate"] == pytest.approx(5e-5)
    assert config["num_train_epochs"] == 2.0
    assert config["lora_dropout"] == pytest.approx(0.05)
    assert config["eval_strategy"] == "steps"
    assert config["save_strategy"] == "steps"
    assert config["eval_steps"] == config["save_steps"] == 19
    assert config["save_total_limit"] == 4
    assert config["load_best_model_at_end"] is True
    assert config["metric_for_best_model"] == "eval_loss"
    assert config["greater_is_better"] is False
    assert config["output_dir"].endswith("/runs/tool-sft-lora-v2")


def test_v2_runtime_contract_is_exact_and_shared_by_train_and_eval() -> None:
    import formal_preflight

    contract_path = REPO_ROOT / "deliverables/week6/day31/configs/runtime_versions_v2.json"
    contract = formal_preflight.load_runtime_contract(contract_path)
    observed = dict(contract["packages"])

    result = formal_preflight.verify_runtime_versions(
        contract, resolver=lambda distribution: observed[distribution]
    )

    assert result["status"] == "ready"
    assert contract["consumers"] == ["trainer", "evaluator"]
    assert contract["llamafactory"]["version"] == "0.9.3"
    assert contract["llamafactory"]["git_revision"] == (
        "ca75f1edf3cb50343ed1c98605141c3e22075b5f"
    )
    assert set(contract["packages"]) >= {
        "torch",
        "transformers",
        "peft",
        "datasets",
        "bitsandbytes",
        "accelerate",
        "langchain",
        "langchain-core",
        "langgraph",
        "pydantic",
    }


def test_v2_runtime_contract_rejects_one_package_mismatch() -> None:
    import formal_preflight

    contract_path = REPO_ROOT / "deliverables/week6/day31/configs/runtime_versions_v2.json"
    contract = formal_preflight.load_runtime_contract(contract_path)
    observed = dict(contract["packages"])
    observed["transformers"] = "0.0.0"

    result = formal_preflight.verify_runtime_versions(
        contract, resolver=lambda distribution: observed[distribution]
    )

    assert result["status"] == "blocked"
    assert result["mismatches"] == {
        "transformers": {
            "expected": contract["packages"]["transformers"],
            "observed": "0.0.0",
        }
    }


def test_formal_preflight_cli_exposes_v2_runtime_and_data_contracts() -> None:
    completed = subprocess.run(
        [sys.executable, str(PREFLIGHT), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--profile" in completed.stdout
    assert "--runtime-contract" in completed.stdout
    assert "--dataset-manifest" in completed.stdout


def test_preflight_prefers_manifest_with_recomputed_inventory_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import day31_harness

    model_dir = tmp_path / "merged"
    manifest = _manifest(model_dir)
    digest = _manifest_digest(manifest)
    monkeypatch.setattr(day31_harness, "WEEK4_MERGED_MANIFEST_SHA256", digest)
    lineage = resolve_scheme_a_lineage(
        merged_manifest_path=manifest, merged_model_dir=model_dir,
        base_manifest_path=None, base_model_dir=None, dpo_adapter_dir=None,
        day30_config_path=REPO_ROOT / "deliverables/week6/day30/source/configs/agent_config.json",
    )
    assert lineage["load_mode"] == "verified_week4_merged"
    assert lineage["merged_manifest_sha256"] == digest


def test_preflight_rejects_manifest_file_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import day31_harness

    model_dir = tmp_path / "merged"
    manifest = _manifest(model_dir)
    monkeypatch.setattr(day31_harness, "WEEK4_MERGED_MANIFEST_SHA256", _manifest_digest(manifest))
    (model_dir / "config.json").write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="model manifest mismatch"):
        resolve_scheme_a_lineage(
            merged_manifest_path=manifest, merged_model_dir=model_dir,
            base_manifest_path=None, base_model_dir=None, dpo_adapter_dir=None,
            day30_config_path=REPO_ROOT / "deliverables/week6/day30/source/configs/agent_config.json",
        )


def test_preflight_rejects_self_declared_manifest_identity(tmp_path: Path) -> None:
    model_dir = tmp_path / "merged"
    manifest = _manifest(model_dir, WEEK4_MERGED_MANIFEST_SHA256)
    with pytest.raises(ValueError, match="aggregate manifest identity mismatch"):
        resolve_scheme_a_lineage(
            merged_manifest_path=manifest, merged_model_dir=model_dir,
            base_manifest_path=None, base_model_dir=None, dpo_adapter_dir=None,
            day30_config_path=REPO_ROOT / "deliverables/week6/day30/source/configs/agent_config.json",
        )


def test_preflight_records_fake_tokenizer_lengths_and_parser_failure() -> None:
    row = {
        "conversations": [
            {"from": "human", "value": "calc"},
            {"from": "function_call", "value": '{"name":"calculator","arguments":{"expression":"1+1"}}'},
            {"from": "observation", "value": '{"status":"success"}'},
            {"from": "gpt", "value": "2"},
        ],
        "tools": "[]",
    }
    import hashlib, json
    receipt = {"command": ["llamafactory-cli", "train"], "version": "x", "git_revision": "abc", "exit_code": 0, "dataset_sha256": {"train": hashlib.sha256(json.dumps([row], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "eval_frozen": hashlib.sha256(json.dumps([row], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}}
    receipt["receipt_sha256"] = hashlib.sha256(json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report = build_preflight_report(lineage={"scheme": "A"}, train_rows=[row], eval_rows=[row], tokenizer=FakeTokenizer(), cutoff=8, official_parser_receipt=receipt)
    assert report["token_length_stats"]["train"]["count"] == 1
    assert report["local_schema_ok"] is True
    assert report["official_parser_smoke"]["status"] == "blocked"
    malformed = dict(row, conversations=row["conversations"][:-1])
    blocked = build_preflight_report(lineage={"scheme": "A"}, train_rows=[malformed], eval_rows=[row], tokenizer=FakeTokenizer(), cutoff=8, official_parser_receipt=receipt)
    assert blocked["status"] == "blocked"


def test_frozen_evaluator_scores_tool_sequence_and_observation_integrity() -> None:
    row = {
        "id": "case", "conversations": [
            {"from": "human", "value": "calculate"},
            {"from": "function_call", "value": '{"name":"calculator","arguments":{"expression":"1 + 1"}}'},
            {"from": "observation", "value": "{}"}, {"from": "gpt", "value": "2"},
        ],
    }
    model = FakeModel({"status": "success", "final_answer": "2", "trace": [{"action": "calculator", "action_input": {"expression": "1 + 1"}, "observation": {"status": "success", "data": {"value": 2}, "error_code": None}}]})
    report = evaluate_frozen_cases([row], model.invoke, {"do_sample": False, "seed": 42}, {"model": "fake"})
    assert report["metrics"]["full_tool_sequence"] == 1.0
    assert report["metrics"]["no_hallucinated_observations"] == 1.0


def test_frozen_evaluator_flags_loop_wrong_tool_and_hallucination() -> None:
    row = {
        "id": "case", "conversations": [
            {"from": "human", "value": "calculate"},
            {"from": "function_call", "value": '{"name":"calculator","arguments":{"expression":"1 + 1"}}'},
            {"from": "observation", "value": "{}"}, {"from": "gpt", "value": "2"},
        ],
    }
    response = {"status": "failed", "final_answer": "", "trace": [
        {"action": "knowledge_retrieval", "action_input": {"query": "x"}, "observation": {"status": "success"}},
        {"action": "knowledge_retrieval", "action_input": {"query": "x"}, "observation": {"status": "success"}},
    ]}
    report = evaluate_frozen_cases([row], FakeModel(response).invoke, {"do_sample": False, "seed": 42}, {"model": "fake"})
    checks = report["case_results"][0]["checks"]
    assert checks["first_tool_choice"] is False
    assert checks["no_loops"] is False
    assert checks["no_hallucinated_observations"] is False


def _expected_success_result(row: dict[str, object]) -> dict[str, object]:
    conversations = row["conversations"]
    trace = []
    for index, message in enumerate(conversations):
        if message["from"] == "function_call":
            call = json.loads(message["value"])
            trace.append({
                "action": call["name"],
                "action_input": call["arguments"],
                "observation": json.loads(conversations[index + 1]["value"]),
            })
    return {
        "status": "success",
        "final_answer": conversations[-1]["value"],
        "trace": trace,
    }


def test_recovery_argument_correctness_is_not_applicable_and_uses_24_case_denominator() -> None:
    rows = json.loads(
        (REPO_ROOT / "deliverables/week6/day31/source/data/tool_sft_eval_frozen.json").read_text(encoding="utf-8")
    )
    report = evaluate_frozen_cases(rows, _expected_success_result, {"seed": 42}, {"model": "fake"})
    recovery = next(case for case in report["case_results"] if case["case_id"] == "eval-recovery-02")
    assert recovery["checks"]["argument_correctness"] is None
    assert recovery["passed"] is True
    assert report["applicable_counts"]["argument_correctness"] == 24
    assert report["metrics"]["argument_correctness"] == 1.0


def test_recovery_case_rejects_successful_call_that_never_exercises_expected_error() -> None:
    rows = json.loads(
        (REPO_ROOT / "deliverables/week6/day31/source/data/tool_sft_eval_frozen.json").read_text(encoding="utf-8")
    )
    row = next(item for item in rows if item["id"] == "eval-recovery-01")
    score = score_frozen_case(
        row,
        {
            "status": "success",
            "final_answer": "计算成功，流程结束。",
            "trace": [
                {
                    "action": "calculator",
                    "action_input": {"expression": "1 + 2"},
                    "observation": {"status": "success", "data": {"value": 3}, "error_code": None},
                }
            ],
        },
    )
    assert score["checks"]["first_tool_choice"] is True
    assert score["checks"]["full_tool_sequence"] is True
    assert score["checks"]["argument_correctness"] is None
    assert score["checks"]["no_hallucinated_observations"] is True
    assert score["checks"]["completion"] is False
    assert score["passed"] is False


@pytest.mark.parametrize(
    ("case_id", "malformed_final"),
    [
        ("eval-recovery-02", '{"name":"calculator","arguments":{"expression":"lambda x: x"}}'),
        ("eval-recovery-04", '<tool_call>{"name":"knowledge_retrieval","arguments":{"query":"松风音箱"}}'),
    ],
)
def test_completion_rejects_unresolved_or_malformed_tool_syntax(case_id: str, malformed_final: str) -> None:
    rows = json.loads(
        (REPO_ROOT / "deliverables/week6/day31/source/data/tool_sft_eval_frozen.json").read_text(encoding="utf-8")
    )
    row = next(item for item in rows if item["id"] == case_id)
    result = _expected_success_result(row)
    result["final_answer"] = malformed_final
    score = score_frozen_case(row, result)
    assert score["checks"]["full_tool_sequence"] is True
    assert score["checks"]["completion"] is False
    assert score["passed"] is False


def test_v2_budget_score_rejects_observation_argument_and_final_conflict() -> None:
    """A fluent answer cannot pass when it conflicts with the executed calculator call."""
    rows = json.loads((V2_DATA / "tool_sft_test_v2.json").read_text(encoding="utf-8"))
    row = next(item for item in rows if item["category"] == "knowledge_plus_calculator")
    result = _expected_success_result(row)
    expected = row["success_predicates"]["expected_answer"]
    wrong_expression = f"{expected['price_cny'] + 100} + {expected['shipping_cny']}"
    result["trace"][1]["action_input"] = {"expression": wrong_expression}
    from build_tool_sft_dataset import execute_tool

    result["trace"][1]["observation"] = execute_tool(
        "calculator", {"expression": wrong_expression}, V2_KNOWLEDGE_BASE
    )
    result["final_answer"] = (
        f"{expected['name']}商品价 {expected['price_cny']} 元，运费 {expected['shipping_cny']} 元，"
        f"总价 {expected['total_cny']} 元，预算 {expected['budget_cny']} 元，"
        f"关系为 {expected['relation']}，差额 {expected['difference_cny']} 元。"
    )

    score = score_case_v2(row, result, V2_KNOWLEDGE_BASE)

    assert score["checks"]["argument_correctness"] is False
    assert score["checks"]["observation_integrity"] is True
    assert score["checks"]["final_answer_grounded"] is False
    assert score["passed"] is False


def test_v2_semantic_score_rejects_nonempty_budget_answer_missing_required_facts() -> None:
    """Completion text alone does not prove the business task was answered."""
    rows = json.loads((V2_DATA / "tool_sft_test_v2.json").read_text(encoding="utf-8"))
    row = next(item for item in rows if item["category"] == "knowledge_plus_calculator")
    result = _expected_success_result(row)
    result["final_answer"] = "查询和计算已经完成。"

    score = score_case_v2(row, result, V2_KNOWLEDGE_BASE)

    assert score["checks"]["format_valid"] is True
    assert score["checks"]["semantic_correctness"] is False
    assert score["checks"]["final_answer_grounded"] is False
    assert score["passed"] is False


def test_v2_score_marks_truncated_tool_json_as_format_failure() -> None:
    """An incomplete tool call must not be accepted as a terminal answer."""
    rows = json.loads((V2_DATA / "tool_sft_test_v2.json").read_text(encoding="utf-8"))
    row = next(item for item in rows if item["category"] == "calculator")
    result = _expected_success_result(row)
    result["final_answer"] = '<tool_call>{"name":"calculator","arguments":{"expression":"1 +'

    score = score_case_v2(row, result, V2_KNOWLEDGE_BASE)

    assert score["checks"]["format_valid"] is False
    assert score["checks"]["completion"] is False
    assert score["passed"] is False


def test_v2_recovery_arguments_are_scored_and_safe_termination_is_required() -> None:
    """Self-contained recovery cases use exact arguments in the formal denominator."""
    rows = json.loads((V2_DATA / "tool_sft_test_v2.json").read_text(encoding="utf-8"))
    row = next(item for item in rows if item["category"] == "recovery_or_termination")
    result = _expected_success_result(row)

    score = score_case_v2(row, result, V2_KNOWLEDGE_BASE)

    assert score["checks"]["argument_correctness"] is True
    assert score["checks"]["safe_termination"] is True
    assert score["checks"]["semantic_correctness"] is True
    assert score["passed"] is True


def test_v2_expected_traces_pass_every_semantic_metric() -> None:
    """The hand-derived dataset answers define a complete executable scoring contract."""
    rows = json.loads((V2_DATA / "tool_sft_test_v2.json").read_text(encoding="utf-8"))

    report = evaluate_frozen_cases(
        rows,
        _expected_success_result,
        {"do_sample": False, "seed": 42},
        {"model": "expected-trace"},
        knowledge_base_path=V2_KNOWLEDGE_BASE,
    )

    assert report["schema_version"] == "2.0"
    assert report["case_count"] == 100
    assert set(report["metrics"]) == {
        "first_tool_choice",
        "full_tool_sequence",
        "argument_schema_valid",
        "argument_correctness",
        "observation_integrity",
        "format_valid",
        "semantic_correctness",
        "final_answer_grounded",
        "safe_termination",
        "completion",
        "no_loops",
    }
    assert all(value == 1.0 for value in report["metrics"].values())
    assert all(case["passed"] for case in report["case_results"])


def test_pre_post_summary_builder_hash_binds_raw_inputs(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    post = tmp_path / "post.json"
    output = tmp_path / "summary.json"
    common = {
        "schema_version": "1.0", "status": "completed", "case_count": 1,
        "generation_config": {"seed": 42}, "protocol": {"frozen_eval_file_sha256": "a" * 64},
        "case_results": [{"case_id": "case", "passed": True, "checks": {"completion": True}}],
    }
    _write(baseline, json.dumps({**common, "model_identity": {"mode": "baseline"}, "metrics": {"completion": 1.0}, "applicable_counts": {"completion": 1}}))
    _write(post, json.dumps({**common, "model_identity": {"mode": "post"}, "metrics": {"completion": 1.0}, "applicable_counts": {"completion": 1}}))
    completed = subprocess.run(
        [sys.executable, str(SUMMARY_BUILDER), "--baseline", str(baseline), "--post", str(post), "--output", str(output)],
        cwd=REPO_ROOT, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["baseline_sha256"] == hashlib.sha256(baseline.read_bytes()).hexdigest()
    assert summary["post_sha256"] == hashlib.sha256(post.read_bytes()).hexdigest()


def test_evidence_collector_keeps_not_run_without_real_files() -> None:
    summary, adapter = collect_training_evidence(None, None, None)
    assert summary["status"] == "not_run"
    assert adapter["status"] == "not_run"


def test_evidence_collector_cli_returns_nonzero_for_not_run(tmp_path: Path) -> None:
    summary, adapter = tmp_path / "summary.json", tmp_path / "adapter.json"
    completed = subprocess.run(
        [sys.executable, str(COLLECTOR), "--summary-output", str(summary), "--adapter-output", str(adapter)],
        cwd=REPO_ROOT, text=True, capture_output=True, check=False,
    )
    assert completed.returncode != 0
    assert json.loads(summary.read_text(encoding="utf-8"))["status"] == "not_run"
    assert json.loads(adapter.read_text(encoding="utf-8"))["status"] == "not_run"


def test_evidence_collector_rejects_forged_minimal_ready_preflight(tmp_path: Path) -> None:
    trainer = tmp_path / "trainer"
    _write(trainer / "trainer_state.json", '{"log_history":[{"loss":1.0}]}')
    (trainer / "adapter_model.safetensors").parent.mkdir(parents=True, exist_ok=True)
    (trainer / "adapter_model.safetensors").write_bytes(_valid_safetensors())
    _write(trainer / "adapter_config.json", '{"peft_type":"LORA","task_type":"CAUSAL_LM","r":8,"lora_alpha":16,"target_modules":["q_proj"],"base_model_name_or_path":"model"}')
    effective, preflight = tmp_path / "effective.yaml", tmp_path / "preflight.json"
    launch, exit_status, trainable = tmp_path / "launch.txt", tmp_path / "exit.json", tmp_path / "trainable.json"
    _write(effective, f"stage: sft\ndo_train: true\ndataset: week6_tool_sft_train\neval_dataset: week6_tool_sft_eval_frozen\nmodel_name_or_path: model\noutput_dir: {trainer}\nquantization_bit: 4\nquantization_type: nf4\ndouble_quantization: true\ngradient_checkpointing: true\n")
    _write(preflight, '{"status":"ready","lineage":{"scheme":"A"}}')
    import hashlib, json
    launch_value = {"argv":["llamafactory-cli","train",str(effective.resolve())],"cwd":"/tmp","config_sha256":hashlib.sha256(effective.read_bytes()).hexdigest()}
    _write(launch, json.dumps(launch_value))
    _write(exit_status, json.dumps({"exit_code":0,"launch_sha256":hashlib.sha256(launch.read_bytes()).hexdigest(),"config_sha256":hashlib.sha256(effective.read_bytes()).hexdigest()}))
    _write(trainable, '{"names":["lora_a"]}')
    summary, adapter = collect_training_evidence(trainer, effective, preflight, launch, exit_status, trainable)
    assert summary["status"] == "not_run"
    assert adapter["status"] == "not_run"


def test_evidence_collector_collects_with_verified_merged_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import day31_harness

    trainer = tmp_path / "trainer"
    model = tmp_path / "merged"
    manifest = _manifest(model)
    manifest_digest = _manifest_digest(manifest)
    monkeypatch.setattr(day31_harness, "WEEK4_MERGED_MANIFEST_SHA256", manifest_digest)
    official_calls: list[object] = []

    def verified_official_receipt(*args, **kwargs):
        official_calls.append((args, kwargs))
        return True

    monkeypatch.setattr(day31_harness, "verify_official_parser_receipt", verified_official_receipt)
    preflight = tmp_path / "preflight.json"
    _write(preflight, json.dumps({
        "status": "ready",
        "lineage": {
            "scheme": "A",
            "load_mode": "verified_week4_merged",
            "model_dir": str(model.resolve()),
            "merged_manifest_path": str(manifest.resolve()),
            "merged_manifest_sha256": manifest_digest,
        },
        "official_parser_smoke": {"test": "verified through monkeypatched official API"},
    }))
    _write(trainer / "trainer_state.json", '{"log_history":[{"loss":1.0}]}')
    (trainer / "adapter_model.safetensors").parent.mkdir(parents=True, exist_ok=True)
    (trainer / "adapter_model.safetensors").write_bytes(_valid_safetensors())
    _write(trainer / "adapter_config.json", json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 8, "lora_alpha": 16,
        "target_modules": ["q_proj"], "base_model_name_or_path": str(model.resolve()),
    }))
    effective = tmp_path / "effective.yaml"
    launch, exit_status, trainable = tmp_path / "launch.json", tmp_path / "exit.json", tmp_path / "trainable.json"
    _write(effective, f"stage: sft\ndo_train: true\ndataset: week6_tool_sft_train\neval_dataset: week6_tool_sft_eval_frozen\nmodel_name_or_path: {model.resolve()}\noutput_dir: {trainer}\nquantization_bit: 4\nquantization_type: nf4\ndouble_quantization: true\ngradient_checkpointing: true\n")
    launch_value = {"argv":["llamafactory-cli","train",str(effective.resolve())],"cwd":"/tmp","config_sha256":hashlib.sha256(effective.read_bytes()).hexdigest()}
    _write(launch, json.dumps(launch_value))
    _write(exit_status, json.dumps({"exit_code":0,"launch_sha256":hashlib.sha256(launch.read_bytes()).hexdigest(),"config_sha256":hashlib.sha256(effective.read_bytes()).hexdigest()}))
    _write(trainable, '{"names":["lora_a"]}')
    monkeypatch.chdir(tmp_path)
    relative_trainer = Path("trainer")
    summary, adapter = collect_training_evidence(relative_trainer, effective, preflight, launch, exit_status, trainable)
    assert summary["status"] == "collected"
    assert adapter["status"] == "collected"
    assert len(official_calls) == 1

    _write(
        effective,
        f"stage: sft\ndo_train: true\ndataset: week6_tool_sft_train_v2\n"
        f"eval_dataset: week6_tool_sft_dev_v2\nmodel_name_or_path: {model.resolve()}\n"
        f"output_dir: {trainer}\nquantization_bit: 4\nquantization_type: nf4\n"
        "double_quantization: true\ngradient_checkpointing: true\n",
    )
    launch_value["config_sha256"] = hashlib.sha256(effective.read_bytes()).hexdigest()
    _write(launch, json.dumps(launch_value))
    _write(
        exit_status,
        json.dumps(
            {
                "exit_code": 0,
                "launch_sha256": hashlib.sha256(launch.read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256(effective.read_bytes()).hexdigest(),
            }
        ),
    )
    summary_v2, adapter_v2 = collect_training_evidence(
        relative_trainer, effective, preflight, launch, exit_status, trainable
    )
    assert summary_v2["status"] == "collected"
    assert adapter_v2["status"] == "collected"


def test_evidence_collector_rejects_placeholder_adapter_bytes(tmp_path: Path) -> None:
    trainer = tmp_path / "trainer"
    _write(trainer / "trainer_state.json", '{"log_history":[{"loss":1.0}]}')
    _write(trainer / "adapter_model.safetensors", "weights")
    _write(trainer / "adapter_config.json", '{"peft_type":"LORA","task_type":"CAUSAL_LM","r":8,"lora_alpha":16,"target_modules":["q_proj"],"base_model_name_or_path":"model"}')
    effective, preflight = tmp_path / "effective.yaml", tmp_path / "preflight.json"
    launch, exit_status, trainable = tmp_path / "launch.json", tmp_path / "exit.json", tmp_path / "trainable.json"
    _write(effective, f"stage: sft\ndo_train: true\ndataset: week6_tool_sft_train\neval_dataset: week6_tool_sft_eval_frozen\nmodel_name_or_path: model\noutput_dir: {trainer}\nquantization_bit: 4\n")
    _write(preflight, '{"status":"ready","lineage":{"scheme":"A"}}')
    launch_value = {"argv":["llamafactory-cli","train",str(effective.resolve())],"cwd":"/tmp","config_sha256":hashlib.sha256(effective.read_bytes()).hexdigest()}
    _write(launch, json.dumps(launch_value))
    _write(exit_status, json.dumps({"exit_code":0,"launch_sha256":hashlib.sha256(launch.read_bytes()).hexdigest(),"config_sha256":hashlib.sha256(effective.read_bytes()).hexdigest()}))
    _write(trainable, '{"names":["lora_a"]}')
    summary, adapter = collect_training_evidence(trainer, effective, preflight, launch, exit_status, trainable)
    assert summary["status"] == "not_run"
    assert adapter["status"] == "not_run"


def test_evidence_collector_fails_closed_for_empty_preflight(tmp_path: Path) -> None:
    trainer = tmp_path / "trainer"
    _write(trainer / "trainer_state.json", '{"log_history":[{"loss":1.0}]}')
    _write(trainer / "adapter_model.safetensors", "weights")
    _write(trainer / "adapter_config.json", "{}")
    effective, preflight = tmp_path / "effective.yaml", tmp_path / "preflight.json"
    _write(effective, "stage: sft\ndo_train: true\ndataset: week6_tool_sft_train\nmodel_name_or_path: model\n")
    _write(preflight, "{}")
    summary, adapter = collect_training_evidence(trainer, effective, preflight)
    assert summary["status"] == "not_run"
    assert adapter["status"] == "not_run"
