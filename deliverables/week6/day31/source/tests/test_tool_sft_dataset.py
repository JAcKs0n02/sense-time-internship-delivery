from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
BUILD_SCRIPT = REPO_ROOT / "deliverables/week6/day31/source/scripts/build_tool_sft_dataset.py"
VALIDATE_SCRIPT = REPO_ROOT / "deliverables/week6/day31/source/scripts/validate_tool_sft_dataset.py"
sys.path.insert(0, str(BUILD_SCRIPT.parent))


def _v2_bundle() -> dict[str, object]:
    from build_tool_sft_dataset import build_dataset_bundle

    return build_dataset_bundle()


def _build_dataset(tmp_path: Path) -> Path:
    output_directory = tmp_path / "dataset"
    completed = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), "--output-directory", str(output_directory)],
        cwd=REPO_ROOT, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return output_directory


def _validate(data_directory: Path) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    completed = subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT), "--data-directory", str(data_directory)],
        cwd=REPO_ROOT, text=True, capture_output=True, check=False,
    )
    return completed, json.loads(completed.stdout)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_builder_writes_exactly_100_unique_training_rows(tmp_path: Path) -> None:
    """The deterministic builder must produce the required 100-row training split."""
    output_directory = tmp_path / "dataset"
    completed = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), "--output-directory", str(output_directory)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    rows = json.loads((output_directory / "tool_sft_train_100.json").read_text(encoding="utf-8"))
    assert len(rows) == 100
    assert len({row["id"] for row in rows}) == 100


def test_validator_accepts_a_fresh_deterministic_dataset(tmp_path: Path) -> None:
    """Validation gates the frozen split and every real-tool observation."""
    output_directory = tmp_path / "dataset"
    build = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), "--output-directory", str(output_directory)],
        cwd=REPO_ROOT, text=True, capture_output=True, check=False,
    )
    assert build.returncode == 0, build.stderr

    completed = subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT), "--data-directory", str(output_directory)],
        cwd=REPO_ROOT, text=True, capture_output=True, check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["valid"] is True
    assert report["checks"]["reproducible_observations"] is True


def test_train_has_required_category_distribution_and_react_audit_view(tmp_path: Path) -> None:
    """The teaching audit view remains one-to-one with ShareGPT tool messages."""
    output_directory = tmp_path / "dataset"
    completed = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), "--output-directory", str(output_directory)],
        cwd=REPO_ROOT, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    rows = json.loads((output_directory / "tool_sft_train_100.json").read_text(encoding="utf-8"))

    assert {category: sum(row["category"] == category for row in rows) for category in (
        "calculator", "knowledge_retrieval", "code_executor", "knowledge_plus_calculator", "recovery_or_termination",
    )} == {
        "calculator": 20, "knowledge_retrieval": 20, "code_executor": 20,
        "knowledge_plus_calculator": 25, "recovery_or_termination": 15,
    }
    row = next(row for row in rows if row["category"] == "knowledge_plus_calculator")
    assert [message["from"] for message in row["conversations"]] == [
        "human", "function_call", "observation", "function_call", "observation", "gpt",
    ]
    assert all({"Thought", "Action", "Action Input", "Observation"} <= set(step) for step in row["audit"]["react"])


def test_validator_rejects_a_hand_edited_observation(tmp_path: Path) -> None:
    """An observation is evidence only when the frozen real tool can recreate it."""
    output_directory = tmp_path / "dataset"
    build = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), "--output-directory", str(output_directory)],
        cwd=REPO_ROOT, text=True, capture_output=True, check=False,
    )
    assert build.returncode == 0, build.stderr
    train_path = output_directory / "tool_sft_train_100.json"
    rows = json.loads(train_path.read_text(encoding="utf-8"))
    rows[0]["conversations"][2]["value"] = '{"status":"success","data":{"value":999},"error_code":null}'
    train_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT), "--data-directory", str(output_directory)],
        cwd=REPO_ROOT, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 1
    report = json.loads(completed.stdout)
    assert report["valid"] is False
    assert any(error.startswith("nonreproducible_observation") for error in report["errors"])


def test_validator_rejects_shared_semantic_tool_case_despite_new_prompt_wording(tmp_path: Path) -> None:
    """Frozen evaluation must not reuse a train tool-input case behind paraphrase."""
    output_directory = _build_dataset(tmp_path)
    train = json.loads((output_directory / "tool_sft_train_100.json").read_text(encoding="utf-8"))
    evaluation_path = output_directory / "tool_sft_eval_frozen.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation[0]["conversations"][0]["value"] = "完全不同措辞的冻结问题。"
    evaluation[0]["audit"]["prompt"] = "完全不同措辞的冻结问题。"
    evaluation[0]["conversations"][1]["value"] = train[0]["conversations"][1]["value"]
    evaluation[0]["conversations"][2]["value"] = train[0]["conversations"][2]["value"]
    evaluation[0]["audit"]["react"][0]["Action"] = train[0]["audit"]["react"][0]["Action"]
    evaluation[0]["audit"]["react"][0]["Action Input"] = train[0]["audit"]["react"][0]["Action Input"]
    evaluation[0]["audit"]["react"][0]["Observation"] = train[0]["audit"]["react"][0]["Observation"]
    _write_json(evaluation_path, evaluation)

    completed, report = _validate(output_directory)
    assert completed.returncode == 1
    assert report["checks"]["semantic_case_overlap_zero"] is False


def test_validator_rejects_malformed_tools_json_as_a_reported_failure(tmp_path: Path) -> None:
    """An invalid tools column must not crash validation before it writes JSON."""
    output_directory = _build_dataset(tmp_path)
    train_path = output_directory / "tool_sft_train_100.json"
    train = json.loads(train_path.read_text(encoding="utf-8"))
    train[0]["tools"] = "{"
    _write_json(train_path, train)

    completed, report = _validate(output_directory)
    assert completed.returncode == 1
    assert report["valid"] is False
    assert report["checks"]["tools_schema_valid"] is False


def test_validator_rejects_audit_prompt_final_and_thought_drift(tmp_path: Path) -> None:
    """The audit view is evidence only when it fully projects the conversation."""
    output_directory = _build_dataset(tmp_path)
    train_path = output_directory / "tool_sft_train_100.json"
    train = json.loads(train_path.read_text(encoding="utf-8"))
    train[0]["audit"]["prompt"] = "different audit prompt"
    train[0]["audit"]["final"] = "different audit final"
    train[0]["audit"]["react"][0]["Thought"] = " "
    _write_json(train_path, train)

    completed, report = _validate(output_directory)
    assert completed.returncode == 1
    assert report["checks"]["audit_content_projection"] is False


def test_validator_rejects_duplicate_ids_and_wrong_split_membership(tmp_path: Path) -> None:
    """The train/eval boundary is an explicit row-level contract."""
    output_directory = _build_dataset(tmp_path)
    train_path = output_directory / "tool_sft_train_100.json"
    train = json.loads(train_path.read_text(encoding="utf-8"))
    train[1]["id"] = train[0]["id"]
    train[0]["split"] = "eval_frozen"
    _write_json(train_path, train)

    completed, report = _validate(output_directory)
    assert completed.returncode == 1
    assert report["checks"]["unique_train_ids"] is False
    assert report["checks"]["split_membership"] is False


def test_validator_rejects_determinism_and_manifest_file_hash_drift(tmp_path: Path) -> None:
    """A changed committed byte stream cannot still claim frozen reconstruction."""
    output_directory = _build_dataset(tmp_path)
    train_path = output_directory / "tool_sft_train_100.json"
    train = json.loads(train_path.read_text(encoding="utf-8"))
    train[0]["conversations"][-1]["value"] = "changed final"
    train[0]["audit"]["final"] = "changed final"
    _write_json(train_path, train)

    completed, report = _validate(output_directory)
    assert completed.returncode == 1
    assert report["checks"]["deterministic_rebuild"] is False
    assert report["checks"]["manifest_file_sha256s"] is False


def test_validator_rejects_bad_json_arguments_and_recovery_loop(tmp_path: Path) -> None:
    """Malformed calls and repeated recovery actions cannot pass as safe termination."""
    output_directory = _build_dataset(tmp_path)
    train_path = output_directory / "tool_sft_train_100.json"
    train = json.loads(train_path.read_text(encoding="utf-8"))
    train[0]["conversations"][1]["value"] = "{"
    recovery = next(row for row in train if row["category"] == "recovery_or_termination")
    call = dict(recovery["conversations"][1])
    observation = dict(recovery["conversations"][2])
    recovery["conversations"][-1:-1] = [call, observation]
    recovery["audit"]["react"].append(dict(recovery["audit"]["react"][0]))
    _write_json(train_path, train)

    completed, report = _validate(output_directory)
    assert completed.returncode == 1
    assert report["checks"]["valid_json_arguments"] is False
    assert report["checks"]["recovery_termination_semantics"] is False


def test_validator_enforces_actual_zero_shipping_and_code_only_sequences(tmp_path: Path) -> None:
    """Day30 regressions are sequence checks, not marker fields."""
    output_directory = _build_dataset(tmp_path)
    evaluation_path = output_directory / "tool_sft_eval_frozen.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    zero_shipping = next(row for row in evaluation if row["id"] == "eval-aurora-zero-shipping-calculator-required")
    del zero_shipping["conversations"][3:5]
    del zero_shipping["audit"]["react"][1]
    code_only = next(row for row in evaluation if row["id"] == "eval-code-no-unnecessary-knowledge")
    extra_call = next(row for row in evaluation if row["category"] == "knowledge_plus_calculator")["conversations"][1:3]
    code_only["conversations"][-1:-1] = extra_call
    code_only["audit"]["react"].append({
        "Thought": "This should not be allowed.", "Action": "knowledge_retrieval",
        "Action Input": json.loads(extra_call[0]["value"])["arguments"],
        "Observation": json.loads(extra_call[1]["value"]),
    })
    _write_json(evaluation_path, evaluation)

    completed, report = _validate(output_directory)
    assert completed.returncode == 1
    assert report["checks"]["zero_shipping_calculator_required"] is False
    assert report["checks"]["code_only_no_unnecessary_retrieval"] is False


def test_v2_bundle_has_core_expanded_dev_and_final_test_splits() -> None:
    """A missing split or undersized final set would make model selection unreliable."""
    bundle = _v2_bundle()

    assert len(bundle["train_core_100"]) == 100
    assert len(bundle["train_expanded"]) >= 200
    assert len(bundle["dev_v2"]) >= 40
    assert len(bundle["test_v2"]) >= 100
    all_rows = [
        *bundle["train_core_100"],
        *bundle["train_expanded"],
        *bundle["dev_v2"],
        *bundle["test_v2"],
    ]
    assert len({row["id"] for row in all_rows}) == len(all_rows)
    assert all(
        not row["conversations"][0]["value"].startswith(("训练", "冻结"))
        for row in all_rows
    )


def test_v2_direct_inputs_are_visible_and_derived_inputs_are_grounded() -> None:
    """No expected payload may exist only in evaluator-private data."""
    bundle = _v2_bundle()

    for split_name in ("train_core_100", "train_expanded", "dev_v2", "test_v2"):
        for row in bundle[split_name]:
            prompt = row["conversations"][0]["value"]
            calls = [
                json.loads(message["value"])
                for message in row["conversations"]
                if message["from"] == "function_call"
            ]
            observations = [
                json.loads(message["value"])
                for message in row["conversations"]
                if message["from"] == "observation"
            ]
            first_value = next(iter(calls[0]["arguments"].values()))
            assert str(first_value) in prompt, row["id"]
            if len(calls) == 2:
                retrieved = observations[0]["data"]
                expected_expression = (
                    f"{retrieved['price_cny']} + {retrieved['shipping_cny']}"
                )
                assert calls[1] == {
                    "name": "calculator",
                    "arguments": {"expression": expected_expression},
                }


def test_v2_train_dev_test_tool_cases_are_semantically_disjoint() -> None:
    """Entity-level leakage would let the model memorize catalog values."""
    bundle = _v2_bundle()

    def product_ids(rows: list[dict[str, object]]) -> set[str]:
        ids: set[str] = set()
        for row in rows:
            if row["category"] != "knowledge_plus_calculator":
                continue
            observation = json.loads(row["conversations"][2]["value"])
            ids.add(observation["data"]["product_id"])
        return ids

    train_ids = product_ids([
        *bundle["train_core_100"], *bundle["train_expanded"]
    ])
    dev_ids = product_ids(bundle["dev_v2"])
    test_ids = product_ids(bundle["test_v2"])
    assert train_ids
    assert dev_ids
    assert test_ids
    assert train_ids.isdisjoint(dev_ids)
    assert train_ids.isdisjoint(test_ids)
    assert dev_ids.isdisjoint(test_ids)


def test_v2_cli_writes_versioned_bundle_and_manifest(tmp_path: Path) -> None:
    """The formal CLI must materialize every split and bind its bytes."""
    output_directory = tmp_path / "v2"
    completed = subprocess.run(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--profile",
            "v2",
            "--output-directory",
            str(output_directory),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    expected = {
        "tool_sft_train_core_100.json",
        "tool_sft_train_expanded.json",
        "tool_sft_train_v2.json",
        "tool_sft_dev_v2.json",
        "tool_sft_test_v2.json",
        "dataset_manifest_v2.json",
    }
    assert {path.name for path in output_directory.iterdir()} == expected
    manifest = json.loads(
        (output_directory / "dataset_manifest_v2.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "2.0"
    assert manifest["splits"]["train_core_100"]["row_count"] == 100
    assert manifest["splits"]["train_v2"]["row_count"] >= 300
    assert manifest["splits"]["test_v2"]["row_count"] >= 100


def test_v2_validator_reports_self_contained_and_disjoint_splits(tmp_path: Path) -> None:
    """The release gate must recompute the v2 data contract from real files."""
    output_directory = tmp_path / "v2"
    build = subprocess.run(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--profile",
            "v2",
            "--output-directory",
            str(output_directory),
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    assert build.returncode == 0

    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATE_SCRIPT),
            "--profile",
            "v2",
            "--data-directory",
            str(output_directory),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["valid"] is True
    assert report["checks"]["self_contained_inputs"] is True
    assert report["checks"]["split_product_ids_disjoint"] is True
    assert report["checks"]["final_test_minimum_100"] is True
    assert report["checks"]["manifest_file_sha256s"] is True
