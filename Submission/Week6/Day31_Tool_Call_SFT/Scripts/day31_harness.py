"""Pure local helpers for Day31 AutoDL preflight, scoring, and evidence capture."""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable


WEEK4_MERGED_MANIFEST_SHA256 = "aca9a2932ed427f8fc6e9962e70274eb286a09c446d177af2bc00f6df1675e4c"
LLAMAFACTORY_VERSION = "0.9.3"
LLAMAFACTORY_REVISION = "ca75f1edf3cb50343ed1c98605141c3e22075b5f"
DAY31_SOURCE = Path(__file__).resolve().parents[1]


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _verify_manifest_directory(manifest: dict[str, Any], model_dir: Path) -> None:
    if manifest.get("valid") is not True or not isinstance(manifest.get("files"), list):
        raise ValueError("invalid model manifest")
    entries = manifest["files"]
    if manifest.get("file_count") != len(entries) or not entries:
        raise ValueError("model manifest file_count mismatch")
    declared_paths: list[str] = []
    for entry in entries:
        relative = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("unsafe model manifest path")
        declared_paths.append(relative)
    if declared_paths != sorted(declared_paths) or len(set(declared_paths)) != len(declared_paths):
        raise ValueError("model manifest paths must be unique and sorted")
    if manifest.get("model_dir") and Path(str(manifest["model_dir"])).resolve() != model_dir.resolve():
        raise ValueError("model manifest directory mismatch")
    actual_paths = sorted(str(path.relative_to(model_dir)) for path in model_dir.rglob("*") if path.is_file())
    if actual_paths != declared_paths:
        raise ValueError("model manifest file set mismatch")
    aggregate = hashlib.sha256()
    total_bytes = 0
    for entry in entries:
        relative = entry["path"]
        candidate = model_dir / relative
        if not candidate.is_file() or candidate.stat().st_size != entry.get("bytes") or sha256_file(candidate) != entry.get("sha256"):
            raise ValueError(f"model manifest mismatch: {relative}")
        aggregate.update(f"{entry['sha256']}  {relative}\n".encode("utf-8"))
        total_bytes += candidate.stat().st_size
    if manifest.get("total_bytes") is not None and manifest["total_bytes"] != total_bytes:
        raise ValueError("model manifest total_bytes mismatch")
    if manifest.get("manifest_sha256") != aggregate.hexdigest():
        raise ValueError("aggregate manifest identity mismatch")


def resolve_scheme_a_lineage(
    *,
    merged_manifest_path: Path | None,
    merged_model_dir: Path | None,
    base_manifest_path: Path | None,
    base_model_dir: Path | None,
    dpo_adapter_dir: Path | None,
    day30_config_path: Path,
    day30_preflight_path: Path | None = None,
) -> dict[str, Any]:
    """Prefer the verified Week4 merged model, else prove base-plus-DPO lineage."""
    merged_error: str | None = None
    if merged_manifest_path and merged_model_dir and merged_manifest_path.is_file() and merged_model_dir.is_dir():
        manifest = _load_json(merged_manifest_path)
        if manifest.get("manifest_sha256") == WEEK4_MERGED_MANIFEST_SHA256:
            try:
                _verify_manifest_directory(manifest, merged_model_dir)
                return {
                    "scheme": "A",
                    "load_mode": "verified_week4_merged",
                    "model_dir": str(merged_model_dir.resolve()),
                    "merged_manifest_path": str(merged_manifest_path.resolve()),
                    "merged_manifest_sha256": WEEK4_MERGED_MANIFEST_SHA256,
                }
            except ValueError as error:
                merged_error = str(error)
        else:
            merged_error = "Week4 merged manifest identity mismatch"

    if not (base_manifest_path and base_model_dir and dpo_adapter_dir):
        detail = f" ({merged_error})" if merged_error else ""
        raise ValueError(f"Scheme A lineage unavailable: provide verified Week4 merged model or Week3 base plus DPO adapter{detail}")
    config = _load_json(day30_config_path)
    preflight_path = day30_preflight_path or day30_config_path.parents[1] / "results/model_preflight.json"
    authoritative = _load_json(preflight_path)
    manifest = _load_json(base_manifest_path)
    if manifest.get("manifest_sha256") != config.get("base_archive_manifest_sha256"):
        raise ValueError("Week3 base manifest hash mismatch")
    _verify_manifest_directory(manifest, base_model_dir)
    adapter_model = dpo_adapter_dir / "adapter_model.safetensors"
    adapter_config = dpo_adapter_dir / "adapter_config.json"
    if not adapter_model.is_file() or not adapter_config.is_file():
        raise ValueError("DPO adapter files missing")
    if adapter_model.stat().st_size != config.get("expected_adapter_bytes") or sha256_file(adapter_model) != config.get("expected_adapter_sha256"):
        raise ValueError("DPO adapter hash mismatch")
    if sha256_file(adapter_config) != authoritative.get("adapter", {}).get("config_sha256"):
        raise ValueError("DPO adapter config hash mismatch")
    if authoritative.get("adapter_source_manifest_sha256") is None or authoritative.get("base_archive_manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("Day30 source/lineage evidence mismatch")
    return {
        "scheme": "A",
        "load_mode": "week3_base_plus_verified_dpo_adapter",
        "base_model_dir": str(base_model_dir.resolve()),
        "base_manifest_path": str(base_manifest_path.resolve()),
        "base_manifest_sha256": manifest["manifest_sha256"],
        "dpo_adapter_dir": str(dpo_adapter_dir.resolve()),
        "dpo_adapter_sha256": sha256_file(adapter_model),
        "dpo_adapter_config_sha256": sha256_file(adapter_config),
        "day30_model_preflight_sha256": sha256_file(preflight_path),
        "day30_adapter_source_manifest_sha256": authoritative["adapter_source_manifest_sha256"],
    }


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _parser_compatible(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        conversations = row.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 4:
            return False
        roles = [message.get("from") for message in conversations if isinstance(message, dict)]
        if roles != ["human", *[role for _ in range((len(roles) - 2) // 2) for role in ("function_call", "observation")], "gpt"]:
            return False
        try:
            tools = json.loads(row["tools"])
            if not isinstance(tools, list):
                return False
            for index in range(1, len(conversations) - 1, 2):
                call = json.loads(conversations[index]["value"])
                if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), dict):
                    return False
        except (KeyError, TypeError, json.JSONDecodeError):
            return False
    return True


def token_length_stats(rows: list[dict[str, Any]], tokenizer: Any, cutoff: int) -> dict[str, Any]:
    lengths: list[int] = []
    for row in rows:
        messages = []
        for message in row["conversations"]:
            role = {"human": "user", "gpt": "assistant", "function_call": "assistant", "observation": "tool"}[message["from"]]
            messages.append({"role": role, "content": message["value"]})
        tools = json.loads(row["tools"])
        try:
            tokenized = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=False)
        except TypeError:
            tokenized = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        length = len(tokenized[0]) if isinstance(tokenized, list) and tokenized and isinstance(tokenized[0], list) else len(tokenized)
        lengths.append(int(length))
    ordered = sorted(lengths)
    recommendation = max(ordered[-1], ordered[(len(ordered) * 95 - 1) // 100]) if ordered else 0
    return {
        "count": len(lengths), "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
        "p95": ordered[(len(ordered) * 95 - 1) // 100] if ordered else 0,
        "over_cutoff_count": sum(length > cutoff for length in lengths),
        "cutoff": cutoff,
        "recommended_cutoff": min(max(cutoff, recommendation), 8192),
    }


def _official_parser_provenance() -> dict[str, Any]:
    """Resolve the installed official parser implementation, never a CLI claim."""
    package = importlib.import_module("llamafactory")
    modules = [
        importlib.import_module("llamafactory.data.parser"),
        importlib.import_module("llamafactory.data.converter"),
        importlib.import_module("llamafactory.data.loader"),
    ]
    package_root = Path(inspect.getfile(package)).resolve().parent
    revision = subprocess.check_output(
        ["git", "-C", str(package_root), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "version": importlib.metadata.version("llamafactory"),
        "git_revision": revision,
        "official_module_hashes": {
            module.__name__: {
                "path": str(Path(inspect.getfile(module)).resolve()),
                "sha256": sha256_file(Path(inspect.getfile(module)).resolve()),
            }
            for module in modules
        },
    }


def official_profile_spec(profile: str = "legacy") -> dict[str, Any]:
    profiles = {
        "legacy": {
            "registry_names": ["week6_tool_sft_train", "week6_tool_sft_eval_frozen"],
            "file_names": ["tool_sft_train_100.json", "tool_sft_eval_frozen.json"],
            "row_counts": [100, 30],
            "dataset_info_name": "dataset_info.json",
        },
        "v2": {
            "registry_names": ["week6_tool_sft_train_v2", "week6_tool_sft_dev_v2"],
            "file_names": ["tool_sft_train_v2.json", "tool_sft_dev_v2.json"],
            "row_counts": [300, 40],
            "dataset_info_name": "dataset_info_v2.json",
        },
    }
    if profile not in profiles:
        raise ValueError(f"unsupported official parser profile: {profile}")
    return profiles[profile]


def _official_aligned_projection_hashes(profile: str = "legacy") -> dict[str, str]:
    """Re-run the official alignment path before accepting its projection hashes."""
    runner = importlib.import_module("run_official_parser_smoke")
    parser_module = importlib.import_module("llamafactory.data.parser")
    converter_module = importlib.import_module("llamafactory.data.converter")
    datasets_module = importlib.import_module("datasets")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        spec = official_profile_spec(profile)
        for name in spec["file_names"]:
            shutil.copy2(DAY31_SOURCE / "data" / name, temp / name)
        shutil.copy2(
            DAY31_SOURCE.parent / "configs" / spec["dataset_info_name"],
            temp / "dataset_info.json",
        )
        projections = runner.align_registered_datasets(
            temp,
            parser_module.get_dataset_list,
            converter_module.align_dataset,
            datasets_module.load_dataset,
            dataset_names=spec["registry_names"],
        )
    if [count for count, _ in projections] != spec["row_counts"]:
        raise ValueError("official aligned row counts mismatch")
    return {"train": projections[0][1], "eval_frozen": projections[1][1]}


def verify_official_parser_receipt(
    receipt: dict[str, Any] | None,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    runner_path: Path,
    *,
    provenance_provider: Callable[[], dict[str, Any]] = _official_parser_provenance,
    alignment_provider: Callable[[], dict[str, str]] = _official_aligned_projection_hashes,
    profile: str = "legacy",
) -> bool:
    """Recompute every receipt binding from current code, data, and package source."""
    try:
        if not isinstance(receipt, dict) or receipt.get("exit_code") != 0 or not isinstance(receipt.get("command"), list):
            return False
        if receipt.get("version") != LLAMAFACTORY_VERSION or receipt.get("git_revision") != LLAMAFACTORY_REVISION:
            return False
        spec = official_profile_spec(profile)
        if receipt.get("profile", "legacy") != profile:
            return False
        if (
            receipt.get("train_aligned_rows") != spec["row_counts"][0]
            or receipt.get("eval_aligned_rows") != spec["row_counts"][1]
        ):
            return False
        projections = receipt.get("aligned_projection_sha256")
        if not isinstance(projections, dict) or not all(
            isinstance(projections.get(name), str)
            and len(projections[name]) == 64
            and all(character in "0123456789abcdef" for character in projections[name])
            for name in ("train", "eval_frozen")
        ):
            return False
        signed = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if receipt.get("receipt_sha256") != hashlib.sha256(canonical_json(signed).encode()).hexdigest():
            return False
        if not runner_path.is_file() or receipt.get("runner_script_sha256") != sha256_file(runner_path):
            return False
        expected = {
            "train": hashlib.sha256(canonical_json(train_rows).encode()).hexdigest(),
            "eval_frozen": hashlib.sha256(canonical_json(eval_rows).encode()).hexdigest(),
        }
        if receipt.get("dataset_sha256") != expected:
            return False
        frozen_train = DAY31_SOURCE / "data" / spec["file_names"][0]
        frozen_eval = DAY31_SOURCE / "data" / spec["file_names"][1]
        if not (frozen_train.is_file() and frozen_eval.is_file()):
            return False
        current_train = json.loads(frozen_train.read_text(encoding="utf-8"))
        current_eval = json.loads(frozen_eval.read_text(encoding="utf-8"))
        if canonical_json(current_train) != canonical_json(train_rows) or canonical_json(current_eval) != canonical_json(eval_rows):
            return False
        file_hashes = receipt.get("dataset_file_sha256")
        dataset_info = DAY31_SOURCE.parent / "configs" / spec["dataset_info_name"]
        if file_hashes != {
            "train": sha256_file(frozen_train),
            "eval_frozen": sha256_file(frozen_eval),
            "dataset_info": sha256_file(dataset_info),
        }:
            return False
        provenance = provenance_provider()
        return (
            provenance.get("version") == LLAMAFACTORY_VERSION
            and provenance.get("git_revision") == LLAMAFACTORY_REVISION
            and receipt.get("official_module_hashes") == provenance.get("official_module_hashes")
            and projections
            == (
                alignment_provider(profile)
                if alignment_provider is _official_aligned_projection_hashes
                else alignment_provider()
            )
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError, subprocess.SubprocessError):
        return False


def _official_receipt(
    receipt: dict[str, Any] | None,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    profile: str = "legacy",
) -> bool:
    return verify_official_parser_receipt(
        receipt,
        train_rows,
        eval_rows,
        Path(__file__).with_name("run_official_parser_smoke.py"),
        profile=profile,
    )


def verify_ready_merged_preflight(preflight_path: Path, model_path: Path) -> dict[str, Any]:
    """Revalidate a ready Scheme-A preflight before an evaluator loads a model."""
    preflight = _load_json(preflight_path)
    lineage = preflight.get("lineage")
    if preflight.get("status") != "ready" or not isinstance(lineage, dict):
        raise ValueError("preflight is not ready")
    if lineage.get("scheme") != "A" or lineage.get("load_mode") != "verified_week4_merged":
        raise ValueError("preflight requires a materialized verified merged model")
    profile = preflight.get("profile", "legacy")
    spec = official_profile_spec(profile)
    train = json.loads((DAY31_SOURCE / "data" / spec["file_names"][0]).read_text(encoding="utf-8"))
    evaluation = json.loads((DAY31_SOURCE / "data" / spec["file_names"][1]).read_text(encoding="utf-8"))
    if not verify_official_parser_receipt(
        preflight.get("official_parser_smoke"),
        train,
        evaluation,
        Path(__file__).with_name("run_official_parser_smoke.py"),
        profile=profile,
    ):
        raise ValueError("official parser smoke receipt is invalid")
    manifest_path = lineage.get("merged_manifest_path")
    if not isinstance(manifest_path, str) or not Path(manifest_path).is_file():
        raise ValueError("merged manifest is missing")
    manifest = _load_json(Path(manifest_path))
    if manifest.get("manifest_sha256") != WEEK4_MERGED_MANIFEST_SHA256 or lineage.get("merged_manifest_sha256") != WEEK4_MERGED_MANIFEST_SHA256:
        raise ValueError("merged manifest identity mismatch")
    if Path(lineage.get("model_dir", "")).resolve() != model_path.resolve():
        raise ValueError("preflight model path mismatch")
    _verify_manifest_directory(manifest, model_path)
    return preflight


def build_preflight_report(
    *, lineage: dict[str, Any], train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], tokenizer: Any, cutoff: int, official_parser_receipt: dict[str, Any] | None = None, profile: str = "legacy"
) -> dict[str, Any]:
    versions = package_versions(["transformers", "peft", "torch", "llamafactory"])
    parser_ok = _parser_compatible(train_rows) and _parser_compatible(eval_rows)
    official_ok = _official_receipt(official_parser_receipt, train_rows, eval_rows, profile)
    stats = {"train": token_length_stats(train_rows, tokenizer, cutoff), "eval_frozen": token_length_stats(eval_rows, tokenizer, cutoff)}
    return {
        "schema_version": "1.0",
        "status": "ready" if parser_ok and official_ok and all(versions[name] is not None for name in ("transformers", "peft", "torch", "llamafactory")) else "blocked",
        "lineage": lineage,
        "versions": versions,
        "local_schema_ok": parser_ok,
        "official_parser_smoke": official_parser_receipt if official_ok else {"status": "blocked"},
        "token_length_stats": stats,
        "recommended_cutoff_len": max(stats["train"]["recommended_cutoff"], stats["eval_frozen"]["recommended_cutoff"]),
    }


def _expected_calls(row: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    calls = []
    for message in row["conversations"]:
        if message.get("from") == "function_call":
            value = json.loads(message["value"])
            calls.append((value["name"], value["arguments"]))
    return calls


def _expected_observation_outcomes(row: dict[str, Any]) -> list[tuple[Any, Any]]:
    outcomes = []
    for message in row["conversations"]:
        if message.get("from") == "observation":
            value = json.loads(message["value"])
            outcomes.append((value.get("status"), value.get("error_code")))
    return outcomes


def _real_observation(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    from build_tool_sft_dataset import execute_tool

    return execute_tool(name, arguments)


def _terminal_answer_is_resolved(final_answer: str, expected_call_count: int, actual_call_count: int) -> bool:
    """Reject emitted tool-call syntax; completion needs a terminal answer after all calls."""
    stripped = final_answer.strip()
    if not stripped or actual_call_count != expected_call_count:
        return False
    lowered = stripped.lower()
    if any(marker in lowered for marker in ("<tool", "</tool", "tool_call", "function_call", "action input")):
        return False
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return '"name"' not in stripped and '"arguments"' not in stripped
    return not (isinstance(parsed, dict) and ("name" in parsed or "arguments" in parsed))


def score_frozen_case(row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    expected = _expected_calls(row)
    trace = result.get("trace", []) if isinstance(result.get("trace"), list) else []
    actual = [(entry.get("action"), entry.get("action_input")) for entry in trace if isinstance(entry, dict)]
    signatures = [(name, canonical_json(arguments)) for name, arguments in actual]
    hallucinated = False
    for entry, (name, arguments) in zip(trace, actual):
        if not isinstance(name, str) or not isinstance(arguments, dict) or entry.get("observation") != _real_observation(name, arguments):
            hallucinated = True
    final_answer = str(result.get("final_answer", ""))
    recovery = row.get("category") == "recovery_or_termination"
    recovery_outcome_ok = True
    if recovery:
        expected_outcomes = _expected_observation_outcomes(row)
        actual_outcomes = [
            (entry.get("observation", {}).get("status"), entry.get("observation", {}).get("error_code"))
            for entry in trace
            if isinstance(entry, dict) and isinstance(entry.get("observation"), dict)
        ]
        recovery_outcome_ok = bool(expected_outcomes) and actual_outcomes == expected_outcomes
    checks: dict[str, bool | None] = {
        "first_tool_choice": bool(actual) and bool(expected) and actual[0][0] == expected[0][0],
        "full_tool_sequence": [name for name, _ in actual] == [name for name, _ in expected],
        "argument_correctness": None if recovery else actual == expected,
        "completion": result.get("status") == "success" and recovery_outcome_ok and _terminal_answer_is_resolved(final_answer, len(expected), len(actual)),
        "no_loops": len(signatures) == len(set(signatures)),
        "no_hallucinated_observations": not hallucinated,
    }
    return {"case_id": row["id"], "checks": checks, "passed": all(value for value in checks.values() if value is not None), "raw_trace": result}


_ARGUMENT_SCHEMAS: dict[str, dict[str, type]] = {
    "calculator": {"expression": str},
    "knowledge_retrieval": {"query": str},
    "code_executor": {"source": str},
}
_ERROR_STATUSES = {"rejected", "not_found", "ambiguous", "error"}
_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?(?![A-Za-z0-9_.])")


def _arguments_match_schema(name: object, arguments: object) -> bool:
    schema = _ARGUMENT_SCHEMAS.get(name) if isinstance(name, str) else None
    return bool(
        schema
        and isinstance(arguments, dict)
        and set(arguments) == set(schema)
        and all(isinstance(arguments[key], value_type) for key, value_type in schema.items())
    )


def _contains_number(text: str, value: int | float) -> bool:
    numbers = [float(match.group()) for match in _NUMBER_PATTERN.finditer(text)]
    return any(abs(number - float(value)) < 1e-9 for number in numbers)


def _numeric_values(value: object) -> set[float]:
    if isinstance(value, bool):
        return set()
    if isinstance(value, (int, float)):
        return {float(value)}
    if isinstance(value, dict):
        return {number for item in value.values() for number in _numeric_values(item)}
    if isinstance(value, list):
        return {number for item in value for number in _numeric_values(item)}
    return set()


def _semantic_answer_ok(final_answer: str, expected: dict[str, Any]) -> bool:
    kind = expected.get("kind")
    lowered = final_answer.casefold()
    if kind == "calculator":
        return _contains_number(final_answer, expected["value"])
    if kind == "knowledge":
        return (
            str(expected["name"]).casefold() in lowered
            and _contains_number(final_answer, expected["price_cny"])
            and _contains_number(final_answer, expected["stock"])
        )
    if kind == "budget":
        relation_words = {
            "above": ("超过", "高于", "above", "over"),
            "below": ("未超过", "没有超过", "低于", "below", "within"),
            "equal": ("等于", "相等", "equal"),
        }
        required_numbers = (
            expected["price_cny"],
            expected["shipping_cny"],
            expected["total_cny"],
            expected["budget_cny"],
            expected["difference_cny"],
        )
        return (
            str(expected["name"]).casefold() in lowered
            and all(_contains_number(final_answer, value) for value in required_numbers)
            and any(word in lowered for word in relation_words[expected["relation"]])
        )
    if kind == "code":
        risk_nodes = expected.get("risk_nodes", [])
        risks_ok = (
            all(str(node).casefold() in lowered for node in risk_nodes)
            if risk_nodes
            else any(marker in lowered for marker in ("[]", "无风险", "没有风险", "none"))
        )
        error_code = expected.get("error_code")
        error_ok = (
            str(error_code).casefold() in lowered
            if error_code is not None
            else any(marker in lowered for marker in ("none", "无错误", "null"))
        )
        return (
            str(expected["status"]).casefold() in lowered
            and str(expected["syntax_valid"]).casefold() in lowered
            and risks_ok
            and error_ok
        )
    if kind == "recovery":
        return (
            str(expected["status"]).casefold() in lowered
            and str(expected["error_code"]).casefold() in lowered
        )
    return False


def _final_answer_grounded(
    final_answer: str,
    trace: list[dict[str, Any]],
    expected_answer: dict[str, Any],
) -> bool:
    if not _semantic_answer_ok(final_answer, expected_answer):
        return False
    allowed = {
        number
        for entry in trace
        for number in _numeric_values(entry.get("observation"))
    }
    if expected_answer.get("kind") == "budget":
        budget = float(expected_answer["budget_cny"])
        allowed.add(budget)
        calculator_values = [
            entry.get("observation", {}).get("data", {}).get("value")
            for entry in trace
            if entry.get("action") == "calculator"
            and isinstance(entry.get("observation"), dict)
        ]
        if calculator_values and isinstance(calculator_values[-1], (int, float)):
            allowed.add(abs(float(calculator_values[-1]) - budget))
    final_numbers = {float(match.group()) for match in _NUMBER_PATTERN.finditer(final_answer)}
    return final_numbers.issubset(allowed)


def score_case_v2(
    row: dict[str, Any],
    result: dict[str, Any],
    knowledge_base_path: Path,
) -> dict[str, Any]:
    """Score a self-contained formal case with semantic and grounding checks."""
    expected = _expected_calls(row)
    trace = result.get("trace", []) if isinstance(result.get("trace"), list) else []
    actual = [
        (entry.get("action"), entry.get("action_input"))
        for entry in trace
        if isinstance(entry, dict)
    ]
    signatures = [
        (name, canonical_json(arguments))
        for name, arguments in actual
        if isinstance(arguments, dict)
    ]
    observation_integrity = len(trace) == len(actual) and all(
        isinstance(name, str)
        and isinstance(arguments, dict)
        and entry.get("observation")
        == _real_observation_v2(name, arguments, knowledge_base_path)
        for entry, (name, arguments) in zip(trace, actual)
    )
    final_answer = str(result.get("final_answer", ""))
    expected_answer = row.get("success_predicates", {}).get("expected_answer", {})
    recovery = row.get("category") == "recovery_or_termination"
    safe_termination = True
    if recovery:
        safe_termination = (
            len(trace) == 1
            and isinstance(trace[0].get("observation"), dict)
            and trace[0]["observation"].get("status") in _ERROR_STATUSES
        )
    format_valid = _terminal_answer_is_resolved(final_answer, len(expected), len(actual))
    semantic_correctness = _semantic_answer_ok(final_answer, expected_answer)
    checks: dict[str, bool] = {
        "first_tool_choice": bool(actual) and bool(expected) and actual[0][0] == expected[0][0],
        "full_tool_sequence": [name for name, _ in actual] == [name for name, _ in expected],
        "argument_schema_valid": all(
            _arguments_match_schema(name, arguments) for name, arguments in actual
        ),
        "argument_correctness": actual == expected,
        "observation_integrity": observation_integrity,
        "format_valid": format_valid,
        "semantic_correctness": semantic_correctness,
        "final_answer_grounded": _final_answer_grounded(
            final_answer, trace, expected_answer
        ),
        "safe_termination": safe_termination,
        "completion": result.get("status") == "success" and format_valid and safe_termination,
        "no_loops": len(signatures) == len(set(signatures)) and len(signatures) == len(actual),
    }
    return {
        "case_id": row["id"],
        "checks": checks,
        "passed": all(checks.values()),
        "raw_trace": result,
    }


def _real_observation_v2(
    name: str, arguments: dict[str, Any], knowledge_base_path: Path
) -> dict[str, Any]:
    from build_tool_sft_dataset import execute_tool

    return execute_tool(name, arguments, knowledge_base_path)


def evaluate_frozen_cases(rows: list[dict[str, Any]], runner: Callable[[dict[str, Any]], dict[str, Any]], generation_config: dict[str, Any], model_identity: dict[str, Any], protocol: dict[str, Any] | None = None, knowledge_base_path: Path | None = None) -> dict[str, Any]:
    v2 = bool(rows) and all(
        isinstance(row.get("success_predicates", {}).get("expected_answer"), dict)
        for row in rows
    )
    if v2 and knowledge_base_path is None:
        raise ValueError("v2 evaluation requires knowledge_base_path")
    case_results = [
        score_case_v2(row, runner(row), knowledge_base_path)
        if v2 and knowledge_base_path is not None
        else score_frozen_case(row, runner(row))
        for row in rows
    ]
    check_names = (
        list(case_results[0]["checks"])
        if case_results
        else []
    )
    applicable_counts = {
        name: sum(case["checks"][name] is not None for case in case_results)
        for name in check_names
    }
    return {
        "schema_version": "2.0" if v2 else "1.0",
        "status": "completed",
        "model_identity": model_identity,
        "generation_config": generation_config,
        "protocol": protocol or {},
        "case_count": len(case_results),
        "applicable_counts": applicable_counts,
        "metrics": {
            name: (
                sum(case["checks"][name] is True for case in case_results) / applicable_counts[name]
                if applicable_counts[name] else None
            )
            for name in check_names
        },
        "case_results": case_results,
    }


def collect_training_evidence(trainer_output: Path | None, effective_config: Path | None, preflight: Path | None, launch_command: Path | None = None, exit_status: Path | None = None, trainable_evidence: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return not_run placeholders until real output files provide all evidence."""
    base = {"schema_version": "1.0", "status": "not_run", "reason": "real AutoDL trainer output was not supplied"}
    if trainer_output is None:
        return base, {**base, "weights_location": "AutoDL only; do not commit adapter or model weights."}
    trainer_output = trainer_output.resolve()
    state_path = trainer_output / "trainer_state.json"
    model_path = trainer_output / "adapter_model.safetensors"
    config_path = trainer_output / "adapter_config.json"
    if not (state_path.is_file() and model_path.is_file() and config_path.is_file() and effective_config and effective_config.is_file() and preflight and preflight.is_file() and launch_command and launch_command.is_file() and exit_status and exit_status.is_file() and trainable_evidence and trainable_evidence.is_file()):
        return {**base, "reason": "required real trainer state, adapter, effective config, or preflight evidence missing"}, {**base, "reason": "required real adapter files or provenance missing"}
    state = _load_json(state_path)
    try:
        raw=model_path.read_bytes(); header_size=int.from_bytes(raw[:8],"little"); header=json.loads(raw[8:8+header_size]); tensors=[(name,value) for name,value in header.items() if name!="__metadata__"]
        adapter_value=_load_json(config_path)
        offsets_ok=all(isinstance(value,dict) and isinstance(value.get("data_offsets"),list) and len(value["data_offsets"])==2 and 8+header_size<=8+header_size+value["data_offsets"][0]<=8+header_size+value["data_offsets"][1]<=len(raw) for _,value in tensors)
        adapter_ok=adapter_value.get("peft_type")=="LORA" and adapter_value.get("task_type")=="CAUSAL_LM" and adapter_value.get("r",0)>0 and adapter_value.get("lora_alpha",0)>0 and bool(adapter_value.get("target_modules")) and any("lora_" in name.lower() for name,_ in tensors)
    except Exception: offsets_ok=adapter_ok=False
    exit_value = _load_json(exit_status)
    import yaml
    config_value, launch_value, trainable_value = yaml.safe_load(effective_config.read_text(encoding="utf-8")), _load_json(launch_command), _load_json(trainable_evidence)
    dataset_pair = (
        config_value.get("dataset"),
        config_value.get("eval_dataset"),
    ) if isinstance(config_value, dict) else (None, None)
    config_ok = isinstance(config_value, dict) and config_value.get("stage") == "sft" and config_value.get("do_train") is True and dataset_pair in {
        ("week6_tool_sft_train", "week6_tool_sft_eval_frozen"),
        ("week6_tool_sft_train_v2", "week6_tool_sft_dev_v2"),
    } and config_value.get("output_dir") == str(trainer_output) and config_value.get("model_name_or_path") and config_value.get("quantization_bit") == 4 and config_value.get("quantization_type") == "nf4" and config_value.get("double_quantization") is True and config_value.get("gradient_checkpointing") is True
    launch_ok = launch_value.get("argv") == ["llamafactory-cli", "train", str(effective_config.resolve())] and launch_value.get("cwd") and launch_value.get("config_sha256") == sha256_file(effective_config)
    try:
        if not config_ok:
            raise ValueError("effective config is not a required QLoRA SFT contract")
        preflight_value = verify_ready_merged_preflight(preflight, Path(config_value["model_name_or_path"]))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {**base, "reason": "ready merged preflight verification failed"}, {**base, "reason": "ready merged preflight verification failed"}
    if not isinstance(state.get("log_history"), list) or exit_value.get("exit_code") != 0 or exit_value.get("launch_sha256") != sha256_file(launch_command) or exit_value.get("config_sha256") != sha256_file(effective_config) or not launch_ok or not offsets_ok or not adapter_ok or adapter_value.get("base_model_name_or_path") != config_value.get("model_name_or_path") or not isinstance(trainable_value.get("names"), list) or not trainable_value["names"]:
        return {**base, "reason": "trainer_state.json lacks log_history"}, {**base, "reason": "trainer_state.json lacks log_history"}
    files = [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(trainer_output.iterdir()) if path.is_file()]
    profile = "v2" if dataset_pair[0] == "week6_tool_sft_train_v2" else "legacy"
    adapter = {
        "schema_version": "2.0" if profile == "v2" else "1.0", "profile": profile, "status": "collected", "trainer_output": str(trainer_output.resolve()),
        "effective_config_sha256": sha256_file(effective_config), "preflight_sha256": sha256_file(preflight), "launch_command_sha256": sha256_file(launch_command), "exit_status_sha256": sha256_file(exit_status), "lineage": preflight_value["lineage"],
        "files": files, "total_bytes": sum(file["bytes"] for file in files),
        "adapter": {"path": model_path.name, "bytes": model_path.stat().st_size, "sha256": sha256_file(model_path), "config_sha256": sha256_file(config_path)},
    }
    summary = {"schema_version": "2.0" if profile == "v2" else "1.0", "profile": profile, "status": "collected", "trainer_state": state, "per_step_loss": [row.get("loss") for row in state["log_history"] if isinstance(row, dict) and "loss" in row], "trainable_parameter_names": trainable_value["names"], "trainable_evidence_sha256": sha256_file(trainable_evidence), "launch_command": launch_command.read_text(encoding="utf-8"), "exit_code": 0, "effective_config_sha256": sha256_file(effective_config), "preflight_sha256": sha256_file(preflight), "adapter_manifest_sha256": hashlib.sha256(canonical_json(adapter).encode()).hexdigest()}
    return summary, adapter
