"""Single-variable Prompt ablation for the formal Week6 Agent evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from langgraph.prebuilt import create_react_agent

from week6_agent.day30_agent import (
    DAY30_SYSTEM_PROMPT,
    DAY30_SYSTEM_PROMPT_VERSION,
    Day30CompiledAgent,
    build_day30_tools,
)


DAY32_SYSTEM_PROMPT_VERSION = "week6-day32-three-tool-react-v2"
INPUT_FIDELITY_RULE = (
    "- Copy every user-provided arithmetic expression or Python source exactly into "
    "the tool argument, preserving operators, equals signs, quotes, spaces, and line breaks."
)
ABLATION_SUFFIX = f"\n\nInput-fidelity ablation:\n{INPUT_FIDELITY_RULE}"
DAY32_SYSTEM_PROMPT = DAY30_SYSTEM_PROMPT + ABLATION_SUFFIX


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prompt_manifest(baseline_path: Path, optimized_path: Path) -> dict[str, Any]:
    """Prove the ablation adds one rule and changes no baseline bytes."""
    baseline = baseline_path.read_text(encoding="utf-8").rstrip("\n")
    optimized = optimized_path.read_text(encoding="utf-8").rstrip("\n")
    if baseline != DAY30_SYSTEM_PROMPT:
        raise ValueError("baseline prompt does not match Day30 runtime prompt")
    if not optimized.startswith(baseline):
        raise ValueError("optimized prompt changes the baseline prompt")
    added = optimized[len(baseline) :]
    added_rules = [line for line in added.splitlines() if line.startswith("- ")]
    if len(added_rules) != 1:
        raise ValueError("prompt ablation must add exactly one rule")
    if optimized != DAY32_SYSTEM_PROMPT or added_rules != [INPUT_FIDELITY_RULE]:
        raise ValueError("optimized prompt does not match the declared single rule")
    return {
        "schema_version": "2.0",
        "change_scope": "single_rule_prompt_ablation",
        "selection_split": "dev_v2",
        "final_test_used_for_selection": False,
        "baseline": {
            "version": DAY30_SYSTEM_PROMPT_VERSION,
            "path": baseline_path.name,
            "sha256": _sha256_text(baseline),
        },
        "optimized": {
            "version": DAY32_SYSTEM_PROMPT_VERSION,
            "path": optimized_path.name,
            "sha256": _sha256_text(optimized),
        },
        "unchanged_prefix_sha256": _sha256_text(baseline),
        "added_rules": added_rules,
        "targeted_metric": "argument_correctness",
    }


def validate_prompt_selection_split(
    evaluation_path: Path, dataset_manifest_path: Path
) -> dict[str, Any]:
    """Allow Prompt selection only on the manifest-bound development split."""
    manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    selection = manifest.get("selection_policy", {})
    dev = manifest.get("splits", {}).get("dev_v2", {})
    if (
        manifest.get("schema_version") != "2.0"
        or selection.get("dev_only_for_checkpoint_and_prompt_selection") is not True
        or selection.get("test_v2_reserved_for_final_evaluation") is not True
    ):
        raise ValueError("dataset selection policy is invalid")
    if (
        evaluation_path.name != dev.get("file_name")
        or _sha256_file(evaluation_path) != dev.get("file_sha256")
    ):
        raise ValueError("Prompt selection requires the development split")
    return {
        "split": "dev_v2",
        "path": str(evaluation_path.resolve()),
        "row_count": dev.get("row_count"),
        "sha256": dev.get("file_sha256"),
    }


def build_day32_react_agent(
    model: Any,
    knowledge_base_path: Path,
    max_steps: int = 4,
    runtime_limit_seconds: float = 60,
    policy_mode: str = "raw",
) -> Day30CompiledAgent:
    """Build the same three-tool Agent with only one Prompt rule added."""
    tools = build_day30_tools(Path(knowledge_base_path))
    graph = create_react_agent(
        model,
        tools,
        prompt=DAY32_SYSTEM_PROMPT,
        version="v2",
        name="week6_day32_prompt_v2_agent",
    )
    return Day30CompiledAgent(
        graph, tools, max_steps, runtime_limit_seconds, policy_mode
    )
