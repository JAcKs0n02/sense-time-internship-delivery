#!/usr/bin/env python3
"""Teacher-facing Day29 initialization and offline-load preflight."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DAY29_SOURCE = Path(__file__).resolve().parents[1]
WEEK6_SOURCE = DAY29_SOURCE.parents[1] / "source"
REPO_ROOT = DAY29_SOURCE.parents[3]
CONFIG_PATH = DAY29_SOURCE / "configs" / "agent_config.json"
KNOWLEDGE_BASE = DAY29_SOURCE.parents[1] / "day28/source/data/knowledge_base.json"
DAY28_VALIDATOR = DAY29_SOURCE.parents[1] / "day28/source/scripts/validate_day28.py"
sys.path.insert(0, str(WEEK6_SOURCE))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-model-path", type=Path, required=True)
    parser.add_argument("--remote-adapter-path", type=Path, required=True)
    parser.add_argument("--expected-source-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args(argv)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from week6_agent.agent_factory import (
            build_formal_evidence_bundle,
            build_react_agent,
        )
        from week6_agent.model_adapter import HuggingFaceLocalChatModel

        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        day28 = subprocess.run(
            [sys.executable, str(DAY28_VALIDATOR)], cwd=REPO_ROOT, check=False
        )
        if day28.returncode != 0:
            raise ValueError("Day28 validator did not pass")
        artifacts, _, run_spec_document = build_formal_evidence_bundle(
            config=config,
            config_path=CONFIG_PATH,
            model_path=args.remote_model_path,
            adapter_path=args.remote_adapter_path,
            source_manifest_path=args.expected_source_manifest,
            repo_root=REPO_ROOT,
            knowledge_base_path=KNOWLEDGE_BASE,
        )
        model = HuggingFaceLocalChatModel.from_local_model(
            str(args.remote_model_path),
            adapter_path=str(args.remote_adapter_path),
            **config["generation"],
        )
        agent = build_react_agent(
            model,
            KNOWLEDGE_BASE,
            max_steps=config["max_steps"],
            runtime_limit_seconds=config["runtime_limit_seconds"],
        )
        if agent.recursion_limit != artifacts["effective_config"]["recursion_limit"]:
            raise ValueError("compiled agent recursion limit mismatch")
        _write(args.output_directory / "model_preflight.json", artifacts["model_preflight"])
        _write(args.output_directory / "environment.json", artifacts["environment"])
        _write(args.output_directory / "effective_config.json", artifacts["effective_config"])
        _write(
            args.output_directory / "code_tool_prompt_manifest.json",
            artifacts["code_tool_prompt_manifest"],
        )
        _write(args.output_directory / "run_spec.json", run_spec_document)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
