#!/usr/bin/env python3
"""Freeze Day30 model, source, tool, prompt, cases, and environment evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DAY30_SOURCE = Path(__file__).resolve().parents[1]
WEEK6_SOURCE = DAY30_SOURCE.parents[1] / "source"
REPO_ROOT = DAY30_SOURCE.parents[3]
CONFIG_PATH = DAY30_SOURCE / "configs/agent_config.json"
CASES_PATH = DAY30_SOURCE / "data/multistep_cases.json"
KNOWLEDGE_BASE = DAY30_SOURCE.parents[1] / "day28/source/data/knowledge_base.json"
sys.path.insert(0, str(WEEK6_SOURCE))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-model-path", type=Path, required=True)
    parser.add_argument("--remote-adapter-path", type=Path, required=True)
    parser.add_argument("--expected-source-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--run-spec", type=Path, required=True)
    return parser.parse_args(argv)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from week6_agent.day30_agent import build_day30_evidence_bundle

        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        artifacts, _, run_spec = build_day30_evidence_bundle(
            config=config,
            config_path=CONFIG_PATH,
            model_path=args.remote_model_path,
            adapter_path=args.remote_adapter_path,
            source_manifest_path=args.expected_source_manifest,
            repo_root=REPO_ROOT,
            knowledge_base_path=KNOWLEDGE_BASE,
            cases_path=CASES_PATH,
        )
        args.output_directory.mkdir(parents=True, exist_ok=True)
        for name, value in artifacts.items():
            _write_json(args.output_directory / f"{name}.json", value)
        _write_json(args.run_spec, run_spec)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
