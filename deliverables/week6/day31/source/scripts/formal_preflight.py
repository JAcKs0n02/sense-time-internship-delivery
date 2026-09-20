#!/usr/bin/env python3
"""Verify Day31 Scheme A lineage, packages, parser shape, and token lengths."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from day31_harness import build_preflight_report, resolve_scheme_a_lineage

DAY31 = Path(__file__).resolve().parents[1]
WEEK6 = DAY31.parents[1]


def load_runtime_contract(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "2.0":
        raise ValueError("runtime contract schema mismatch")
    if value.get("consumers") != ["trainer", "evaluator"]:
        raise ValueError("runtime contract must bind trainer and evaluator")
    packages = value.get("packages")
    if not isinstance(packages, dict) or not packages:
        raise ValueError("runtime package contract missing")
    if any(not isinstance(name, str) or not isinstance(version, str) for name, version in packages.items()):
        raise ValueError("runtime package contract invalid")
    return value


def verify_runtime_versions(
    contract: dict[str, object],
    *,
    resolver=importlib.metadata.version,
) -> dict[str, object]:
    packages = contract.get("packages")
    if not isinstance(packages, dict):
        raise ValueError("runtime package contract missing")
    observed: dict[str, str | None] = {}
    mismatches: dict[str, dict[str, str | None]] = {}
    for distribution, expected in packages.items():
        try:
            actual = resolver(distribution)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        observed[distribution] = actual
        if actual != expected:
            mismatches[distribution] = {"expected": expected, "observed": actual}
    return {
        "status": "ready" if not mismatches else "blocked",
        "observed": observed,
        "mismatches": mismatches,
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("legacy", "v2"), default="legacy")
    parser.add_argument("--merged-manifest", type=Path)
    parser.add_argument("--merged-model-dir", type=Path)
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--base-model-dir", type=Path)
    parser.add_argument("--dpo-adapter-dir", type=Path)
    parser.add_argument("--day30-config", type=Path, default=WEEK6 / "day30/source/configs/agent_config.json")
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--cutoff", type=int, default=2048)
    parser.add_argument("--official-parser-receipt", type=Path, required=True)
    parser.add_argument(
        "--runtime-contract",
        type=Path,
        default=DAY31.parent / "configs/runtime_versions_v2.json",
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DAY31 / "data/dataset_manifest_v2.json",
    )
    args = parser.parse_args()
    try:
        lineage = resolve_scheme_a_lineage(merged_manifest_path=args.merged_manifest, merged_model_dir=args.merged_model_dir, base_manifest_path=args.base_manifest, base_model_dir=args.base_model_dir, dpo_adapter_dir=args.dpo_adapter_dir, day30_config_path=args.day30_config)
        if args.tokenizer_path is None:
            raise ValueError("tokenizer path required after lineage verification")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
        if args.profile == "v2":
            contract = load_runtime_contract(args.runtime_contract)
            runtime = verify_runtime_versions(contract)
            if runtime["status"] != "ready":
                raise ValueError(f"runtime version mismatch: {runtime['mismatches']}")
            manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
            validation = json.loads(
                (DAY31 / "results/data_validation_v2.json").read_text(encoding="utf-8")
            )
            if (
                manifest.get("schema_version") != "2.0"
                or manifest.get("profile") != "week6_agent_formal_v2"
                or validation.get("valid") is not True
            ):
                raise ValueError("dataset manifest is not valid")
            train = json.loads((DAY31 / "data/tool_sft_train_v2.json").read_text(encoding="utf-8"))
            evaluation = json.loads((DAY31 / "data/tool_sft_dev_v2.json").read_text(encoding="utf-8"))
        else:
            contract = None
            runtime = None
            manifest = None
            train = json.loads((DAY31 / "data/tool_sft_train_100.json").read_text(encoding="utf-8"))
            evaluation = json.loads((DAY31 / "data/tool_sft_eval_frozen.json").read_text(encoding="utf-8"))
        receipt = json.loads(args.official_parser_receipt.read_text(encoding="utf-8"))
        report = build_preflight_report(
            lineage=lineage,
            train_rows=train,
            eval_rows=evaluation,
            tokenizer=tokenizer,
            cutoff=args.cutoff,
            official_parser_receipt=receipt,
            profile=args.profile,
        )
        if args.profile == "v2":
            report.update(
                {
                    "schema_version": "2.0",
                    "profile": "v2",
                    "runtime_contract": contract,
                    "runtime_verification": runtime,
                    "dataset_manifest": manifest,
                }
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if report["status"] != "ready":
            print("preflight blocked", file=sys.stderr)
            return 1
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"lineage/preflight error: {error}", file=sys.stderr)
        return 2
if __name__ == "__main__": raise SystemExit(main())
