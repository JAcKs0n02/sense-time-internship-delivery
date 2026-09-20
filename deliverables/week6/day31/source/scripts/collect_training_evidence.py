#!/usr/bin/env python3
"""Collect only real LLaMA-Factory output evidence; never invent training facts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from day31_harness import collect_training_evidence

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-output", type=Path)
    parser.add_argument("--effective-config", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--launch-command", type=Path)
    parser.add_argument("--exit-status", type=Path)
    parser.add_argument("--trainable-evidence", type=Path)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--adapter-output", type=Path, required=True)
    args = parser.parse_args()
    summary, adapter = collect_training_evidence(args.trainer_output, args.effective_config, args.preflight, args.launch_command, args.exit_status, args.trainable_evidence)
    for path, value in ((args.summary_output, summary), (args.adapter_output, adapter)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if summary.get("status") == "collected" and adapter.get("status") == "collected" else 1
if __name__ == "__main__": raise SystemExit(main())
