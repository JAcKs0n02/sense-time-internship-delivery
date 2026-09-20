#!/usr/bin/env python3
"""Build comparison CSV/JSON only from three valid raw benchmark receipts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from week7_deployment.benchmark import aggregate_model_results


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--awq", type=Path, required=True)
    parser.add_argument("--gptq", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--acceptance-output", type=Path, required=True)
    args = parser.parse_args()

    summary = aggregate_model_results(
        [_load(args.bf16), _load(args.awq), _load(args.gptq)]
    )
    for path in (args.json_output, args.csv_output, args.acceptance_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = [
        "model",
        "weight_bytes",
        "static_load_mib",
        "peak_mib",
        "memory_reduction",
        "tokens_per_second",
        "ttft_median_seconds",
        "nll",
        "perplexity",
        "ppl_delta_vs_bf16",
        "throughput_ratio_vs_bf16",
        "acceptance_passed",
    ]
    with args.csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in summary["models"]:
            writer.writerow({field: row.get(field) for field in fields})
    acceptance = {
        "schema_version": "1.0",
        "status": (
            "PASS" if summary["at_least_one_quantization_passed"] else "FAIL"
        ),
        "requirement": "at least one quantized model reduces peak memory by 30%",
        "protocol_sha256": summary["protocol_sha256"],
        "models": [
            {
                "model": row["model"],
                "memory_reduction": row["memory_reduction"],
                "acceptance_passed": row["acceptance_passed"],
            }
            for row in summary["models"]
            if row["model"] != "bf16"
        ],
    }
    args.acceptance_output.write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": acceptance["status"]}, sort_keys=True))
    return 0 if acceptance["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
