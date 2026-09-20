#!/usr/bin/env python3
"""Freeze Week 7 calibration and perplexity datasets from Week 2 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from week7_deployment.datasets import freeze_quantization_datasets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--calibration-output", type=Path, required=True)
    parser.add_argument("--perplexity-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-count", type=int, default=128)
    parser.add_argument("--perplexity-count", type=int, default=256)
    args = parser.parse_args()

    frozen = freeze_quantization_datasets(
        source=args.source,
        calibration_output=args.calibration_output,
        perplexity_output=args.perplexity_output,
        manifest_output=args.manifest_output,
        seed=args.seed,
        calibration_count=args.calibration_count,
        perplexity_count=args.perplexity_count,
    )
    print(
        json.dumps(
            {
                "calibration_rows": len(frozen.calibration),
                "perplexity_rows": len(frozen.perplexity),
                "calibration_sha256": frozen.calibration_sha256,
                "perplexity_sha256": frozen.perplexity_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
