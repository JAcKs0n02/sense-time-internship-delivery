#!/usr/bin/env bash
set -euo pipefail
week8_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PIPELINE_PYTHON:-python3}" "$week8_root/scripts/pipeline/step2_train.py" "$@"
