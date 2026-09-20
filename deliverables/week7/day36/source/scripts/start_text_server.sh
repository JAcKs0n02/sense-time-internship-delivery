#!/usr/bin/env bash
# Foreground server: Ctrl-C stops it. No background PID is guessed or killed.
set -euo pipefail
: "${WEEK7_SERVING_PYTHON:?Set the verified serving venv Python}"
: "${WEEK7_TEXT_MODEL_PATH:?Set the verified AWQ model directory}"
[[ -x "$WEEK7_SERVING_PYTHON" && -d "$WEEK7_TEXT_MODEL_PATH" ]] || exit 2
week7_gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$week7_gpu_processes" ]] || { echo 'GPU occupied' >&2; exit 3; }
unset PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
exec "$WEEK7_SERVING_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$WEEK7_TEXT_MODEL_PATH" --host 127.0.0.1 --port 8000 \
  --served-model-name week4-dpo-quantized --quantization awq --dtype half \
  --max-model-len 2048 --gpu-memory-utilization 0.75 --max-num-seqs 4 \
  --enforce-eager --disable-log-requests
