#!/usr/bin/env bash
# Foreground, one GPU backend only. Caller must verify model/environment hashes.
set -euo pipefail
: "${WEEK7_VISION_PYTHON:?Set the verified vision venv Python}"
: "${WEEK7_VISION_MODEL_PATH:?Set the verified Week5 base model directory}"
[[ -x "$WEEK7_VISION_PYTHON" && -d "$WEEK7_VISION_MODEL_PATH" ]] || exit 2
week7_gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$week7_gpu_processes" ]] || { echo 'GPU occupied' >&2; exit 3; }
unset PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
exec "$WEEK7_VISION_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$WEEK7_VISION_MODEL_PATH" --host 127.0.0.1 --port 8001 \
  --served-model-name week5-qwen2-vl-base --dtype half \
  --max-model-len 2048 --max-num-seqs 1 --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt image=1,video=0 \
  --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":401408}' \
  --enforce-eager --disable-log-requests
