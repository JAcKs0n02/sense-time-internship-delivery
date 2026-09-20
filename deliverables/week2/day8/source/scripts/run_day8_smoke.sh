#!/usr/bin/env bash
set -uo pipefail

DAY8_ROOT="/root/autodl-tmp/qwen25-week2/day8"
LLAMA_FACTORY_ROOT="/root/autodl-tmp/qwen25-week1/week1/third_party/LlamaFactory"
CLI="/root/autodl-tmp/conda/envs/llm_exp/bin/llamafactory-cli"
CONFIG="${DAY8_ROOT}/source/configs/qwen25_7b_week2_lora_smoke.yaml"
OUTPUT_DIR="${DAY8_ROOT}/smoke/qwen25-7b-week2-qlora-smoke"
LOG="${DAY8_ROOT}/logs/day8_smoke_test.txt"
STATUS_FILE="${DAY8_ROOT}/results/day8_smoke_exit_code.txt"
SUMMARY_FILE="${DAY8_ROOT}/results/day8_smoke_runtime.txt"

mkdir -p "${DAY8_ROOT}/logs" "${DAY8_ROOT}/results" "${DAY8_ROOT}/smoke"

if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  printf 'Refusing to overwrite non-empty smoke output: %s\n' "${OUTPUT_DIR}" | tee "${LOG}"
  printf '2\n' > "${STATUS_FILE}"
  exit 2
fi

{
  printf 'smoke_started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'config=%s\n' "${CONFIG}"
  sha256sum "${CONFIG}"
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu \
    --format=csv,noheader
} > "${SUMMARY_FILE}"

cd "${LLAMA_FACTORY_ROOT}" || exit 3
CUDA_VISIBLE_DEVICES=0 "${CLI}" train "${CONFIG}" 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}

{
  printf 'smoke_finished_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'smoke_exit_code=%s\n' "${status}"
} >> "${SUMMARY_FILE}"
printf '%s\n' "${status}" > "${STATUS_FILE}"
exit "${status}"
