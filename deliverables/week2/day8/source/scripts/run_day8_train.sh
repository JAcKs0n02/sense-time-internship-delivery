#!/usr/bin/env bash
set -uo pipefail

DAY8_ROOT="/root/autodl-tmp/qwen25-week2/day8"
LLAMA_FACTORY_ROOT="/root/autodl-tmp/qwen25-week1/week1/third_party/LlamaFactory"
CLI="/root/autodl-tmp/conda/envs/llm_exp/bin/llamafactory-cli"
CONFIG="${DAY8_ROOT}/configs/qwen25_7b_week2_lora_annotated.yaml"
OUTPUT_DIR="${DAY8_ROOT}/saves/qwen25-7b-week2-qlora-sft-run1"
LOG="${DAY8_ROOT}/logs/day8_sft_train.txt"
STATUS_FILE="${DAY8_ROOT}/results/day8_train_exit_code.txt"
SUMMARY_FILE="${DAY8_ROOT}/results/day8_train_runtime.txt"

mkdir -p "${DAY8_ROOT}/logs" "${DAY8_ROOT}/results" "${DAY8_ROOT}/saves"

if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  printf 'Refusing to overwrite non-empty formal output: %s\n' "${OUTPUT_DIR}" | tee "${LOG}"
  printf '2\n' > "${STATUS_FILE}"
  exit 2
fi

{
  printf 'training_started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'config=%s\n' "${CONFIG}"
  sha256sum "${CONFIG}"
  printf 'dataset_sha256='
  sha256sum "${DAY8_ROOT}/data/week2_clean_alpaca.jsonl" | awk '{print $1}'
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu \
    --format=csv,noheader
} > "${SUMMARY_FILE}"

cd "${LLAMA_FACTORY_ROOT}" || exit 3
CUDA_VISIBLE_DEVICES=0 "${CLI}" train "${CONFIG}" 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}

{
  printf 'training_finished_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'training_exit_code=%s\n' "${status}"
} >> "${SUMMARY_FILE}"
printf '%s\n' "${status}" > "${STATUS_FILE}"
exit "${status}"
