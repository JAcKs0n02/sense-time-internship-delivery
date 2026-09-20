#!/usr/bin/env bash
set -uo pipefail

WEEK3_ROOT="/root/autodl-tmp/qwen25-week3"
BASE_MODEL="/root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct"
LLAMA_FACTORY_ROOT="/root/autodl-tmp/qwen25-week1/week1/third_party/LlamaFactory"
PYTHON="/root/autodl-tmp/conda/envs/llm_exp/bin/python"
CLI="/root/autodl-tmp/conda/envs/llm_exp/bin/llamafactory-cli"
CONFIG="${WEEK3_ROOT}/configs/qwen25_7b_week3_smoke.yaml"
DATA="${WEEK3_ROOT}/data/week2_clean_alpaca.jsonl"
DATASET_INFO="${WEEK3_ROOT}/data/dataset_info.json"
OUTPUT_DIR="${WEEK3_ROOT}/smoke/day11-qlora-smoke"
LOG="${WEEK3_ROOT}/logs/day11_smoke.log"
STATUS_FILE="${WEEK3_ROOT}/logs/day11_smoke_status.json"
PREFLIGHT="${WEEK3_ROOT}/logs/day11_smoke_preflight.txt"

mkdir -p "${WEEK3_ROOT}/logs" "${WEEK3_ROOT}/smoke"

if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  printf 'Refusing to overwrite non-empty smoke output: %s\n' "${OUTPUT_DIR}" | tee "${LOG}"
  printf '{"status":"refused_existing_output","exit_code":2}\n' > "${STATUS_FILE}"
  exit 2
fi

for required in \
  "${BASE_MODEL}/config.json" \
  "${BASE_MODEL}/model.safetensors.index.json" \
  "${CONFIG}" \
  "${DATA}" \
  "${DATASET_INFO}" \
  "${CLI}"; do
  if [[ ! -f "${required}" ]]; then
    printf 'Missing required file: %s\n' "${required}" | tee "${LOG}"
    printf '{"status":"missing_required_file","exit_code":3,"path":"%s"}\n' "${required}" > "${STATUS_FILE}"
    exit 3
  fi
done

started_at="$(${PYTHON} -c 'from datetime import datetime; print(datetime.now().astimezone().isoformat())')"
{
  printf 'smoke_started_at=%s\n' "${started_at}"
  printf 'config=%s\n' "${CONFIG}"
  printf 'data=%s\n' "${DATA}"
  printf 'base_model=%s\n' "${BASE_MODEL}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import torch, transformers, bitsandbytes; print(f"torch={torch.__version__}"); print(f"cuda={torch.version.cuda}"); print(f"cuda_available={torch.cuda.is_available()}"); print(f"transformers={transformers.__version__}"); print(f"bitsandbytes={bitsandbytes.__version__}")'
  "${CLI}" version
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
  df -h /root/autodl-tmp
  wc -l "${DATA}"
  sha256sum "${CONFIG}" "${DATA}" "${DATASET_INFO}" "${BASE_MODEL}/config.json" "${BASE_MODEL}/model.safetensors.index.json"
} > "${PREFLIGHT}" 2>&1

cd "${LLAMA_FACTORY_ROOT}" || exit 4
CUDA_VISIBLE_DEVICES=0 "${CLI}" train "${CONFIG}" 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}
finished_at="$(${PYTHON} -c 'from datetime import datetime; print(datetime.now().astimezone().isoformat())')"

SMOKE_STATUS="${status}" SMOKE_STARTED="${started_at}" SMOKE_FINISHED="${finished_at}" \
  SMOKE_OUTPUT="${OUTPUT_DIR}" "${PYTHON}" -c '
import json
import os
from pathlib import Path

output = Path(os.environ["SMOKE_OUTPUT"])
files = sorted(path.name for path in output.iterdir()) if output.exists() else []
payload = {
    "status": "completed" if os.environ["SMOKE_STATUS"] == "0" else "failed",
    "exit_code": int(os.environ["SMOKE_STATUS"]),
    "started_at": os.environ["SMOKE_STARTED"],
    "finished_at": os.environ["SMOKE_FINISHED"],
    "output_dir": str(output),
    "output_files": files,
}
print(json.dumps(payload, ensure_ascii=False, indent=2))
' > "${STATUS_FILE}"

exit "${status}"
