#!/usr/bin/env bash
set -euo pipefail

WEEK5_ROOT=${WEEK5_ROOT:-/root/autodl-tmp/qwen2-vl-week5}
STAGING_DIR=${STAGING_DIR:-${WEEK5_ROOT}/staging}
MODEL_DIR=${MODEL_DIR:-${WEEK5_ROOT}/models/Qwen2-VL-7B-Instruct}
EVIDENCE_DIR=${EVIDENCE_DIR:-${WEEK5_ROOT}/evidence/day22}
LOG_DIR=${LOG_DIR:-${WEEK5_ROOT}/logs}
VENV_DIR=${VENV_DIR:-${WEEK5_ROOT}/venv}
BASE_PYTHON=${BASE_PYTHON:-/root/autodl-tmp/conda/envs/llm_exp/bin/python}
SMOKE_IMAGE=${SMOKE_IMAGE:-${STAGING_DIR}/images/w5-scene-01.jpg}
REPOSITORY=Qwen/Qwen2-VL-7B-Instruct
REVISION=eed13092ef92e448dd6875b2a00151bd3f7db0ac
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

mkdir -p "${MODEL_DIR}" "${EVIDENCE_DIR}" "${LOG_DIR}" "${WEEK5_ROOT}/hf-cache"
RUN_LOG=${LOG_DIR}/day22_run.log
exec > >(tee -a "${RUN_LOG}") 2>&1

date -u +%Y-%m-%dT%H:%M:%SZ
nvidia-smi
df -h /root/autodl-tmp

if [[ ! -x "${BASE_PYTHON}" ]]; then
  BASE_PYTHON=$(command -v python3)
fi
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
fi
PYTHON=${VENV_DIR}/bin/python

"${PYTHON}" -m pip install --disable-pip-version-check --no-input \
  'qwen-vl-utils==0.0.14' \
  'transformers>=4.45,<5' \
  'accelerate>=0.26,<2' \
  'huggingface-hub>=0.25,<1'

export HF_HOME=${WEEK5_ROOT}/hf-cache
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# A fresh 7B download needs roughly 16.6 GB plus temporary headroom. On an
# idempotent rerun, the verified five shards already occupy that space, so the
# post-download free-space gate can be lower without weakening file checks.
PREFLIGHT_MIN_FREE_GIB=22
if [[ -f "${MODEL_DIR}/model.safetensors.index.json" ]]; then
  SHARD_COUNT=$(find "${MODEL_DIR}" -maxdepth 1 -type f -name 'model-*-of-*.safetensors' | wc -l | tr -d ' ')
  if [[ "${SHARD_COUNT}" -eq 5 ]]; then
    PREFLIGHT_MIN_FREE_GIB=10
  fi
fi

"${PYTHON}" "${SCRIPT_DIR}/prepare_day22_remote.py" preflight \
  --storage-path /root/autodl-tmp \
  --min-vram-mib 16000 \
  --min-free-gib "${PREFLIGHT_MIN_FREE_GIB}" \
  --output "${EVIDENCE_DIR}/environment.json"

"${PYTHON}" "${SCRIPT_DIR}/prepare_day22_remote.py" download \
  --repository "${REPOSITORY}" \
  --revision "${REVISION}" \
  --model-dir "${MODEL_DIR}" \
  --output "${EVIDENCE_DIR}/model_manifest.json"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
"${PYTHON}" "${SCRIPT_DIR}/run_vlm_smoke.py" \
  --model-dir "${MODEL_DIR}" \
  --image "${SMOKE_IMAGE}" \
  --repository "${REPOSITORY}" \
  --revision "${REVISION}" \
  --output "${EVIDENCE_DIR}/offline_load_smoke.json" \
  --min-pixels 200704 \
  --max-pixels 301056 \
  --max-new-tokens 32 \
  --seed 42

"${PYTHON}" - "${EVIDENCE_DIR}" "${MODEL_DIR}" "${REVISION}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

evidence_dir = Path(sys.argv[1])
model_dir = Path(sys.argv[2])
revision = sys.argv[3]
payload = {
    "status": "PASS",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "repository": "Qwen/Qwen2-VL-7B-Instruct",
    "revision": revision,
    "remote_model_path": str(model_dir),
    "evidence": ["environment.json", "model_manifest.json", "offline_load_smoke.json"],
}
(evidence_dir / "remote_status.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

nvidia-smi
df -h /root/autodl-tmp
date -u +%Y-%m-%dT%H:%M:%SZ
