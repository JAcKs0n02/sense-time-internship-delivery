#!/usr/bin/env bash
set -euo pipefail

DAY20_ROOT=/root/autodl-tmp/qwen25-week4/day20
CODE_ROOT="$DAY20_ROOT/code"
RESULTS_ROOT="$DAY20_ROOT/results"
SFT_MODEL=/root/autodl-tmp/qwen25-week3/best_model/qwen25-7b-week3-best-merged
DPO_ADAPTER=/root/autodl-tmp/qwen25-week4/day19/runs/formal/attempt_002/trainer_output
DPO_MODEL=/root/autodl-tmp/qwen25-week4/best_model/qwen25-7b-week4-dpo-merged
EXPECTED_ADAPTER_SHA=a1d7c5eafba28dd927e676836f30338870fbda3dc239caf6ffe950f064b7319b

mkdir -p "$RESULTS_ROOT"

write_status() {
  python - "$1" "$2" "$RESULTS_ROOT/status.json" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
stage, state, output = sys.argv[1:]
Path(output).write_text(json.dumps({
    "stage": stage,
    "state": state,
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

on_error() {
  exit_code=$?
  write_status "${CURRENT_STAGE:-unknown}" "failed_exit_${exit_code}"
  exit "$exit_code"
}
trap on_error ERR

source /root/miniconda3/etc/profile.d/conda.sh
conda activate llm_exp

CURRENT_STAGE=preflight
write_status "$CURRENT_STAGE" running
python - "$SFT_MODEL" "$DPO_ADAPTER" "$RESULTS_ROOT/preflight.json" "$EXPECTED_ADAPTER_SHA" "$CODE_ROOT/source/data/sft_model_inventory.json" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

model = Path(sys.argv[1])
adapter = Path(sys.argv[2])
output = Path(sys.argv[3])
expected_adapter_sha = sys.argv[4]
expected_inventory = json.loads(Path(sys.argv[5]).read_text(encoding="utf-8"))
adapter_file = adapter / "adapter_model.safetensors"
digest = hashlib.sha256(adapter_file.read_bytes()).hexdigest()
if digest != expected_adapter_sha:
    raise SystemExit(f"adapter SHA mismatch: {digest}")
model_files = sorted(path for path in model.rglob("*") if path.is_file())
actual_files = {str(path.relative_to(model)): path.stat().st_size for path in model_files}
if actual_files != expected_inventory["files"]:
    raise SystemExit(f"SFT model per-file inventory mismatch: {actual_files}")
model_bytes = sum(actual_files.values())
gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"], text=True).strip()
disk = subprocess.check_output(["df", "-h", "/root/autodl-tmp"], text=True).strip().splitlines()[-1]
payload = {
    "valid": True,
    "gpu": gpu,
    "data_disk": disk,
    "sft_model_path": str(model),
    "sft_model_file_count": len(actual_files),
    "sft_model_total_bytes": model_bytes,
    "sft_model_manifest_sha256_from_day19": "e4512f7dd0f85b2f1d8154d123aec176295f557046142aab406a030c8e76e971",
    "adapter_path": str(adapter),
    "adapter_sha256": digest,
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
write_status "$CURRENT_STAGE" completed

CURRENT_STAGE=merge
if [[ -e "$DPO_MODEL" ]]; then
  echo "Refusing to overwrite existing DPO merged model: $DPO_MODEL" >&2
  exit 3
fi
write_status "$CURRENT_STAGE" running
llamafactory-cli export "$CODE_ROOT/configs/qwen25_7b_week4_dpo_export.yaml" 2>&1 | tee "$RESULTS_ROOT/merge_log.txt"
python "$CODE_ROOT/source/scripts/validate_merged_model.py" \
  --model-dir "$DPO_MODEL" \
  --output "$RESULTS_ROOT/merged_model_manifest.json"
write_status "$CURRENT_STAGE" completed

CURRENT_STAGE=smoke
write_status "$CURRENT_STAGE" running
python "$CODE_ROOT/source/scripts/smoke_merged_model.py" \
  --model-path "$DPO_MODEL" \
  --output "$RESULTS_ROOT/merged_model_smoke.json" \
  2>&1 | tee "$RESULTS_ROOT/merged_model_smoke.log"
write_status "$CURRENT_STAGE" completed

CURRENT_STAGE=evaluate_sft_only
write_status "$CURRENT_STAGE" running
python "$CODE_ROOT/source/scripts/run_model_evaluation.py" \
  --config "$CODE_ROOT/configs/evaluation_config.json" \
  --model-dir "$SFT_MODEL" \
  --model-alias sft_only \
  --output "$RESULTS_ROOT/sft_only_responses.jsonl" \
  --manifest-output "$RESULTS_ROOT/sft_only_manifest.json" \
  2>&1 | tee "$RESULTS_ROOT/sft_only_evaluation.log"
write_status "$CURRENT_STAGE" completed

CURRENT_STAGE=evaluate_sft_dpo
write_status "$CURRENT_STAGE" running
python "$CODE_ROOT/source/scripts/run_model_evaluation.py" \
  --config "$CODE_ROOT/configs/evaluation_config.json" \
  --model-dir "$DPO_MODEL" \
  --model-alias sft_dpo \
  --output "$RESULTS_ROOT/sft_dpo_responses.jsonl" \
  --manifest-output "$RESULTS_ROOT/sft_dpo_manifest.json" \
  2>&1 | tee "$RESULTS_ROOT/sft_dpo_evaluation.log"
write_status "$CURRENT_STAGE" completed

CURRENT_STAGE=complete
write_status "$CURRENT_STAGE" completed
