#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/qwen25-week4/day20-remediation
CODE="$ROOT/code"
DATA="$ROOT/data"
RUNS="$ROOT/runs"
SFT=/root/autodl-tmp/qwen25-week3/best_model/qwen25-7b-week3-best-merged
EXPECTED_TRAIN=99f90060ba8b0904e561e14296722e46e1aff55f8f61ab16a0c173a720e5cfcc
EXPECTED_VALIDATION=8af26c924daee6a47daf61f44ee0d17b082d6ec55cd52e7bdbc761a7b4f84e60

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CANDIDATE CONFIG" >&2
  exit 2
fi
CANDIDATE=$1
CONFIG=$2
mkdir -p "$RUNS/$CANDIDATE"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate llm_exp

python - "$DATA" "$SFT" "$EXPECTED_TRAIN" "$EXPECTED_VALIDATION" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
data, sft, expected_train, expected_validation = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
assert sft.is_dir(), "missing SFT model"
assert sha(data / "week4_dpo_train_v2.json") == expected_train
assert sha(data / "week4_dpo_validation_v2.json") == expected_validation
train = json.loads((data / "week4_dpo_train_v2.json").read_text())
validation = json.loads((data / "week4_dpo_validation_v2.json").read_text())
assert len(train) == 783 and len(validation) == 87
print(json.dumps({"valid": True, "train": len(train), "validation": len(validation), "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"], text=True).strip()}))
PY

index=1
while [[ -e "$RUNS/$CANDIDATE/attempt_$(printf '%03d' "$index")" ]]; do index=$((index + 1)); done
ATTEMPT="$RUNS/$CANDIDATE/attempt_$(printf '%03d' "$index")"
mkdir "$ATTEMPT"

python - "$CONFIG" "$ATTEMPT/runtime.yaml" "$ATTEMPT/trainer_output" "$CANDIDATE" <<'PY'
import sys, yaml
from pathlib import Path
source, output, trainer, candidate = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
config = yaml.safe_load(source.read_text())
config["output_dir"] = trainer
config["run_name"] = candidate
config["overwrite_output_dir"] = False
config["resume_from_checkpoint"] = None
output.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
PY

python - "$ATTEMPT/environment.json" <<'PY'
import json, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
payload={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"hostname":platform.node(),"gpu":subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader"],text=True).strip()}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2)+"\n")
PY

set +e
llamafactory-cli train "$ATTEMPT/runtime.yaml" >"$ATTEMPT/launch.log" 2>&1
status=$?
set -e
python - "$ATTEMPT/status.json" "$status" <<'PY'
import json, sys
from pathlib import Path
code=int(sys.argv[2]); Path(sys.argv[1]).write_text(json.dumps({"status":"completed" if code == 0 else "failed","returncode":code},indent=2)+"\n")
PY
if [[ $status -ne 0 ]]; then exit "$status"; fi
python "$CODE/source/scripts/summarize_candidate.py" --trainer-state "$ATTEMPT/trainer_output/trainer_state.json" --adapter-dir "$ATTEMPT/trainer_output" --output "$ATTEMPT/trend_summary.json"
printf '%s\n' "$ATTEMPT"

