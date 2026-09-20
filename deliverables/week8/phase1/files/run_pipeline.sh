#!/usr/bin/env bash
set -euo pipefail
week8_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
week8_python="${PIPELINE_PYTHON:-python3}"
week8_quick=0; week8_skip_train=0; week8_skip_eval=0; week8_deploy=0
week8_run="$week8_root/logs/week8-$(date +%Y%m%d-%H%M%S)-$$"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) week8_quick=1; week8_skip_train=1; shift ;;
    --skip-train) week8_skip_train=1; shift ;;
    --skip-eval) week8_skip_eval=1; shift ;;
    --deploy) week8_deploy=1; shift ;;
    --run-dir) [[ $# -ge 2 ]] || exit 2; week8_run="$2"; shift 2 ;;
    --help) echo 'run_pipeline.sh [--quick] [--skip-train] [--skip-eval] [--deploy] [--run-dir DIR]'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$week8_quick" == 0 || "$week8_deploy" == 0 ]] || { echo 'quick cannot deploy a model' >&2; exit 2; }
mkdir -p "$week8_run"
week8_run="$(cd -- "$week8_run" && pwd)"
[[ ! -e "$week8_run/data" ]] || { echo 'Use a new --run-dir' >&2; exit 2; }
week8_data_args=(--output-dir "$week8_run/data")
if [[ "$week8_quick" == 1 ]]; then
  week8_data_args+=(--quick)
  [[ -z "${DATA_INPUT:-}" ]] || week8_data_args+=(--input "$DATA_INPUT")
else
  [[ -z "${DATA_INPUT:-}" && -z "${TOKENIZER_PATH:-}" ]] || { echo 'Formal preparation uses the frozen protocol; unset DATA_INPUT and TOKENIZER_PATH' >&2; exit 2; }
  week8_data_args+=(--protocol "$week8_root/configs/week8_data_protocol.json")
fi
"$week8_python" "$week8_root/scripts/step1_data_prep.py" "${week8_data_args[@]}" > "$week8_run/data_prep.log" 2>&1
if [[ "$week8_skip_train" == 0 ]]; then
  : "${BASE_MODEL_PATH:?Set BASE_MODEL_PATH}"
  "$week8_python" "$week8_root/scripts/train_pipeline.py" --data-dir "$week8_run/data" --run-dir "$week8_run/training" --base-model "$BASE_MODEL_PATH"
  week8_model="$week8_run/training/models/final_dpo"
else
  week8_model="${EVAL_MODEL_PATH:-}"
fi
if [[ "$week8_skip_eval" == 0 ]]; then
  if [[ "$week8_quick" == 1 ]]; then
    "$week8_python" "$week8_root/scripts/step3_eval.py" --mode replay --output-dir "$week8_run/evaluation"
  else
    : "${week8_model:?Set EVAL_MODEL_PATH when skipping training}"
    : "${JUDGE_MODEL:?Set JUDGE_MODEL}" "${JUDGE_BASE_URL:?Set explicit JUDGE_BASE_URL}"
    "$week8_python" "$week8_root/scripts/step3_eval.py" --mode fresh --model "$week8_model" --judge-model "$JUDGE_MODEL" --judge-url "$JUDGE_BASE_URL" --output-dir "$week8_run/evaluation"
  fi
fi
if [[ "$week8_deploy" == 1 ]]; then
  bash "$week8_root/scripts/step4_deploy.sh" --run-dir "$week8_run/deployment"
fi
echo "Pipeline finished: $week8_run (quick=$week8_quick; skip_train=$week8_skip_train; skip_eval=$week8_skip_eval)"
