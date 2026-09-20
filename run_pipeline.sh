#!/usr/bin/env bash
set -euo pipefail
week8_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
week8_python="${PIPELINE_PYTHON:-python3}"
week8_quick=0; week8_skip_train=0; week8_skip_eval=0; week8_deploy=0
week8_score_model=''; week8_execute_score=0
week8_clean_action=''; week8_eval_plan=''; week8_eval_sha=''
week8_dry_run=0; week8_eval_limit=0; week8_eval_phase=all
week8_run="$week8_root/logs/week8-$(date +%Y%m%d-%H%M%S)-$$"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) week8_quick=1; week8_skip_train=1; shift ;;
    --skip-train) week8_skip_train=1; shift ;;
    --skip-eval) week8_skip_eval=1; shift ;;
    --deploy) week8_deploy=1; shift ;;
    --dry-run) week8_dry_run=1; shift ;;
    --eval-limit-per-subject) [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || exit 2; week8_eval_limit="$2"; shift 2 ;;
    --eval-phase) [[ $# -ge 2 ]] || exit 2; case "$2" in all|generate) ;; *) exit 2 ;; esac; week8_eval_phase="$2"; shift 2 ;;
    --clean-eval) [[ $# -ge 2 && -n "$2" && -z "$week8_clean_action" ]] || exit 2; week8_clean_action="$2"; shift 2 ;;
    --eval-plan) [[ $# -ge 2 && -n "$2" && -z "$week8_eval_plan" ]] || exit 2; week8_eval_plan="$2"; shift 2 ;;
    --eval-plan-sha256) [[ $# -ge 2 && -n "$2" && -z "$week8_eval_sha" ]] || exit 2; week8_eval_sha="$2"; shift 2 ;;
    --score-only) [[ $# -ge 2 ]] || exit 2; [[ -n "$2" && -z "$week8_score_model" ]] || exit 2; week8_score_model="$2"; shift 2 ;;
    --execute-score) week8_execute_score=1; shift ;;
    --run-dir) [[ $# -ge 2 ]] || exit 2; week8_run="$2"; shift 2 ;;
    --help) echo 'run_pipeline.sh [--quick] [--skip-train] [--skip-eval] [--deploy] [--run-dir DIR]'
      echo '  --dry-run: prepare real data and plans; no training/inference/API/deployment'
      echo '  --eval-limit-per-subject N: explicit sample validation (default 0 = full)'
      echo '  --eval-phase {all|generate}: optionally stop before Gemini scoring'
      echo 'run_pipeline.sh --clean-eval {audit|generate|score|verify-score} --eval-plan FILE --eval-plan-sha256 SHA --run-dir DIR'
      echo 'run_pipeline.sh --score-only {original_base|final_sft|final_dpo|all} [--execute-score] [--run-dir NEW_DIR]'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if [[ -n "$week8_clean_action$week8_score_model" ]]; then
  [[ "$week8_dry_run" == 0 && "$week8_eval_limit" == 0 && "$week8_eval_phase" == all ]] || { echo 'legacy modes cannot combine with current pipeline options' >&2; exit 2; }
fi
[[ "$week8_dry_run" == 0 || "$week8_deploy" == 0 ]] || { echo 'dry-run cannot deploy' >&2; exit 2; }
[[ "$week8_quick" == 0 || "$week8_dry_run$week8_eval_limit$week8_eval_phase" == 00all ]] || { echo 'quick cannot combine with fresh evaluation options' >&2; exit 2; }
[[ "$week8_skip_eval" == 0 || "$week8_eval_limit$week8_eval_phase" == 0all ]] || { echo 'skip-eval cannot combine with evaluation options' >&2; exit 2; }
if [[ -n "$week8_clean_action" ]]; then
  [[ "$week8_quick$week8_skip_train$week8_skip_eval$week8_deploy$week8_execute_score" == 00000 && -z "$week8_score_model" ]] || { echo 'clean-eval cannot be combined with other stages' >&2; exit 2; }
  case "$week8_clean_action" in audit|generate|score|verify-score) ;; *) echo 'unknown clean-eval action' >&2; exit 2 ;; esac
  [[ -n "$week8_eval_plan" && "$week8_eval_sha" =~ ^[0-9a-f]{64}$ ]] || { echo 'clean-eval requires plan and SHA256' >&2; exit 2; }
  exec "$week8_python" "$week8_root/scripts/week8_clean_eval.py" "$week8_clean_action" --plan "$week8_eval_plan" --plan-sha256 "$week8_eval_sha" --output "$week8_run"
fi
[[ -z "$week8_eval_plan$week8_eval_sha" ]] || { echo 'eval-plan requires clean-eval' >&2; exit 2; }
if [[ -n "$week8_score_model" ]]; then
  [[ "$week8_quick$week8_skip_train$week8_skip_eval$week8_deploy" == 0000 ]] || { echo 'score-only cannot be combined with other pipeline stages' >&2; exit 2; }
  [[ "$week8_score_model" != all || "$week8_execute_score" == 0 ]] || { echo 'execute-score requires a single model' >&2; exit 2; }
  week8_score_args=(--model "$week8_score_model" --run-dir "$week8_run")
  [[ "$week8_execute_score" == 0 ]] || week8_score_args+=(--execute)
  exec "$week8_python" "$week8_root/scripts/week8_pipeline_score.py" "${week8_score_args[@]}"
fi
[[ "$week8_execute_score" == 0 ]] || { echo 'execute-score requires score-only' >&2; exit 2; }
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
  week8_train_args=(--data-dir "$week8_run/data" --run-dir "$week8_run/training" --base-model "$BASE_MODEL_PATH")
  [[ "$week8_dry_run" == 0 ]] || week8_train_args+=(--dry-run)
  PIPELINE_PYTHON="$week8_python" bash "$week8_root/scripts/step2_train.sh" "${week8_train_args[@]}"
  week8_model="$week8_run/training/models/final_dpo"
else
  week8_model="${EVAL_MODEL_PATH:-}"
fi
if [[ "$week8_skip_eval" == 0 ]]; then
  if [[ "$week8_quick" == 1 ]]; then
    "$week8_python" "$week8_root/scripts/pipeline/step3_eval.py" --mode replay --output-dir "$week8_run/evaluation"
  else
    : "${week8_model:?Set EVAL_MODEL_PATH when skipping training}"
    week8_eval_args=(--mode fresh --model "$week8_model" --output-dir "$week8_run/evaluation" --limit-per-subject "$week8_eval_limit" --phase "$week8_eval_phase")
    [[ "$week8_dry_run" == 0 ]] || week8_eval_args+=(--dry-run)
    "$week8_python" "$week8_root/scripts/pipeline/step3_eval.py" "${week8_eval_args[@]}"
  fi
fi
if [[ "$week8_deploy" == 1 ]]; then
  bash "$week8_root/scripts/step4_deploy.sh" --run-dir "$week8_run/deployment"
fi
echo "Pipeline finished: $week8_run (quick=$week8_quick; dry_run=$week8_dry_run; skip_train=$week8_skip_train; skip_eval=$week8_skip_eval)"
