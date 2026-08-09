#!/usr/bin/env bash
# LLM-as-judge (gpt-5.4-mini via OpenRouter) on remaining model×dataset rollouts.
# Default: --flip-turns-only (enough for U^pers A; ~3.3k calls).
# Pass --all-turns as first arg to score every persuasion turn (~20k calls).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"

MODE="${1:---flip-turns-only}"
JUDGE_MODEL="${OPENROUTER_JUDGE_MODEL:-openai/gpt-5.4-mini}"
SLEEP_S="${JUDGE_SLEEP_S:-0.35}"
FEVER_META="$ROOT/output_wood/dataset_subsampling/fever/csv/fever600_200x3_complexity_wood_v1_lr_40.csv"
LOG_DIR="$ROOT/output_wood/persuasion/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/arg_quality_judge_remaining_${STAMP}.log"

CONDITIONS=(
  "Qwen:fever"
  "Qwen:popqa"
  "Qwen:debateqa"
  "Gemma:fever"
  "Gemma:popqa"
  "Gemma:debateqa"
  "GPT-4o:fever"
  "GPT-4o:popqa"
  "GPT-4o:debateqa"
)

echo "Mode=$MODE judge=$JUDGE_MODEL sleep=${SLEEP_S}s" | tee "$LOG"
echo "Log: $LOG" | tee -a "$LOG"

for cond in "${CONDITIONS[@]}"; do
  MODEL="${cond%%:*}"
  DS="${cond##*:}"
  for rel in "rollout/csv/expl.csv" "csv/expl.csv"; do
    INPUT="$ROOT/output_wood/persuasion/$MODEL/$DS/$rel"
    if [[ -f "$INPUT" ]]; then
      break
    fi
  done
  if [[ ! -f "$INPUT" ]]; then
    echo "SKIP $MODEL/$DS (no expl.csv)" | tee -a "$LOG"
    continue
  fi

  OUT_DIR="$ROOT/output_wood/persuasion/$MODEL/$DS/arg_quality"
  mkdir -p "$OUT_DIR"
  if [[ "$MODE" == "--all-turns" ]]; then
    RUN_NAME="${DS}_all_turns_v1"
    EXTRA=(--all-turns)
  else
    RUN_NAME="${DS}_flip_turns_v1"
    EXTRA=(--flip-turns-only)
  fi

  echo "" | tee -a "$LOG"
  echo "=== $MODEL / $DS  run=$RUN_NAME ===" | tee -a "$LOG"
  META_ARGS=()
  if [[ "$DS" == "fever" && -f "$FEVER_META" ]]; then
    META_ARGS=(--fever-meta "$FEVER_META")
  fi

  python3 run_persuasion_arg_quality_judge.py \
    --input "$INPUT" \
    ${META_ARGS[@]+"${META_ARGS[@]}"} \
    --out-dir "$OUT_DIR" \
    "${EXTRA[@]}" \
    --judge openrouter \
    --openrouter-model "$JUDGE_MODEL" \
    --run-name "$RUN_NAME" \
    --quality-scheme top3 \
    --sleep-s "$SLEEP_S" \
    2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "All remaining judge runs finished." | tee -a "$LOG"
