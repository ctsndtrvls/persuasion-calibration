#!/usr/bin/env bash
# Flip-turn arg-quality judge for DeepSeek PopQA + DebateQA (after persuasion rollouts).
# Uses openai/gpt-5.4-mini via OpenRouter; resumes from cache.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"

JUDGE_MODEL="${OPENROUTER_JUDGE_MODEL:-openai/gpt-5.4-mini}"
SLEEP_S="${JUDGE_SLEEP_S:-0.35}"
LOG_DIR="$ROOT/output_wood/persuasion/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/arg_quality_judge_deepseek_qa_${STAMP}.log"

echo "DeepSeek QA flip-turn judge | model=$JUDGE_MODEL" | tee "$LOG"

for DS in popqa debateqa; do
  INPUT=""
  for rel in "rollout/csv/expl.csv" "csv/expl.csv"; do
    CAND="$ROOT/output_wood/persuasion/DeepSeek/$DS/$rel"
    if [[ -f "$CAND" ]]; then
      INPUT="$CAND"
      break
    fi
  done
  if [[ -z "$INPUT" ]]; then
    echo "SKIP DeepSeek/$DS (no expl.csv)" | tee -a "$LOG"
    continue
  fi
  OUT_DIR="$ROOT/output_wood/persuasion/DeepSeek/$DS/arg_quality"
  mkdir -p "$OUT_DIR"
  echo "" | tee -a "$LOG"
  echo "=== DeepSeek / $DS ===" | tee -a "$LOG"
  python3 run_persuasion_arg_quality_judge.py \
    --input "$INPUT" \
    --out-dir "$OUT_DIR" \
    --flip-turns-only \
    --judge openrouter \
    --openrouter-model "$JUDGE_MODEL" \
    --run-name "${DS}_flip_turns_v1" \
    --quality-scheme top3 \
    --sleep-s "$SLEEP_S" \
    2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "DeepSeek QA judge finished." | tee -a "$LOG"
