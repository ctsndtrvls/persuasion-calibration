#!/usr/bin/env bash
# Retry dialogues that stopped with OpenRouter 402 / credit errors in the continue CSV.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTINUE_CSV="$ROOT/output_wood/persuasion/DeepSeek/fever/csv/expl.csv"
FEVER_CSV="$ROOT/output_wood/dataset_subsampling/fever/csv/fever600_200x3_complexity_wood_v1_lr_40.csv"

cd "$ROOT/src"
python3 run_persuasion_fever_pilot.py \
  --fever-input "$FEVER_CSV" \
  --retry-credit-errors-from-csv "$CONTINUE_CSV" \
  --out "$CONTINUE_CSV" \
  --persuader-model "${PERSUADER_MODEL:-openai/gpt-5.4-mini}" \
  --target-model "${TARGET_MODEL:-deepseek-chat}" \
  --max-items 600 \
  --persuasion-mode until_flip \
  --max-turns 15 \
  --persuader-max-tokens "${PERSUADER_MAX_TOKENS:-512}" \
  --complexity-level "${COMPLEXITY:-1}" \
  --target-temperature 0.6 \
  --persuader-temperature 0.8 \
  --sleep-s 0.35
