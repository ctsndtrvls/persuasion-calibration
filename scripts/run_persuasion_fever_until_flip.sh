#!/usr/bin/env bash
# FEVER persuasion until target flips (or hard cap), with decision explanations.
# Target: DeepSeek; persuader: GPT-5.4-mini (OpenRouter).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FEVER_CSV="$ROOT/output_wood/dataset_subsampling/fever/csv/fever600_200x3_complexity_wood_v1_lr_40.csv"

cd "$ROOT/src"
python3 run_persuasion_fever_pilot.py \
  --fever-input "$FEVER_CSV" \
  --persuader-model "${PERSUADER_MODEL:-openai/gpt-5.4-mini}" \
  --target-model "${TARGET_MODEL:-deepseek-chat}" \
  --max-items "${MAX_ITEMS:-600}" \
  --persuasion-mode until_flip \
  --max-turns "${MAX_TURNS:-15}" \
  --complexity-level "${COMPLEXITY:-1}" \
  --target-temperature 0.6 \
  --persuader-temperature 0.8 \
  --sleep-s 0.35
