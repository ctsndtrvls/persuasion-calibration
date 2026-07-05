#!/usr/bin/env bash
# Continue persuasion only for dialogues that did NOT flip after turn 1
# in the prior 1-turn rollout. Runs until flip or max 15 total persuasion turns.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRIOR_CSV="$ROOT/output_wood/persuasion/DeepSeek/fever/csv/t1_expl.csv"
FEVER_CSV="$ROOT/output_wood/dataset_subsampling/fever/csv/fever600_200x3_complexity_wood_v1_lr_40.csv"

cd "$ROOT/src"
python3 run_persuasion_fever_pilot.py \
  --fever-input "$FEVER_CSV" \
  --continue-from-csv "$PRIOR_CSV" \
  --continue-after-turn 1 \
  --persuader-model "${PERSUADER_MODEL:-openai/gpt-5.4-mini}" \
  --target-model "${TARGET_MODEL:-deepseek-chat}" \
  --max-items 600 \
  --persuasion-mode until_flip \
  --max-turns 15 \
  --complexity-level "${COMPLEXITY:-1}" \
  --out "$ROOT/output_wood/persuasion/DeepSeek/fever/csv/expl.csv" \
  --target-temperature 0.6 \
  --persuader-temperature 0.8 \
  --sleep-s 0.35
