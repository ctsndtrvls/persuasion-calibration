#!/usr/bin/env bash
# Mixed 100-item persuasion (FEVER + PopQA + DebateQA), until flip or max 15 turns.
# Target: DeepSeek; persuader: GPT-5.4-mini (OpenRouter).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIXED_CSV="$ROOT/output_wood/dataset_subsampling/mixed/csv/mixed_100_34_33_33_fever_popqa_debateqa.csv"

LOG_DIR="$ROOT/output_wood/persuasion/DeepSeek/mixed/rollout/logs"
mkdir -p "$LOG_DIR"

cd "$ROOT/src"
python3 run_persuasion_mixed.py \
  --mixed-input "$MIXED_CSV" \
  --persuader-model "${PERSUADER_MODEL:-openai/gpt-5.4-mini}" \
  --target-model "${TARGET_MODEL:-deepseek-chat}" \
  --persuasion-mode until_flip \
  --max-turns "${MAX_TURNS:-15}" \
  --target-temperature 0.6 \
  --persuader-temperature 0.8 \
  --sleep-s 0.35
