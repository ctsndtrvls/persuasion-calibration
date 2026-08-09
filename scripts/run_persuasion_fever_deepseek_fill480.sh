#!/usr/bin/env bash
# Fill DeepSeek FEVER persuasion to the canonical 480-item Wood subset.
#
# DeepSeek pilot used a 214-claim slice (mostly fever600); GPT-4o/Gemma/Qwen
# ran on fever480. This appends the missing fever480 dialogues to expl.csv
# (already-finished original_index rows are skipped automatically).
#
# Output: output_wood/persuasion/DeepSeek/fever/csv/expl.csv
# For plots, filter to fever480 original_index → n=480 per model.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/src"

FEVER480="$ROOT/output_wood/dataset_subsampling/fever/csv/fever480_160x3_complexity_wood_v1_lr_40_resplit.csv"
OUT="$ROOT/output_wood/persuasion/DeepSeek/fever/csv/expl.csv"

args=(
  run_persuasion_fever_pilot.py
  --fever-input "$FEVER480"
  --target-provider deepseek
  --target-model deepseek-chat
  --persuader-model "${PERSUADER_MODEL:-openai/gpt-5.4-mini}"
  --persuasion-mode until_flip
  --max-turns "${MAX_TURNS:-15}"
  --target-temperature 0.6
  --persuader-temperature 0.8
  --sleep-s 0.35
  --out "$OUT"
)
if [[ -n "${MAX_ITEMS:-}" ]]; then
  args+=(--max-items "$MAX_ITEMS")
fi

python3 "${args[@]}"
