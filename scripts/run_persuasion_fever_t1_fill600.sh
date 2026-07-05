#!/usr/bin/env bash
# One persuasion turn (target t0 + persuader t1) for FEVER-600 claims missing from expl.csv.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"

python3 build_fever600_turn1_for_judge.py

MISSING="$ROOT/output_wood/dataset_subsampling/fever/csv/fever600_missing_persuasion.csv"
N=$(($(wc -l < "$MISSING") - 1))
if [[ "$N" -le 0 ]]; then
  echo "All 600 claims already have turn-1 counterarguments."
  exit 0
fi

echo "Running persuasion t1 for $N missing claims..."
python3 run_persuasion_fever_pilot.py \
  --fever-input "$MISSING" \
  --persuader-model "${PERSUADER_MODEL:-openai/gpt-5.4-mini}" \
  --target-model deepseek-chat \
  --persuasion-mode fixed \
  --max-turns 1 \
  --complexity-level 1 \
  --target-temperature 0.6 \
  --persuader-temperature 0.8 \
  --sleep-s 0.35 \
  --out "$ROOT/output_wood/persuasion/DeepSeek/fever/csv/t1_expl.csv"

python3 build_fever600_turn1_for_judge.py --t1-expl "$ROOT/output_wood/persuasion/DeepSeek/fever/csv/t1_expl.csv"
