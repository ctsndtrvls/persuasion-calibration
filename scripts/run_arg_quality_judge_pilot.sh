#!/usr/bin/env bash
# Pilot LLM-judge (30 turn-1 counterarguments). Requires OPENROUTER_API_KEY in .env.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"

python3 run_persuasion_arg_quality_judge.py \
  --input "$ROOT/output_wood/persuasion/DeepSeek/fever/csv/expl.csv" \
  --fever-meta "$ROOT/output_wood/dataset_subsampling/fever/csv/fever600_200x3_complexity_wood_v1_lr_40.csv" \
  --out-dir "$ROOT/output_wood/persuasion/DeepSeek/fever/arg_quality" \
  --turn 1 \
  --max-items 30 \
  --judge openrouter \
  --openrouter-model "${OPENROUTER_JUDGE_MODEL:-openai/gpt-5.4-mini}" \
  --run-name pilot_v1
