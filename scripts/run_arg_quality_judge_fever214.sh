#!/usr/bin/env bash
# LLM-as-judge on all 214 turn-1 counterarguments (no-flip-at-t1 continue cohort).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"

python3 run_persuasion_arg_quality_judge.py \
  --input "$ROOT/output_wood/persuasion/DeepSeek/fever/csv/expl.csv" \
  --fever-meta "$ROOT/output_wood/dataset_subsampling/fever/csv/fever600_200x3_complexity_wood_v1_lr_40.csv" \
  --out-dir "$ROOT/output_wood/persuasion/DeepSeek/fever/arg_quality" \
  --turn 1 \
  --judge openrouter \
  --openrouter-model "${OPENROUTER_JUDGE_MODEL:-openai/gpt-5.4-mini}" \
  --run-name fever214_no_flip_t1_v1 \
  --quality-scheme top3 \
  --sleep-s 0.5
