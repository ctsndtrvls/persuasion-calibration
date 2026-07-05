#!/usr/bin/env bash
# 100 stratified turn-1 rows; three separate CSVs (one score block per annotator).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"

python3 run_persuasion_arg_quality_judge.py \
  --prepare-human-sample 100 \
  --input "$ROOT/output_wood/persuasion/DeepSeek/fever/csv/expl.csv" \
  --fever-meta "$ROOT/output_wood/dataset_subsampling/fever/csv/fever600_200x3_complexity_wood_v1_lr_40.csv" \
  --out-dir "$ROOT/output_wood/persuasion/DeepSeek/fever/arg_quality" \
  --seed 42
