#!/usr/bin/env bash
# LLM-as-judge on ALL persuasion counterarguments (turns 1..15) in expl.csv.
# Seeds turn-1 scores from the prior fever214 turn-1 run when available.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"

OUT_DIR="$ROOT/output_wood/persuasion/DeepSeek/fever/arg_quality"
RUN_NAME="fever214_all_turns_v1"
JUDGE_SLUG="openrouter__openai_gpt-5.4-mini"
OUT_CSV="$OUT_DIR/arg_quality_long_${RUN_NAME}__top3__${JUDGE_SLUG}.csv"
T1_CSV="$OUT_DIR/arg_quality_long_fever214_no_flip_t1_v1__top3__${JUDGE_SLUG}.csv"

mkdir -p "$OUT_DIR"
if [[ ! -f "$OUT_CSV" && -f "$T1_CSV" ]]; then
  cp "$T1_CSV" "$OUT_CSV"
  echo "Seeded turn-1 scores from $(basename "$T1_CSV")"
fi

python3 run_persuasion_arg_quality_judge.py \
  --input "$ROOT/output_wood/persuasion/DeepSeek/fever/csv/expl.csv" \
  --fever-meta "$ROOT/output_wood/dataset_subsampling/fever/csv/fever600_200x3_complexity_wood_v1_lr_40.csv" \
  --out-dir "$OUT_DIR" \
  --all-turns \
  --judge openrouter \
  --openrouter-model "${OPENROUTER_JUDGE_MODEL:-openai/gpt-5.4-mini}" \
  --run-name "$RUN_NAME" \
  --quality-scheme top3 \
  --sleep-s 0.35

echo "Done. Output -> $OUT_CSV"
