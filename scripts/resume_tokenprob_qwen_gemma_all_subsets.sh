#!/usr/bin/env bash
# Resume OpenRouter token-prob rollout for Qwen3-14B + Gemma 4 26B on ConflictQA (popQA+strategyQA) + FEVER subsets.
# Appends to existing CSV; already-present (model, dataset, original_index) rows are skipped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/src/collect_conflictqa_ece.py"
OUT_DIR="$ROOT/output_wood/token_prob_confidence/qwen-gemma"
CSV_DIR="$OUT_DIR/csv"
OUT_CSV="$CSV_DIR/tokenprob_new_models_v2.csv"
STEM="$CSV_DIR/tokenprob_new_models_v2"

mkdir -p "$CSV_DIR" "$OUT_DIR/png" "$OUT_DIR/logs"

exec python3 "$SCRIPT" \
  --allow-any-out-path \
  --datasets all \
  --use-conflictqa-subset-csv \
  --use-openrouter-for-nonopenai \
  --skip-openai \
  --skip-anthropic \
  --skip-deepseek \
  --skip-gemini \
  --extra-openrouter-tokenprob-model "google/gemma-4-26b-a4b-it" \
  --extra-openrouter-tokenprob-model "qwen/qwen3-14b" \
  --extra-openrouter-temperature 0.0 \
  --out "$OUT_CSV" \
  --also-split-dataset-csv "$STEM" \
  --sleep-s 0.1 \
  --progress-every 25 \
  "$@"
