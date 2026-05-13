#!/usr/bin/env bash
# Single-temperature smoke for Qwen+Gemma on FEVER + ConflictQA PopQA + DebateQA (no StrategyQA).
# Full temperature grids: scripts/run_tokenprob_qwen_gemma_fever_popqa_debateqa_temperatures.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/src/collect_conflictqa_ece.py"
OUT_DIR="$ROOT/output_wood/temperature_experiments/qwen_gemma_tokenprob/csv"

mkdir -p "$OUT_DIR"

python3 "$SCRIPT" \
  --allow-any-out-path \
  --datasets fever_popqa_debateqa \
  --use-conflictqa-subset-csv \
  --skip-openai \
  --skip-anthropic \
  --skip-deepseek \
  --skip-gemini \
  --extra-openrouter-tokenprob-model "google/gemma-4-26b-a4b-it" \
  --extra-openrouter-tokenprob-model "qwen/qwen3-14b" \
  --extra-openrouter-temperature 0.0 \
  --out "$OUT_DIR/tokenprob_new_models.csv" \
  --also-split-dataset-csv "$OUT_DIR/tokenprob_new_models" \
  --sleep-s 0.1 \
  --progress-every 100
