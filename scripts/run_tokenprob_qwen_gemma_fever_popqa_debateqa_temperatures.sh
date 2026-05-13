#!/usr/bin/env bash
# Qwen + Gemma via OpenRouter (token-prob) on FEVER + ConflictQA PopQA + DebateQA (no StrategyQA).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/src/collect_conflictqa_ece.py"
# Token-prob Qwen+Gemma on FEVER + ConflictQA PopQA + DebateQA (split mirrors via --also-split-dataset-csv).
OUT_DIR="$ROOT/output_wood/temperature_experiments/qwen_gemma_tokenprob/csv"
mkdir -p "$OUT_DIR"

TEMPS="${TEMPS:-0.0 0.3 0.6 0.7}"

for T in $TEMPS; do
  tag="$(echo "$T" | tr '.' 'p')"
  echo "=== Qwen+Gemma FEVER+PopQA+DebateQA token-prob temperature=$T ==="
  python3 "$SCRIPT" \
    --allow-any-out-path \
    --datasets fever_popqa_debateqa \
    --use-conflictqa-subset-csv \
    --skip-openai \
    --skip-anthropic \
    --skip-deepseek \
    --skip-gemini \
    --extra-openrouter-tokenprob-model "qwen/qwen3-14b" \
    --extra-openrouter-tokenprob-model "google/gemma-4-26b-a4b-it" \
    --extra-openrouter-temperature "$T" \
    --out "$OUT_DIR/tokenprob_qwen_gemma_fpd_temp${tag}.csv" \
    --also-split-dataset-csv "$OUT_DIR/tokenprob_qwen_gemma_fpd_temp${tag}" \
    --sleep-s 0.15 \
    --progress-every 50
done
