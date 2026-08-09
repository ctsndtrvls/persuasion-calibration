#!/usr/bin/env bash
# Epistemic-marker elicitation (ACL'25 style), temp 0.6, verbal scale 1–10.
# Qwen3-14B + Gemma 4 26B via OpenRouter on FEVER + PopQA + DebateQA.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
set -a; source "$ROOT/.env"; set +a

SCRIPT="$ROOT/src/collect_conflictqa_ece.py"
OUT_DIR="$ROOT/output_wood/epistemic_markers/rollouts_temp06_scale10/csv"
mkdir -p "$OUT_DIR"

TEMP="${TEMP:-0.6}"
STEM="ece_qwen_gemma_fever_popqa_debateqa_epistemic_temp$(echo "$TEMP" | tr '.' 'p')_scale10"

python3 "$SCRIPT" \
  --datasets fever_popqa_debateqa \
  --use-conflictqa-subset-csv \
  --skip-openai \
  --skip-anthropic \
  --skip-deepseek \
  --skip-gemini \
  --elicitation-mode epistemic-marker \
  --extra-openrouter-selfreported-model "qwen/qwen3-14b" \
  --extra-openrouter-selfreported-model "google/gemma-4-26b-a4b-it" \
  --extra-openrouter-temperature "$TEMP" \
  --extra-openrouter-confidence-scale 1-10 \
  --allow-any-out-path \
  --out "$OUT_DIR/${STEM}.csv" \
  --also-split-dataset-csv "$OUT_DIR/$STEM" \
  --sleep-s 0.15 \
  --progress-every 50
