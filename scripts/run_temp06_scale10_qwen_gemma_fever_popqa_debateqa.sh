#!/usr/bin/env bash
# Self-reported confidence, scale 1–10, temperature 0.6, default prompt — aligned with legacy
# temp06_scale10 factorial (T1_P0). Writes under output_wood/self_reported_confidence/ (not token-prob).
# Qwen + Gemma via OpenRouter only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/src/collect_conflictqa_ece.py"
# All self-reported rollouts live under output_wood/self_reported_confidence/ (see collect_conflictqa_ece --out check).
OUT_DIR="$ROOT/output_wood/temperature_experiments/qwen_gemma_fever_popqa_debateqa/csv"
mkdir -p "$OUT_DIR"

TEMP="${TEMP:-0.6}"
STEM="ece_qwen_gemma_fever_popqa_debateqa_temp$(echo "$TEMP" | tr '.' 'p')_scale10"

python3 "$SCRIPT" \
  --datasets fever_popqa_debateqa \
  --use-conflictqa-subset-csv \
  --skip-openai \
  --skip-anthropic \
  --skip-deepseek \
  --skip-gemini \
  --elicitation-mode default \
  --extra-openrouter-selfreported-model "qwen/qwen3-14b" \
  --extra-openrouter-selfreported-model "google/gemma-4-26b-a4b-it" \
  --extra-openrouter-temperature "$TEMP" \
  --extra-openrouter-confidence-scale 1-10 \
  --out "$OUT_DIR/${STEM}.csv" \
  --also-split-dataset-csv "$OUT_DIR/$STEM" \
  --sleep-s 0.15 \
  --progress-every 50
