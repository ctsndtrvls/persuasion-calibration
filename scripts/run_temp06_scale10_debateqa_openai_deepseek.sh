#!/usr/bin/env bash
# Self-reported confidence, scale 1–10, temperature 0.6, default user prompt (same line as legacy
# temp06_scale10 factorial T1_P0). Writes under output_wood/self_reported_confidence/ (not token-prob).
# OpenAI + DeepSeek on DebateQA only (no Claude/Gemini).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/src/collect_conflictqa_ece.py"
OUT_DIR="$ROOT/output_wood/temperature_experiments/debateqa/csv"
mkdir -p "$OUT_DIR"

TEMP="${TEMP:-0.6}"
STEM="debateqa_ece_openai_deepseek_temp$(echo "$TEMP" | tr '.' 'p')_scale10"

python3 "$SCRIPT" \
  --datasets debateqa \
  --skip-anthropic \
  --skip-gemini \
  --elicitation-mode default \
  --openai-temperature "$TEMP" \
  --openai-confidence-scale 1-10 \
  --deepseek-temperature "$TEMP" \
  --deepseek-confidence-scale 1-10 \
  --out "$OUT_DIR/${STEM}.csv" \
  --also-split-dataset-csv "$OUT_DIR/$STEM" \
  --sleep-s 0.2 \
  --progress-every 50
