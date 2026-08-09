#!/usr/bin/env bash
# Epistemic-marker elicitation (ACL'25 style), temp 0.6, verbal scale 1–10.
# OpenAI + DeepSeek on DebateQA only (FEVER/PopQA reused from factorial T1_P1_S1).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
set -a; source "$ROOT/.env"; set +a

SCRIPT="$ROOT/src/collect_conflictqa_ece.py"
OUT_DIR="$ROOT/output_wood/epistemic_markers/rollouts_temp06_scale10/csv"
mkdir -p "$OUT_DIR"

TEMP="${TEMP:-0.6}"
STEM="debateqa_ece_openai_deepseek_epistemic_temp$(echo "$TEMP" | tr '.' 'p')_scale10"

python3 "$SCRIPT" \
  --datasets debateqa \
  --skip-anthropic \
  --skip-gemini \
  --elicitation-mode epistemic-marker \
  --openai-temperature "$TEMP" \
  --openai-confidence-scale 1-10 \
  --deepseek-temperature "$TEMP" \
  --deepseek-confidence-scale 1-10 \
  --allow-any-out-path \
  --out "$OUT_DIR/${STEM}.csv" \
  --also-split-dataset-csv "$OUT_DIR/$STEM" \
  --sleep-s 0.2 \
  --progress-every 50
