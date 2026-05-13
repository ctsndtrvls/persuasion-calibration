#!/usr/bin/env bash
# Token-probability (logprob-based confidence) on DebateQA for four models:
# OpenAI GPT-4o, DeepSeek Chat, Gemma 4 26B, Qwen3-14B (OpenRouter for the last two).
# Writes CSV under output_wood/token_prob_confidence/debateqa/csv/; then run:
#   ./scripts/plot_token_prob_figures.sh --which debateqa
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/src/collect_conflictqa_ece.py"
OUT_DIR="$ROOT/output_wood/token_prob_confidence/debateqa/csv"
mkdir -p "$OUT_DIR"

T="${T:-0.0}"
tag="$(echo "$T" | tr '.' 'p')"

python3 "$SCRIPT" \
  --allow-any-out-path \
  --datasets debateqa \
  --skip-anthropic \
  --skip-gemini \
  --openai-confidence-source token-prob \
  --openai-confidence-scale 0-1 \
  --openai-temperature "$T" \
  --deepseek-confidence-source token-prob \
  --deepseek-confidence-scale 0-1 \
  --deepseek-temperature "$T" \
  --extra-openrouter-tokenprob-model "google/gemma-4-26b-a4b-it" \
  --extra-openrouter-tokenprob-model "qwen/qwen3-14b" \
  --extra-openrouter-temperature "$T" \
  --out "$OUT_DIR/debateqa_tokenprob_four_models_temp${tag}.csv" \
  --sleep-s 0.15 \
  --progress-every 50 \
  "$@"
