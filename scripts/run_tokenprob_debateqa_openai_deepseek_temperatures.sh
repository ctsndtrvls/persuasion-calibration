#!/usr/bin/env bash
# Token-prob runs on DebateQA only: OpenAI + DeepSeek (no Claude/Gemini).
# Temperature grid matches prior factorial-style sweeps; override with TEMPS="0.6" etc.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/src/collect_conflictqa_ece.py"
# Same tree as legacy temp06 self-reported runs: output_wood/temperature_experiments/<dataset>/csv/
OUT_DIR="$ROOT/output_wood/temperature_experiments/debateqa/csv"
mkdir -p "$OUT_DIR"

TEMPS="${TEMPS:-0.0 0.3 0.6 0.7}"

for T in $TEMPS; do
  tag="$(echo "$T" | tr '.' 'p')"
  echo "=== DebateQA token-prob OpenAI+DeepSeek temperature=$T ==="
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
    --out "$OUT_DIR/debateqa_tokenprob_openai_deepseek_temp${tag}.csv" \
    --also-split-dataset-csv "$OUT_DIR/debateqa_tokenprob_openai_deepseek_temp${tag}" \
    --sleep-s 0.2 \
    --progress-every 50
done
