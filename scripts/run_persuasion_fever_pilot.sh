#!/usr/bin/env bash
# Small FEVER persuasion smoke test (until_flip, cap 15 turns).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/src"

python3 run_persuasion_fever_pilot.py \
  --persuader-model "${PERSUADER_MODEL:-openai/gpt-5.4-mini}" \
  --target-model deepseek-chat \
  --max-items "${MAX_ITEMS:-5}" \
  --persuasion-mode until_flip \
  --max-turns "${MAX_TURNS:-15}" \
  --complexity-level "${COMPLEXITY:-1}" \
  --target-temperature 0.6 \
  --persuader-temperature 0.8 \
  --sleep-s 0.35
