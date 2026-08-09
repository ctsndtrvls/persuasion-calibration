#!/usr/bin/env bash
# Run persuasion for GPT-4o, Gemma, and Qwen as targets on FEVER (480-item Wood subset).
# Persuader: openai/gpt-5.4-mini (OpenRouter). Mode: until_flip, cap 15 turns.
#
# Outputs:
#   output_wood/persuasion/<TARGET_LABEL>/fever/rollout/csv/expl.csv
#
# Optional: set MAX_ITEMS to run a smaller smoke test across all three.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT/scripts/run_persuasion_fever_target.sh"

TARGETS=(
  "GPT-4o|openai|gpt-4o-2024-11-20"
  "Gemma|openrouter|google/gemma-4-26b-a4b-it"
  "Qwen|openrouter|qwen/qwen3-14b"
)

for entry in "${TARGETS[@]}"; do
  IFS='|' read -r label provider model <<<"$entry"
  echo "=== target=$label dataset=fever ==="
  TARGET_LABEL="$label" TARGET_PROVIDER="$provider" TARGET_MODEL="$model" \
    MAX_ITEMS="${MAX_ITEMS:-}" MAX_TURNS="${MAX_TURNS:-15}" \
    bash "$RUN"
done

echo "All FEVER persuasion rollouts complete."
