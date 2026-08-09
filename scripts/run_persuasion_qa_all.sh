#!/usr/bin/env bash
# Run persuasion for GPT-4o, Gemma, and Qwen as targets on PopQA + DebateQA.
# Persuader: openai/gpt-5.4-mini (OpenRouter). Mode: until_flip, cap 15 turns.
# Full 480-item stratified Wood subsets (same as the calibration line).
#
# 3 targets x 2 datasets = 6 rollouts, written under
#   output_wood/persuasion/<TARGET_LABEL>/<popqa|debateqa>/rollout/csv/expl.csv
#
# Optional: set MAX_ITEMS to run a smaller smoke test across all six.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT/scripts/run_persuasion_qa_target.sh"

# label | provider | model id
TARGETS=(
  "GPT-4o|openai|gpt-4o-2024-11-20"
  "Gemma|openrouter|google/gemma-4-26b-a4b-it"
  "Qwen|openrouter|qwen/qwen3-14b"
)
DATASETS=(conflictqa_popqa debateqa)

for entry in "${TARGETS[@]}"; do
  IFS='|' read -r label provider model <<<"$entry"
  for ds in "${DATASETS[@]}"; do
    echo "=== target=$label dataset=$ds ==="
    TARGET_LABEL="$label" TARGET_PROVIDER="$provider" TARGET_MODEL="$model" \
      DATASET="$ds" MAX_ITEMS="${MAX_ITEMS:-}" MAX_TURNS="${MAX_TURNS:-15}" \
      bash "$RUN"
  done
done

echo "All persuasion rollouts complete."
