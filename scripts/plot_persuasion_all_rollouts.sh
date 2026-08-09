#!/usr/bin/env bash
# Plot persuasion figures for all completed GPT/Gemma/Qwen rollouts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLOT="$ROOT/scripts/plot_persuasion_rollout.sh"

for LABEL in GPT-4o Gemma Qwen; do
  for DS in popqa debateqa; do
    CSV="$ROOT/output_wood/persuasion/$LABEL/$DS/rollout/csv/expl.csv"
    if [[ -s "$CSV" ]]; then
      echo "=== plotting $LABEL / $DS ==="
      bash "$PLOT" "$LABEL" "$DS"
    else
      echo "=== skip $LABEL / $DS (no data) ==="
    fi
  done
done

echo "Done."
