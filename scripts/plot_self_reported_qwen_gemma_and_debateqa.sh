#!/usr/bin/env bash
# Deprecated: use scripts/plot_self_reported_four_models_fever_debateqa_popqa.sh for the unified chart.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec bash scripts/plot_self_reported_four_models_fever_debateqa_popqa.sh "$@"
