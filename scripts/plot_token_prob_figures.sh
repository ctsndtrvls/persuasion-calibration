#!/usr/bin/env bash
# Local-only: build ECE-style calibration PNGs. CSVs live under
#   output_wood/token_prob_confidence/<provider>/csv/, PNGs under .../png/.
# DebateQA four-model token-prob: debateqa/csv/debateqa_tokenprob_four_models*.csv
# (see scripts/run_tokenprob_debateqa_four_models.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MPLBACKEND="${MPLBACKEND:-Agg}"
cd "$ROOT"
exec python3 src/plot_token_prob_confidence_figures.py "$@"
