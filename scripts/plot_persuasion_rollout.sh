#!/usr/bin/env bash
# Generate persuasion analysis plots for one rollout (mirrors DeepSeek/FEVER png pack).
#
# Usage:
#   ./scripts/plot_persuasion_rollout.sh GPT-4o popqa
#   ./scripts/plot_persuasion_rollout.sh Gemma debateqa
#   ./scripts/plot_persuasion_rollout.sh GPT-4o fever
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${1:?model label, e.g. GPT-4o|Gemma|Qwen}"
DATASET_SLUG="${2:?dataset slug: popqa|debateqa|fever}"

case "$DATASET_SLUG" in
  popqa) DATASET_TAG="PopQA" ;;
  debateqa) DATASET_TAG="DebateQA" ;;
  fever) DATASET_TAG="FEVER" ;;
  *) echo "Unknown dataset slug: $DATASET_SLUG (use popqa, debateqa, or fever)" >&2; exit 1 ;;
esac

INPUT="$ROOT/output_wood/persuasion/$LABEL/$DATASET_SLUG/rollout/csv/expl.csv"
OUT_DIR="$ROOT/output_wood/persuasion/$LABEL/$DATASET_SLUG"

if [[ ! -s "$INPUT" ]]; then
  echo "Missing or empty rollout CSV: $INPUT" >&2
  exit 1
fi

TITLE="Persuasion Summary: $DATASET_TAG (until flip, max 15), $LABEL <- GPT-5.4-mini"

cd "$ROOT/src"
python3 analyze_persuasion_confidence.py --input "$INPUT" --out-dir "$OUT_DIR"
python3 analyze_persuasion_epistemic_calibration.py --input "$INPUT" --out-dir "$OUT_DIR"
python3 plot_persuasion_fever_summary.py --input "$INPUT" --out-dir "$OUT_DIR" --title "$TITLE"
python3 plot_persuasion_overview_table.py --input "$INPUT" --out-dir "$OUT_DIR"
python3 plot_persuasion_uncertainty_overview.py --input "$INPUT" --out-dir "$OUT_DIR"

echo "Saved plots under $OUT_DIR/png"
