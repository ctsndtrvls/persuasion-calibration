#!/usr/bin/env bash
# Sequential factorial runs (2×2×2) for collect_conflictqa_ece.py
# T0_P0_S0 is assumed complete (5760 rows). This script runs the other 7 cells.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
# factorial dir is .../persuasion-calibration/output_wood/<this>; repo root is ../..
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="$REPO/src/collect_conflictqa_ece.py"
CSV_DIR="$REPO/output_wood/self_reported_confidence/factorial_experiments_fullrun/csv"
LOG_DIR="$REPO/output_wood/self_reported_confidence/factorial_experiments_fullrun/logs"
mkdir -p "$CSV_DIR" "$LOG_DIR"
MASTER="$LOG_DIR/run_remaining_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MASTER") 2>&1

run_one() {
  local T="$1" P="$2" S="$3"
  local TAG="T${T}_P${P}_S${S}"
  local TEMP MODE SCALE
  if [[ "$T" == "1" ]]; then TEMP="0.6"; else TEMP="0.0"; fi
  if [[ "$P" == "1" ]]; then MODE="epistemic-marker"; else MODE="default"; fi
  if [[ "$S" == "1" ]]; then SCALE="1-10"; else SCALE="0-1"; fi
  local OUT="$CSV_DIR/rollout_${TAG}.csv"
  local SPLIT="$CSV_DIR/splits_${TAG}"
  local LOG="$LOG_DIR/${TAG}_remaining.log"
  echo "========== [start] $TAG  temp=$TEMP  mode=$MODE  scale=$SCALE  $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
  python3 "$SCRIPT" \
    --datasets all \
    --use-conflictqa-subset-csv \
    --out "$OUT" \
    --also-split-dataset-csv "$SPLIT" \
    --openai-temperature "$TEMP" \
    --anthropic-temperature "$TEMP" \
    --deepseek-temperature "$TEMP" \
    --gemini-temperature "$TEMP" \
    --openai-confidence-scale "$SCALE" \
    --anthropic-confidence-scale "$SCALE" \
    --deepseek-confidence-scale "$SCALE" \
    --gemini-confidence-scale "$SCALE" \
    --elicitation-mode "$MODE" \
    --sleep-s 0.1 \
    --progress-every 100 2>&1 | tee "$LOG"
  echo "========== [done]  $TAG  $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
}

for combo in \
  "0 0 1" \
  "0 1 0" \
  "0 1 1" \
  "1 0 0" \
  "1 0 1" \
  "1 1 0" \
  "1 1 1"; do
  set -- $combo
  run_one "$1" "$2" "$3"
done

echo "All remaining factorial cells finished."
