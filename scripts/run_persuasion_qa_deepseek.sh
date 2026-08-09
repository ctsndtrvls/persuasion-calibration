#!/usr/bin/env bash
# DeepSeek persuasion rollouts on PopQA + DebateQA (same 480 Wood subsets as other models).
# Target: deepseek-chat via DeepSeek API. Persuader: openai/gpt-5.4-mini (OpenRouter).
# Mode: until_flip, cap 15 turns. Resumable via item_id cache in expl.csv.
#
# Outputs:
#   output_wood/persuasion/DeepSeek/popqa/rollout/csv/expl.csv
#   output_wood/persuasion/DeepSeek/debateqa/rollout/csv/expl.csv
#
# Optional: MAX_ITEMS=5 for a smoke test.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT/scripts/run_persuasion_qa_target.sh"
LOG_DIR="$ROOT/output_wood/persuasion/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/run_deepseek_qa_${STAMP}.log"

DATASETS=(conflictqa_popqa debateqa)

echo "DeepSeek PopQA+DebateQA persuasion | log=$LOG" | tee "$LOG"

for ds in "${DATASETS[@]}"; do
  echo "" | tee -a "$LOG"
  echo "=== target=DeepSeek dataset=$ds ===" | tee -a "$LOG"
  TARGET_LABEL=DeepSeek \
    TARGET_PROVIDER=deepseek \
    TARGET_MODEL="${TARGET_MODEL:-deepseek-chat}" \
    DATASET="$ds" \
    MAX_ITEMS="${MAX_ITEMS:-}" \
    MAX_TURNS="${MAX_TURNS:-15}" \
    bash "$RUN" 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "DeepSeek QA persuasion rollouts complete." | tee -a "$LOG"
