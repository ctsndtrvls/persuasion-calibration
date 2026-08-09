#!/usr/bin/env bash
# Multi-turn persuasion on a single QA dataset (PopQA or DebateQA) for one target model.
# Persuader: openai/gpt-5.4-mini (OpenRouter). Mode: until_flip, cap 15 turns.
#
# Usage:
#   TARGET_LABEL=GPT-4o TARGET_PROVIDER=openai TARGET_MODEL=gpt-4o-2024-11-20 \
#     DATASET=conflictqa_popqa ./scripts/run_persuasion_qa_target.sh
#
# Env vars:
#   DATASET          conflictqa_popqa | debateqa            (required)
#   TARGET_PROVIDER  openai | openrouter | deepseek         (required)
#   TARGET_MODEL     provider model id                      (required)
#   TARGET_LABEL     short label for output dir             (required)
#   MAX_ITEMS        limit items (default: all 480)
#   MAX_TURNS        default 15
#   COMPLEXITY       fallback argument complexity (default 1)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${DATASET:?set DATASET=conflictqa_popqa|debateqa}"
: "${TARGET_PROVIDER:?set TARGET_PROVIDER=openai|openrouter|deepseek}"
: "${TARGET_MODEL:?set TARGET_MODEL=<provider model id>}"
: "${TARGET_LABEL:?set TARGET_LABEL=<short label, e.g. GPT-4o>}"

cd "$ROOT/src"

args=(
  run_persuasion_qa_multi.py
  --dataset "$DATASET"
  --target-provider "$TARGET_PROVIDER"
  --target-model "$TARGET_MODEL"
  --target-label "$TARGET_LABEL"
  --persuader-model "${PERSUADER_MODEL:-openai/gpt-5.4-mini}"
  --persuasion-mode until_flip
  --max-turns "${MAX_TURNS:-15}"
  --complexity-level "${COMPLEXITY:-1}"
  --target-temperature 0.6
  --persuader-temperature 0.8
  --sleep-s 0.35
)
if [[ -n "${MAX_ITEMS:-}" ]]; then
  args+=(--max-items "$MAX_ITEMS")
fi

python3 "${args[@]}"
