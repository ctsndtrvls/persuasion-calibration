#!/usr/bin/env bash
# Epistemic-marker summary (ACL'25 metrics + per-model/per-dataset plots + 4x3 summary grid)
# across four models on three datasets, temperature 0.6, verbal confidence scale 1-10:
#
#   models  : Qwen3-14B, Gemma 4 26B, GPT-4o-2024-11-20, DeepSeek Chat v2.5
#   datasets: FEVER, DebateQA, ConflictQA-PopQA
#
# Combines:
#   - Qwen + Gemma rollout across all three datasets
#   - OpenAI/DeepSeek dedicated FEVER, PopQA (conflictqa split) and DebateQA rollouts
#
# Writes under output_wood/epistemic_markers/four_models_fever_debateqa_popqa_temp06_scale10/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/src/build_epistemic_qwen_gemma_openai_deepseek_plots.py"

TE="$ROOT/output_wood/temperature_experiments"
QWEN_GEMMA_CSV="$TE/qwen_gemma_fever_popqa_debateqa/csv/ece_qwen_gemma_fever_popqa_debateqa_temp0p6_scale10.csv"
OPENAI_DEEPSEEK_DEBATEQA_CSV="$TE/debateqa/csv/debateqa_ece_openai_deepseek_temp0p6_scale10.csv"
OPENAI_FEVER_CSV="$TE/fever/csv/conflictqa_ece_openai_fever_temp06_scale10.csv"
DEEPSEEK_FEVER_CSV="$TE/fever/csv/conflictqa_ece_deepseek_fever_temp06_scale10.csv"
OPENAI_POPQA_CSV="$TE/conflictqa/csv/conflictqa_ece_openai_temp06_scale10_popqa.csv"
DEEPSEEK_POPQA_CSV="$TE/conflictqa/csv/conflictqa_ece_deepseek_temp06_scale10_popqa.csv"

OUT_DIR="$ROOT/output_wood/epistemic_markers/four_models_fever_debateqa_popqa_temp06_scale10"
mkdir -p "$OUT_DIR"

python3 "$SCRIPT" \
  --inputs \
    "$QWEN_GEMMA_CSV" \
    "$OPENAI_DEEPSEEK_DEBATEQA_CSV" \
    "$OPENAI_FEVER_CSV" \
    "$DEEPSEEK_FEVER_CSV" \
    "$OPENAI_POPQA_CSV" \
    "$DEEPSEEK_POPQA_CSV" \
  --out-dir "$OUT_DIR" \
  --seed 42 \
  --train-ratio 0.8 \
  --min-occurrences 10
