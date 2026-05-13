#!/usr/bin/env bash
# ECE reliability plots for temperature_experiments (temp 0.6 / scale 10 self-reported):
#   - OpenAI + DeepSeek: DebateQA only → debateqa/png/
#   - Qwen + Gemma: FEVER + DebateQA + ConflictQA PopQA (no StrategyQA in these CSVs) → qwen_gemma_fever_popqa_debateqa/png/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TE="$ROOT/output_wood/temperature_experiments"
PY="$ROOT/src/plot_ece_calibration.py"

mkdir -p "$TE/debateqa/png" "$TE/qwen_gemma_fever_popqa_debateqa/png"

python3 "$PY" \
  --input "$TE/debateqa/csv/debateqa_ece_openai_deepseek_temp0p6_scale10.csv" \
  --out-dir "$TE/debateqa/png"

python3 "$PY" \
  --input "$TE/qwen_gemma_fever_popqa_debateqa/csv/ece_qwen_gemma_fever_popqa_debateqa_temp0p6_scale10.csv" \
  --only-datasets fever debateqa conflictqa_popqa \
  --out-dir "$TE/qwen_gemma_fever_popqa_debateqa/png" \
  --file-suffix _temp06_scale10_fever_debateqa_popqa

echo "Done. PNG → $TE/debateqa/png and $TE/qwen_gemma_fever_popqa_debateqa/png"
