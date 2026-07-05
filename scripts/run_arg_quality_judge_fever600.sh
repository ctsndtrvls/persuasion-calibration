#!/usr/bin/env bash
# LLM-as-judge (gpt-5.4-mini via OpenRouter) on all FEVER-600 turn-1 counterarguments.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"

JUDGE_INPUT="$ROOT/output_wood/persuasion/DeepSeek/fever/csv/fever600_turn1_for_judge.csv"
FEVER_META="$ROOT/output_wood/dataset_subsampling/fever/csv/fever600_200x3_complexity_wood_v1_lr_40.csv"
T1="$ROOT/output_wood/persuasion/DeepSeek/fever/csv/t1_expl.csv"

python3 build_fever600_turn1_for_judge.py
[[ -f "$T1" ]] && python3 build_fever600_turn1_for_judge.py --t1-expl "$T1" || true

N=$(($(wc -l < "$JUDGE_INPUT") - 1))
if [[ "$N" -lt 600 ]]; then
  echo "[warn] Only $N/600 turn-1 counterarguments ready. Run: bash scripts/run_persuasion_fever_t1_fill600.sh"
fi

python3 run_persuasion_arg_quality_judge.py \
  --input "$JUDGE_INPUT" \
  --fever-meta "$FEVER_META" \
  --fever-subsample "$FEVER_META" \
  --out-dir "$ROOT/output_wood/persuasion/DeepSeek/fever/arg_quality" \
  --turn 1 \
  --judge openrouter \
  --openrouter-model "${OPENROUTER_JUDGE_MODEL:-openai/gpt-5.4-mini}" \
  --run-name fever600_v1 \
  --sleep-s 0.5
