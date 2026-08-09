#!/usr/bin/env bash
# Build ACL'25 epistemic-marker dashboard for:
#   GPT-4o, DeepSeek, Gemma, Qwen × FEVER, DebateQA, PopQA
#   (temp 0.6, verbal scale 1–10, elicitation-mode=epistemic-marker)
#
# Inputs:
#   1) Factorial T1_P1_S1 — GPT+DeepSeek on FEVER+PopQA (already collected)
#   2) DebateQA epistemic rollouts for GPT+DeepSeek
#   3) FEVER+PopQA+DebateQA epistemic rollouts for Qwen+Gemma
#
# Writes under:
#   output_wood/epistemic_markers/four_models_fever_debateqa_popqa_temp06_scale10/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/src/build_epistemic_qwen_gemma_openai_deepseek_plots.py"
TE="$ROOT/output_wood/temperature_experiments"
EP="$ROOT/output_wood/epistemic_markers"
ROLL="$EP/rollouts_temp06_scale10/csv"
OUT_DIR="$EP/four_models_fever_debateqa_popqa_temp06_scale10"
FACT="$ROOT/output_wood/factorial_experiments_fullrun/csv/rollout_T1_P1_S1.csv"
mkdir -p "$OUT_DIR" "$ROLL"

# Extract GPT+DeepSeek × FEVER+PopQA from factorial (dedupe GPT double-rows).
EXTRACT="$ROLL/openai_deepseek_fever_popqa_epistemic_from_factorial_T1_P1_S1.csv"
python3 - <<PY
from pathlib import Path
import pandas as pd

fact = Path("$FACT")
out = Path("$EXTRACT")
keep_models = {"openai/gpt-4o-2024-11-20", "deepseek/deepseek-chat-v2.5"}
keep_ds = {"fever", "conflictqa_popqa"}
df = pd.read_csv(fact)
df = df[df["model"].isin(keep_models) & df["dataset"].isin(keep_ds)].copy()
df = df.drop_duplicates(subset=["model", "dataset", "original_index"], keep="first")
df.to_csv(out, index=False)
print(f"Wrote {out} n={len(df)}")
print(df.groupby(["model", "dataset"]).size().unstack(fill_value=0))
PY

DEBATEQA_OD="$ROLL/debateqa_ece_openai_deepseek_epistemic_temp0p6_scale10.csv"
QWEN_GEMMA="$ROLL/ece_qwen_gemma_fever_popqa_debateqa_epistemic_temp0p6_scale10.csv"

missing=0
for f in "$EXTRACT" "$DEBATEQA_OD" "$QWEN_GEMMA"; do
  if [[ ! -f "$f" ]]; then
    echo "MISSING: $f" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "" >&2
  echo "Collect missing epistemic-marker rollouts first:" >&2
  echo "  bash scripts/run_epistemic_temp06_scale10_debateqa_openai_deepseek.sh" >&2
  echo "  bash scripts/run_epistemic_temp06_scale10_qwen_gemma_fever_popqa_debateqa.sh" >&2
  exit 1
fi

python3 "$SCRIPT" \
  --inputs \
    "$EXTRACT" \
    "$DEBATEQA_OD" \
    "$QWEN_GEMMA" \
  --out-dir "$OUT_DIR" \
  --seed 42 \
  --train-ratio 0.8 \
  --min-occurrences 10

echo "Dashboard: $OUT_DIR/overall_acl25_metrics.png"
