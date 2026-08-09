#!/usr/bin/env bash
# Collect DeepSeek token-prob for persuasion214 claims missing from fever480, then merge.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"

MISSING41="$ROOT/output_wood/token_prob_confidence/deepseek/csv/fever600_persuasion214_missing41.csv"
SUPPLEMENT="$ROOT/output_wood/token_prob_confidence/deepseek/csv/deepseek_tokenprob_fever600_persuasion214_missing41_20260714.csv"

if [[ ! -f "$MISSING41" ]]; then
  python3 <<'PY'
from pathlib import Path
import pandas as pd

root = Path("..")
features = pd.read_csv(root / "output_wood/persuasion/DeepSeek/fever/composite_score/01_features/csv/persuasion_uncertainty_features.csv")
tok = pd.read_csv(root / "output_wood/token_prob_confidence/deepseek/csv/deepseek_tokenprob_fever480_160x3_complexity_wood_v1_lr_40_resplit_20260504_162309.csv")
tok = tok.rename(columns={"row_key": "original_index"})
missing = sorted(set(features["original_index"].astype(int)) - set(tok["original_index"].astype(int)))
f600 = pd.read_csv(root / "output_wood/dataset_subsampling/fever/csv/fever600_200x3_complexity_wood_v1_lr_40.csv")
out = root / "output_wood/token_prob_confidence/deepseek/csv/fever600_persuasion214_missing41.csv"
f600[f600["original_index"].isin(missing)][["original_index", "claim", "label"]].to_csv(out, index=False)
print(f"Wrote {len(missing)} missing claims to {out}")
PY
fi

if [[ ! -f "$SUPPLEMENT" ]]; then
  python3 collect_conflictqa_ece.py \
    --datasets fever \
    --fever-input "$MISSING41" \
    --skip-openai --skip-anthropic --skip-gemini \
    --deepseek-confidence-source token-prob \
    --deepseek-confidence-scale 0-1 \
    --deepseek-temperature 0.6 \
    --deepseek-model deepseek-chat \
    --deepseek-model-id deepseek-chat \
    --allow-any-out-path \
    --out "$SUPPLEMENT"
fi

python3 -c "from persuasion_tokenprob import merge_persuasion_tokenprob; merge_persuasion_tokenprob()"
python3 build_persuasion_uncertainty_scores.py
python3 evaluate_persuasion_uncertainty.py
python3 cross_validate_persuasion_uncertainty.py
python3 incremental_persuasion_regression.py
