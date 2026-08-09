#!/usr/bin/env bash
# ECE reliability diagrams for the four-model baseline line:
#   GPT-4o, DeepSeek Chat v2.5, Gemma 4 26B, Qwen3-14B
#   × FEVER, DebateQA, ConflictQA PopQA (temp 0.6, verbal scale 1–10).
#
# Writes under output_wood/temperature_experiments/summary/{csv,png}/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT
TE="$ROOT/output_wood/temperature_experiments"
OUT_CSV="$TE/summary/csv"
OUT_PNG="$TE/summary/png"
mkdir -p "$OUT_CSV" "$OUT_PNG"

python3 <<'PY'
from pathlib import Path
import os
import sys

import pandas as pd

root = Path(os.environ["ROOT"])
sys.path.insert(0, str(root / "src"))
import plot_ece_calibration as pec

TE = root / "output_wood" / "temperature_experiments"
paths = [
    TE / "fever/csv/conflictqa_ece_openai_fever_temp06_scale10.csv",
    TE / "fever/csv/conflictqa_ece_deepseek_fever_temp06_scale10.csv",
    TE / "conflictqa/csv/conflictqa_ece_openai_temp06_scale10_popqa.csv",
    TE / "conflictqa/csv/conflictqa_ece_deepseek_temp06_scale10_popqa.csv",
    TE / "debateqa/csv/debateqa_ece_openai_deepseek_temp0p6_scale10.csv",
    TE / "qwen_gemma_fever_popqa_debateqa/csv/ece_qwen_gemma_fever_popqa_debateqa_temp0p6_scale10.csv",
]
keep_models = {
    "openai/gpt-4o-2024-11-20",
    "deepseek/deepseek-chat-v2.5",
    "google/gemma-4-26b-a4b-it",
    "qwen/qwen3-14b",
}
keep_ds = {"fever", "debateqa", "conflictqa_popqa"}
parts = []
for p in paths:
    df = pd.read_csv(p)
    df = df[df["model"].isin(keep_models) & df["dataset"].isin(keep_ds)]
    parts.append(df[["model", "dataset", "confidence", "correct"]].copy())
out = pd.concat(parts, ignore_index=True)
out_path = TE / "summary" / "csv" / "ece_four_models_fever_debateqa_popqa_temp06_scale10.csv"
out.to_csv(out_path, index=False)
print(f"Wrote {out_path} (n={len(out)})")

edges = pec.bin_edges()
rows = []
for (m, ds), g in out.groupby(["model", "dataset"]):
    conf = g["confidence"].to_numpy(float)
    cor = g["correct"].to_numpy(float)
    _, acc, mean_c, _counts = pec.compute_bin_stats(conf, cor, edges)
    ece = pec.ece_from_bins(mean_c, acc, _counts)
    rows.append(
        {
            "model": pec._label(m),
            "model_id": m,
            "dataset": ds,
            "n": len(g),
            "ece": round(float(ece), 3),
            "accuracy": round(float(cor.mean()), 3),
            "mean_conf": round(float(conf.mean()), 3),
        }
    )
summary = pd.DataFrame(rows).sort_values(["model", "dataset"])
summary_path = TE / "summary" / "csv" / "ece_four_models_fever_debateqa_popqa_temp06_scale10_summary.csv"
summary.to_csv(summary_path, index=False)
print(summary.to_string(index=False))
print(f"Wrote {summary_path}")
PY

python3 "$ROOT/src/plot_ece_calibration.py" \
  --input "$OUT_CSV/ece_four_models_fever_debateqa_popqa_temp06_scale10.csv" \
  --out-dir "$OUT_PNG" \
  --only-datasets fever debateqa conflictqa_popqa \
  --file-suffix temp06_scale10_four_models

echo "Done. PNG → $OUT_PNG"
