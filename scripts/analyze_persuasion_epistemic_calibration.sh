#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"

CSV="$ROOT/output_wood/persuasion/DeepSeek/fever/csv/expl.csv"
python3 analyze_persuasion_epistemic_calibration.py --input "$CSV"
python3 analyze_persuasion_confidence.py --input "$CSV"
python3 plot_persuasion_uncertainty_overview.py
