#!/usr/bin/env bash
# Token-prob ECE panels for GPT / DeepSeek / Gemma / Qwen × FEVER / DebateQA / PopQA.
# Uses existing CSVs only (no API calls).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec python3 src/plot_tokenprob_four_models_fever_debateqa_popqa.py
