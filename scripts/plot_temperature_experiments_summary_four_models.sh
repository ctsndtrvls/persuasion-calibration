#!/usr/bin/env bash
# Summary chart: OpenAI, DeepSeek, Gemma, Qwen × FEVER, DebateQA, PopQA (no StrategyQA)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 "$ROOT/src/plot_temperature_experiments_summary_four_models.py" "$@"
