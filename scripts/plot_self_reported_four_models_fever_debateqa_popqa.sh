#!/usr/bin/env bash
# Combined bar chart: OpenAI, DeepSeek, Qwen, Gemma × (FEVER, DebateQA, PopQA).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec python3 src/plot_self_reported_four_models_fever_debateqa_popqa.py "$@"
