#!/usr/bin/env bash
# Deprecated: the partial-matrix output (Qwen/Gemma on 3 datasets + OpenAI/DeepSeek on DebateQA only)
# duplicated work. All epistemic-marker artifacts now live under:
#   output_wood/epistemic_markers/four_models_fever_debateqa_popqa_temp06_scale10/
# This script delegates to plot_epistemic_four_models_fever_debateqa_popqa.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/plot_epistemic_four_models_fever_debateqa_popqa.sh"
