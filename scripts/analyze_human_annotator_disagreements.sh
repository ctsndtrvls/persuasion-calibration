#!/usr/bin/env bash
# Confusion matrices, label skew, and review-case selection for human annotators.
set -euo pipefail
cd "$(dirname "$0")/../src"
python3 analyze_human_annotator_disagreements.py "$@"
