#!/usr/bin/env bash
# Convert xlsx/pdf submissions and compute inter-annotator agreement.
set -euo pipefail
cd "$(dirname "$0")/../src"
python3 analyze_human_annotator_agreement.py "$@"
