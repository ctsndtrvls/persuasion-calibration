#!/usr/bin/env bash
# Turn-1 human annotation tables for mixed-100 persuasion rollout.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"
python3 prepare_mixed_human_annotation.py
