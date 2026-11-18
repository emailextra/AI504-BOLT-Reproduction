#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python code/analyze_results.py
