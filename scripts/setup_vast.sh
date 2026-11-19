#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,13), "Use Python 3.11–3.13 (recommended: 3.12); set PYTHON_BIN=python3.12"'
mkdir -p .cache/pip .tmp
export PIP_CACHE_DIR="$ROOT/.cache/pip" TMPDIR="$ROOT/.tmp" PYTHONDONTWRITEBYTECODE=1
if [[ ! -x .venv/bin/python ]]; then
    "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements-vast.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/check_runtime.py
printf '\nReady. Activate with: source .venv/bin/activate\n'
