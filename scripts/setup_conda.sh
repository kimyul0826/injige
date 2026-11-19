#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
mkdir -p .tmp .cache/pip .conda outputs/setup
export CONDARC="$ROOT/.conda/condarc"
export CONDA_REGISTER_ENVS=false CONDA_NO_PLUGINS=true
export XDG_CACHE_HOME="$ROOT/.cache" TMPDIR="$ROOT/.tmp" PIP_CACHE_DIR="$ROOT/.cache/pip"
if [[ ! -x "$ROOT/.conda/envs/llm-astar/bin/python" ]]; then
    conda create --prefix "$ROOT/.conda/envs/llm-astar" --override-channels -c conda-forge python=3.12 pip -y
fi
"$ROOT/.conda/envs/llm-astar/bin/python" -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
"$ROOT/.conda/envs/llm-astar/bin/python" -m pip install -r requirements-llama-conda.txt
