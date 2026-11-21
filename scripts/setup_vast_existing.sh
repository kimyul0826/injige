#!/usr/bin/env bash
# Reuse the template's working PyTorch/CUDA; install only experiment dependencies.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" scripts/check_runtime.py
mkdir -p .cache/pip .tmp
export PIP_CACHE_DIR="$ROOT/.cache/pip" TMPDIR="$ROOT/.tmp" PYTHONDONTWRITEBYTECODE=1
REUSE_ENV="$ROOT/.venv-existing-torch"
if [[ ! -x "$REUSE_ENV/bin/python" ]]; then
    "$PYTHON_BIN" -m venv --system-site-packages "$REUSE_ENV"
fi
"$REUSE_ENV/bin/python" - "$PYTHON_BIN" <<'PY'
import json
from pathlib import Path
import subprocess
import sys
import torch
base = json.loads(subprocess.check_output([sys.argv[1], '-c',
    'import json,torch; print(json.dumps([torch.__version__,torch.__file__]))'], text=True))
if [torch.__version__,torch.__file__] != base:
    raise SystemExit('Existing reuse environment has a different torch; choose a fresh environment')
print('Reusing PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, flush=True)
PY
constraints="$(mktemp "$ROOT/.tmp/existing-torch-XXXXXX.txt")"
requirements="$(mktemp "$ROOT/.tmp/experiment-deps-XXXXXX.txt")"
trap 'rm -f "$constraints" "$requirements"' EXIT
"$REUSE_ENV/bin/python" - "$constraints" "$requirements" <<'PY'
import importlib.metadata
from pathlib import Path
import sys
pins = []
for dist in importlib.metadata.distributions():
    name = dist.metadata['Name']
    if name.lower() in ('torch','triton') or name.lower().startswith('nvidia-'):
        pins.append(f'{name}==={dist.version}')
Path(sys.argv[1]).write_text('\n'.join(sorted(set(pins)))+'\n')
lines = Path('requirements-vast.txt').read_text().splitlines()
Path(sys.argv[2]).write_text('\n'.join(line for line in lines
    if line.strip() and not line.lstrip().startswith(('#', 'torch==')))+'\n')
PY
"$REUSE_ENV/bin/python" -m pip install -c "$constraints" -r "$requirements"
"$REUSE_ENV/bin/python" scripts/check_runtime.py
printf '\nReady. Activate with: source .venv-existing-torch/bin/activate\n'
