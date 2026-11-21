#!/usr/bin/env bash
# Llama-3-8B-Instruct extra experiments on a multi-GPU Vast instance (one model replica per GPU).
#   1) RePE2                                  -> outputs/llama3-repe2-v1.zip
#   2) endpoint candidates (aligned demos)    -> outputs/llama3-endpoint-demos-standard-v1.zip
#   3) E1-b: released demos + round procedure -> outputs/llama3-repe2_origdemos-v1.zip
# Same decoding settings as the Table 1 reproduction (configs/llama3_table1_upstream.json): greedy, 8000 new tokens.
#
# Usage (from the repository root):
#   bash scripts/run_vast_llama_extra.sh
# Options (environment variables):
#   MODEL_PATH=/workspace/Meta-Llama-3-8B-Instruct   GPUS=0,1,2,3   SKIP_SETUP=1   LIMIT=8 (quick check)
# Re-running the script resumes unfinished experiments and keeps finished ones.
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
MODEL_PATH="${MODEL_PATH:-/workspace/Meta-Llama-3-8B-Instruct}"
GPUS="${GPUS:-0,1,2,3}"
LIMIT_ARGS=()
SUFFIX=""
if [[ -n "${LIMIT:-}" ]]; then LIMIT_ARGS=(--limit "$LIMIT"); SUFFIX="-limit$LIMIT"; fi

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }

# --- 0. checks -----------------------------------------------------------------------------------
for f in config.json tokenizer.json; do
    if [[ ! -f "$MODEL_PATH/$f" ]]; then
        echo "Model files not found in $MODEL_PATH (missing $f). Copy the weights first or set MODEL_PATH." >&2
        exit 1
    fi
done

# --- 1. dependencies: reuse the instance's PyTorch/CUDA, install only experiment libraries ----------
if [[ "${SKIP_SETUP:-0}" != "1" ]]; then
    log "Installing experiment dependencies (PyTorch/CUDA from the image are reused)"
    bash scripts/setup_vast_existing.sh || { echo "Dependency setup failed" >&2; exit 1; }
fi
PY="$ROOT/.venv-existing-torch/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false

log "Prompt self-tests (no model loaded)"
"$PY" -B -m unittest tests/test_repe2_prompt.py tests/test_barrier_endpoints.py || { echo "Prompt tests failed" >&2; exit 1; }

# --- 2. experiments ------------------------------------------------------------------------------
declare -a FAILED=()
run() {   # run <output name> <config> <prompt...>
    local out="outputs/$1$SUFFIX" config="$2"; shift 2
    log "Start $out  (config=$config prompts=$*  gpus=$GPUS)"
    if "$PY" scripts/run_multigpu.py --config "$config" --model-path "$MODEL_PATH" --gpus "$GPUS" \
            --prompts "$@" --output "$out" ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} --zip; then
        log "Done  $out.zip"
    else
        log "FAILED $out (re-run this script to resume)"
        FAILED+=("$out")
    fi
}
run llama3-repe2-v1                    configs/llama3_table1_upstream.json        repe2
run llama3-endpoint-demos-standard-v1  configs/llama3_barrier_endpoints_demos.json standard
run llama3-repe2_origdemos-v1          configs/llama3_table1_upstream.json        repe2_origdemos

# --- 3. report -----------------------------------------------------------------------------------
log "Result archives:"
ls -lh outputs/llama3-repe2-v1$SUFFIX.zip outputs/llama3-endpoint-demos-standard-v1$SUFFIX.zip \
       outputs/llama3-repe2_origdemos-v1$SUFFIX.zip 2>/dev/null
if (( ${#FAILED[@]} )); then
    echo "Failed or interrupted: ${FAILED[*]}" >&2
    exit 1
fi
log "All three experiments completed."
