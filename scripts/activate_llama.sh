# Usage: source scripts/activate_llama.sh
_LLMA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source /home/skkai/anaconda3/etc/profile.d/conda.sh
conda activate "$_LLMA_ROOT/.conda/envs/llm-astar"
export HF_HOME="$_LLMA_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export XDG_CACHE_HOME="$_LLMA_ROOT/.cache"
export TORCH_HOME="$_LLMA_ROOT/.cache/torch"
export TRITON_CACHE_DIR="$_LLMA_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$_LLMA_ROOT/.cache/torchinductor"
export TMPDIR="$_LLMA_ROOT/.tmp"
export PIP_CACHE_DIR="$_LLMA_ROOT/.cache/pip"
export TOKENIZERS_PARALLELISM=false
cd -- "$_LLMA_ROOT"
unset _LLMA_ROOT
