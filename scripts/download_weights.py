"""Optional pinned HF download; prompts for a token without saving credentials."""
import argparse
import getpass
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_astar.models import workspace_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='models/Meta-Llama-3-8B-Instruct')
    args = parser.parse_args()
    destination = workspace_path(args.output)
    os.environ['HF_HOME'] = str(ROOT/'.cache/huggingface')
    os.environ['HF_XET_CACHE'] = str(ROOT/'.cache/huggingface/xet')
    # No hf auth login / token-file creation. Environment token is not printed or persisted.
    token = os.environ.get('HF_TOKEN') or getpass.getpass('Hugging Face read token (not saved): ')
    if not token:
        raise SystemExit('An approved Meta Llama 3 access token is required')
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
        revision='8afb486c1db24fe5011ec46dfbe5b5dccdb575c2', token=token,
        local_dir=destination, allow_patterns=['*.json', '*.safetensors', 'tokenizer.model', 'LICENSE', 'USE_POLICY.md'],
        max_workers=4)
    print('Weights downloaded:', destination)


if __name__ == '__main__':
    main()
