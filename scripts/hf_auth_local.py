"""Validate and save an HF token inside this project only. Never print the token."""
import getpass
import os
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
ROOT=Path(__file__).resolve().parents[1]
token=getpass.getpass('Hugging Face token (hidden input): ').strip()
HfApi(token=token).whoami()
hf_hub_download('meta-llama/Meta-Llama-3-8B-Instruct','config.json',token=token,
                cache_dir=ROOT/'.cache/huggingface/hub')
path=ROOT/'.cache/huggingface/token'
path.parent.mkdir(parents=True,exist_ok=True)
with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600),'w') as f:
    f.write(token)
os.chmod(path,0o600)
print('Model access verified; token saved inside project cache.')
