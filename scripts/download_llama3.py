"""Download pinned official weights inside this project; no inference."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_XET_CACHE'] = str(ROOT/'.cache/huggingface/xet')
from huggingface_hub import get_token, snapshot_download

local_token = ROOT/'.cache/huggingface/token'
token = os.environ.get('HF_TOKEN') or (local_token.read_text().strip() if local_token.exists() else get_token())

manifest = json.loads((ROOT/'models/llama3-download-manifest.json').read_text())
folder = ROOT/'models/Meta-Llama-3-8B-Instruct'
files = [f for f in manifest['files'] if not f['name'].startswith('original/')]
snapshot_download(repo_id=manifest['repo_id'], revision=manifest['revision'],
                  token=token, local_dir=folder,
                  cache_dir=ROOT/'.cache/huggingface/hub',
                  allow_patterns=[f['name'] for f in files], max_workers=4)
checks = []
for record in files:
    path = folder/record['name']
    if record['size'] is not None and path.stat().st_size != record['size']:
        raise RuntimeError('Size mismatch: '+record['name'])
    with path.open('rb') as file:
        digest = hashlib.file_digest(file,'sha256').hexdigest()
    if record['sha256'] and digest != record['sha256']:
        raise RuntimeError('SHA256 mismatch: '+record['name'])
    checks.append(dict(name=record['name'],bytes=path.stat().st_size,sha256=digest))
(ROOT/'models/llama3-verified.json').write_text(json.dumps(dict(
    repo_id=manifest['repo_id'],revision=manifest['revision'],files=checks),indent=2))
print('Download and SHA256 verification complete:', folder, flush=True)
