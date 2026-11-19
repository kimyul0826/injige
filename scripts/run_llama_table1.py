"""Launch the experiment and retain logs/status for later inspection. No .deps injection."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import signal
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from llm_astar.models import workspace_path
from llm_astar.artifacts import bundle_results


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2))
    temp.replace(path)


def main():
    def terminate(signum, frame):
        raise KeyboardInterrupt('Received termination signal')
    signal.signal(signal.SIGTERM,terminate)
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='outputs/llama3-table1-upstream-v2')
    parser.add_argument('--config',default='configs/llama3_table1_upstream.json')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--zip',action='store_true',help='Export results and raw responses as ZIP after completion')
    args=parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 1000:
        parser.error('--limit must be between 1 and 1000')
    output=workspace_path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another run is already using this output folder')
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        logfile=output/f'run-{stamp}.log'
        logfile.touch()
        status={'state':'preparing','started_at':timestamp(),'pid':os.getpid(),'log':str(logfile)}
        write_json(output/'status.json',status)
        try:
            verification=ROOT/'models/llama3-verified.json'
            if not verification.exists():
                raise RuntimeError('Weights are not verified. Run scripts/download_llama3.py first.')
            weights=json.loads(verification.read_text())
            config_path=workspace_path(args.config)
            config=json.loads(config_path.read_text())
            if weights['revision'] != config['revision'] or weights['repo_id'] != config['model']:
                raise RuntimeError('Verified model revision does not match the experiment config')
            for item in weights['files']:
                path=ROOT/config['model_path']/item['name']
                if not path.exists() or path.stat().st_size != item['bytes']:
                    raise RuntimeError('Missing or incomplete model file: '+item['name'])
            source_files=list((ROOT/'llm_astar').rglob('*.py'))+[Path(__file__).resolve()]
            hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
            signature=dict(source_sha256=hashes,revision=weights['revision'],config=config)
            previous=output/'experiment-code.json'
            if previous.exists() and json.loads(previous.read_text())!=signature:
                raise RuntimeError('Code/config changed; use a new --output folder')
            write_json(previous,signature)
            env=os.environ.copy()
            env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(ROOT/'.cache/huggingface'),
                       HF_HUB_CACHE=str(ROOT/'.cache/huggingface/hub'),XDG_CACHE_HOME=str(ROOT/'.cache'),
                       TORCH_HOME=str(ROOT/'.cache/torch'),TRITON_CACHE_DIR=str(ROOT/'.cache/triton'),
                       TORCHINDUCTOR_CACHE_DIR=str(ROOT/'.cache/torchinductor'),TMPDIR=str(ROOT/'.tmp'),
                       PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
            command=[sys.executable,'-u','-m','llm_astar','--input','data/environment_50_30.json',
                     '--config',str(config_path),'--methods','astar','llm','llm-astar',
                     '--prompts','standard','cot','repe','--search-mode','upstream',
                     '--output',str(output),'--cache-dir',str(output/'responses')]
            if args.limit:
                command+=['--limit',str(args.limit)]
            status.update(state='running',command=command,expected_samples=args.limit or 1000,
                          expected_rows=(args.limit or 1000)*7)
            write_json(output/'status.json',status)
            write_json(output/f'environment-{stamp}.json',dict(python=sys.version,executable=sys.executable,
                packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
                cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES','all'),weights_revision=weights['revision']))
            with logfile.open('w') as log:
                log.write('Started at '+status['started_at']+'\n');log.flush()
                subprocess.run(['nvidia-smi'],stdout=log,stderr=subprocess.STDOUT,check=False)
                print('Experiment started. Results and logs:',output,flush=True)
                result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            status.update(exit_code=result.returncode,ended_at=timestamp(),
                          state='completed' if result.returncode==0 else 'failed')
            write_json(output/'status.json',status)
            if result.returncode:
                raise SystemExit(f'Inference failed with exit code {result.returncode}; see {logfile}')
            if args.zip:
                print('Result ZIP:',bundle_results(output),flush=True)
            print('Inference completed. Run the offline verification command in README_LLAMA.md.',flush=True)
        except (Exception,KeyboardInterrupt) as exc:
            with logfile.open('a') as log:
                log.write(type(exc).__name__+': '+str(exc)+'\n')
            status.update(state='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                          error=str(exc),ended_at=timestamp())
            write_json(output/'status.json',status)
            raise


if __name__=='__main__':
    main()
