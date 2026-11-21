"""CPU/API Table 1 extension: fixed prompts, cached responses, status and optional ZIP."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_astar.__main__ import load_queries, run as run_experiment
from llm_astar.models import workspace_path
from llm_astar.artifacts import write_json, bundle_results


def now():
    return datetime.now(timezone.utc).isoformat()


def run(args):
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    if config['provider'] not in ('openai', 'fixture'):
        raise ValueError('Use an OpenAI API or offline fixture config')
    queries = list(load_queries(args.input))
    if args.limit is not None:
        if not 1 <= args.limit <= len(queries):
            raise ValueError('limit must be between 1 and the input sample count')
        queries = queries[:args.limit]
    if not queries or len({sid for sid, _ in queries}) != len(queries):
        raise ValueError('Input must contain unique nonempty samples')
    if not args.prompts or len(set(args.prompts)) != len(args.prompts):
        raise ValueError('Prompts must be nonempty and unique')
    plan = dict(model=config['model'] if 'model' in config else config['name'],
                samples=len(queries), prompts=args.prompts,
                generation_jobs=len(queries)*len(args.prompts),
                expected_rows=len(queries)*(1+2*len(args.prompts)),
                max_output_tokens_per_call=config['max_tokens'],
                output=args.output)
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return plan
    if config['provider'] == 'openai' and not os.environ.get(config.get('api_key_env', 'OPENAI_API_KEY')):
        raise ValueError(f"Set {config.get('api_key_env', 'OPENAI_API_KEY')} in your terminal; do not put the key in config files")
    output = workspace_path(args.output)
    if output == ROOT:
        raise ValueError('Use a dedicated experiment output folder')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another run is using this output folder')
        files = [*sorted((ROOT/'llm_astar').rglob('*.py')), Path(__file__).resolve()]
        signature = dict(config=config, source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
        code_file = output/'experiment-code.json'
        if code_file.exists() and json.loads(code_file.read_text()) != signature:
            raise ValueError('Code/config changed; use a new output folder')
        write_json(code_file, signature)
        write_json(output/'input.json', json.loads(Path(args.input).read_text()))
        packages = {}
        for name in ('openai', 'shapely'):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        write_json(output/'runtime.json', dict(python=sys.version, packages=packages))
        status = dict(plan, state='running', started_at=now(), pid=os.getpid())
        write_json(output/'status.json', status)
        try:
            run_experiment(argparse.Namespace(input=args.input, config=[str(config_path)],
                methods=['astar', 'llm', 'llm-astar'], prompts=args.prompts,
                search_mode='upstream', limit=args.limit, offset=0,
                output=str(output), cache_dir=str(output/'responses')))
            rows = [json.loads(line) for line in (output/'results.jsonl').read_text().splitlines()]
            if len(rows) != plan['expected_rows']:
                raise RuntimeError('Result row count does not match the experiment plan')
            status.update(state='completed', rows=len(rows))
        except BaseException as exc:
            status.update(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                          error_type=type(exc).__name__)
            raise
        finally:
            status['ended_at'] = now()
            status['cached_responses'] = len(list((output/'responses').glob('*.json')))
            write_json(output/'status.json', status)
            if args.zip:
                print('Result ZIP:', bundle_results(output), flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'configs/gpt4omini_table1_upstream.json'))
    parser.add_argument('--input', default=str(ROOT/'data/environment_50_30.json'))
    parser.add_argument('--output', default='outputs/gpt4omini-table1-upstream-v1')
    parser.add_argument('--prompts', nargs='+', choices=['standard','cot','repe','vot','repe2','repe2_noround','repe2_origdemos'], default=['standard','cot','repe'])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--zip', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Show job counts without API calls or output writes')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        run(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
