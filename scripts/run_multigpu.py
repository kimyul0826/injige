"""One model replica per GPU, dynamically scheduled inputs, durable resume and ZIP export."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_astar.artifacts import write_json, bundle_results
from llm_astar.models import Model, parse_path, workspace_path
from llm_astar.search import Environment, path_result, search
from llm_astar.__main__ import load_queries, summarize


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def job_id(sample, style):
    return hashlib.sha256(json.dumps([sample, style]).encode()).hexdigest()


def model_identity(config):
    if config['provider'] == 'fixture':
        return {'fixture_sha256': digest(ROOT/config['response_file'])}
    folder = Path(config['model_path'])
    if not folder.is_dir():
        raise ValueError(f'Model directory not found: {folder}')
    if not (folder/'config.json').is_file() or not (folder/'tokenizer.json').is_file():
        raise ValueError('Expected a Hugging Face model directory with config.json and tokenizer.json')
    index = folder/'model.safetensors.index.json'
    names = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
    for name in names:
        path = (folder/name).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing/invalid weight shard: {name}')
    # This catches incomplete copies; it is explicitly NOT a full cryptographic weight verification.
    metadata = {p.name: digest(p) for p in folder.glob('*.json') if p.is_file()}
    return {'path': str(folder), 'metadata_sha256': metadata,
            'shard_bytes': {n: (folder/n).stat().st_size for n in names},
            'weight_content_verified': False}


def devices_for(config, requested):
    if config['provider'] == 'fixture':
        return requested.split(',') if requested != 'all' else ['0', '1']
    probe = subprocess.run([sys.executable, '-c',
        'import torch,json; print(json.dumps({"count":torch.cuda.device_count(),'
        '"names":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))'],
        capture_output=True, text=True, check=True)
    info = json.loads(probe.stdout)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    available = visible.split(',') if visible else [str(i) for i in range(info['count'])]
    if not info['count'] or len(available) != info['count']:
        raise ValueError('No CUDA GPUs available, or CUDA_VISIBLE_DEVICES is invalid')
    devices = available if requested == 'all' else requested.split(',')
    if not devices or len(devices) != len(set(devices)) or any(d not in available for d in devices):
        raise ValueError(f'--gpus must be distinct IDs from {available}')
    return devices


def worker(device, config, tasks, events, output):
    # Spawn ensures CUDA is initialized only AFTER visibility is set, in this process.
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    os.environ.update(TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1')
    output = Path(output)
    log_path = output/'logs'/f'worker-{device.replace("/", "_")}.log'
    with log_path.open('a', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        current = None
        try:
            print(f'{now()} started pid={os.getpid()} visible_gpu={device}', flush=True)
            model = Model(config, str(output/'responses'))
            if config['provider'] == 'transformers':
                import torch
                torch.set_num_threads(1)
                if torch.cuda.device_count() != 1:
                    raise RuntimeError('Worker must see exactly one GPU')
                print(f'GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}; CUDA={torch.version.cuda}', flush=True)
            while True:
                current = tasks.get()
                if current is None:
                    break
                sample, style, query = current
                print(f'{now()} generating {sample} {style}', flush=True)
                response = model.generate(query, style)
                try:
                    nodes, error = parse_path(response['text'], mode='upstream'), None
                except ValueError as exc:
                    nodes, error = [], str(exc)
                common = dict(sample=sample, model=config['name'], prompt=style,
                    cache_key=response['cache_key'], cache_hit=response['cache_hit'],
                    generation_seconds=response['generation_seconds'], usage=response['usage'],
                    finish_reason=response['finish_reason'], parse_error=error,
                    synthetic=response['synthetic'], worker_gpu=device)
                records = []
                for method in ('llm', 'llm-astar'):
                    started = time.perf_counter()
                    result = search(query, nodes, 'upstream') if method == 'llm-astar' else path_result(Environment(query), nodes)
                    records.append(dict(result, **common, method=method, search_seconds=time.perf_counter()-started))
                key = job_id(sample, style)
                write_json(output/'jobs'/f'{key}.json', dict(sample=sample, prompt=style, records=records))
                events.put(('done', key, device))
                print(f'{now()} completed {sample} {style}', flush=True)
            print(f'{now()} worker finished', flush=True)
        except BaseException:
            traceback.print_exc()
            events.put(('error', repr(current[:2] if current else None), device))
            raise


def completed_jobs(output, jobs, config):
    done = set()
    for sample, style, _ in jobs:
        key = job_id(sample, style)
        path = output/'jobs'/f'{key}.json'
        if not path.exists():
            continue
        item = json.loads(path.read_text())
        rows = item.get('records', [])
        if item.get('sample') != sample or item.get('prompt') != style or len(rows) != 2:
            raise ValueError(f'Invalid completed job: {path}')
        if {r['method'] for r in rows} != {'llm', 'llm-astar'}:
            raise ValueError(f'Invalid methods in {path}')
        for row in rows:
            if (row['sample'], row['prompt'], row['model']) != (sample, style, config['name']):
                raise ValueError(f'Mismatched result in {path}')
            response = output/'responses'/f"{row['cache_key']}.json"
            if not response.is_file():
                raise ValueError(f'Missing saved raw response: {response}')
        done.add(key)
    return done


def merge(output, queries, jobs):
    """Also safe after interruption: summaries explicitly count only completed records."""
    baselines = {r['sample']: r for r in json.loads((output/'baselines.json').read_text())}
    records = []
    for sample, _ in queries:
        records.append(baselines[sample])
        for sid, style, _ in jobs:
            if sid != sample:
                continue
            path = output/'jobs'/f'{job_id(sid, style)}.json'
            if path.is_file():
                records.extend(json.loads(path.read_text())['records'])
    temporary = output/'results.jsonl.tmp'
    with temporary.open('w') as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(output/'results.jsonl')
    summary = summarize(records)
    write_json(output/'summary.json', summary)
    import csv
    with (output/'table1.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return len(records)


def run(args):
    config = json.loads(Path(args.config).read_text())
    if config['provider'] not in ('transformers', 'fixture'):
        raise ValueError('This runner is for local GPU replicas, not remote API models')
    if config['provider'] == 'transformers':
        path = Path(args.model_path or config.get('model_path', 'models/Meta-Llama-3-8B-Instruct')).expanduser()
        config.update(model_path=str((path if path.is_absolute() else ROOT/path).resolve()),
                      device_map={'': 'cuda:0'}, require_cuda=True)
    queries = list(load_queries(args.input))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('--limit must be positive')
        queries = queries[:args.limit]
    if not queries or len({sid for sid, _ in queries}) != len(queries):
        raise ValueError('Input must contain unique, nonempty sample IDs')
    if len(args.prompts) != len(set(args.prompts)):
        raise ValueError('Prompts must be unique')
    output = workspace_path(args.output)
    if output == ROOT or output in (ROOT/'models', ROOT/'llm_astar'):
        raise ValueError('Use a dedicated experiment output directory')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another launcher is using this output directory')
        for folder in ('jobs', 'responses', 'logs'):
            (output/folder).mkdir(exist_ok=True)
        packages = {}
        for name in ('torch', 'transformers', 'accelerate', 'shapely', 'huggingface-hub'):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        signature = dict(config=config, model_identity=model_identity(config), packages=packages,
            input_sha256=digest(args.input), samples=[sid for sid, _ in queries], prompts=args.prompts,
            source_sha256={str(p.relative_to(ROOT)): digest(p) for p in
                [*sorted((ROOT/'llm_astar').rglob('*.py')), Path(__file__).resolve()]},
            search_mode='upstream', collision_backend='shapely', version=1)
        manifest = output/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()) != signature:
            raise ValueError('Input/model/code/packages changed; choose a new --output directory')
        write_json(manifest, signature)
        write_json(output/'input.json', json.loads(Path(args.input).read_text()))
        jobs = [(sid, style, query) for sid, query in queries for style in args.prompts]
        done = completed_jobs(output, jobs, config)
        pending = [job for job in jobs if job_id(job[0], job[1]) not in done]
        status = dict(state='preparing', started_at=now(), pid=os.getpid(),
                      expected_jobs=len(jobs), completed_jobs=len(done), expected_rows=len(queries)+2*len(jobs))
        write_json(output/'status.json', status)
        processes = []
        ctx = mp.get_context('spawn')
        tasks, events = ctx.Queue(), ctx.Queue()
        failure = None
        try:
            if not (output/'baselines.json').exists():
                baselines = []
                for sample, query in queries:
                    started = time.perf_counter()
                    r = search(query, mode='upstream')
                    baselines.append(dict(r, sample=sample, method='astar', model='-', prompt='-',
                                          search_seconds=time.perf_counter()-started))
                write_json(output/'baselines.json', baselines)
            devices = devices_for(config, args.gpus) if pending else []
            if len(devices) != len(set(devices)):
                raise ValueError('GPU IDs must be unique')
            devices = devices[:len(pending)]
            write_json(output/'runtime.json', dict(python=sys.version, packages=packages,
                devices=devices, started_at=status['started_at'], resumed_jobs=len(done)))
            for job in pending:
                tasks.put(job)
            for device in devices:
                tasks.put(None)
            for device in devices:
                process = ctx.Process(target=worker, args=(device, config, tasks, events, str(output)))
                process.start()
                processes.append(process)
            status.update(state='running', devices=devices)
            write_json(output/'status.json', status)
            print(f'{len(done)}/{len(jobs)} jobs cached; starting {len(devices)} GPU workers. Output: {output}', flush=True)
            while len(done) < len(jobs):
                try:
                    kind, key, device = events.get(timeout=1)
                    if kind == 'error':
                        raise RuntimeError(f'GPU {device} failed at {key}; see logs/worker-{device}.log')
                    done.add(key)
                    status['completed_jobs'] = len(done)
                    write_json(output/'status.json', status)
                    print(f'Completed {len(done)}/{len(jobs)} (GPU {device})', flush=True)
                except queue.Empty:
                    if any(p.exitcode not in (None, 0) for p in processes):
                        raise RuntimeError('A worker exited unexpectedly; see worker logs')
                    if all(p.exitcode is not None for p in processes):
                        raise RuntimeError('Workers finished without all expected completion records')
            for process in processes:
                process.join(timeout=10)
                if process.is_alive() or process.exitcode != 0:
                    raise RuntimeError('Worker did not shut down cleanly')
            status['state'] = 'completed'
        except BaseException as exc:
            failure = exc
            status.update(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join()
            tasks.cancel_join_thread()
            tasks.close()
            events.close()
            # A worker may have committed a job immediately before interruption.
            status['completed_jobs'] = len(completed_jobs(output, jobs, config))
            if (output/'baselines.json').exists():
                status['rows'] = merge(output, queries, jobs)
            status['ended_at'] = now()
            write_json(output/'status.json', status)
            if args.zip:
                print('Result ZIP:', bundle_results(output), flush=True)
        if failure:
            raise failure
        print('Experiment completed.', flush=True)
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'configs/llama3_table1_upstream.json'))
    parser.add_argument('--model-path', help='Read-only HF checkpoint directory; may be outside this repository')
    parser.add_argument('--input', default=str(ROOT/'data/environment_50_30.json'))
    parser.add_argument('--output', default='outputs/llama3-multigpu')
    parser.add_argument('--gpus', default='all', help='Visible GPU IDs, e.g. 0,1,2,3; default: all visible GPUs')
    parser.add_argument('--prompts', nargs='+', choices=['standard','cot','repe'], default=['standard','cot','repe'])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--zip', action='store_true', help='Export result ZIP on completion or handled interruption/failure')
    parser.add_argument('--zip-only', action='store_true', help='Export existing output without running inference')
    args = parser.parse_args()
    if args.zip_only:
        output = workspace_path(args.output)
        if not (output/'status.json').is_file():
            parser.error('No saved experiment status in output directory')
        with (output/'.run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(bundle_results(output))
        return
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        run(args)
    except (ValueError, RuntimeError, OSError, KeyboardInterrupt) as exc:
        parser.exit(130 if isinstance(exc, KeyboardInterrupt) else 1, f'{type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    main()
