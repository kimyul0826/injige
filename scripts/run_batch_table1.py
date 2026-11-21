"""Submit remaining Table 1 requests to Batch, then collect and evaluate cached outputs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_astar.__main__ import load_queries, run as evaluate
from llm_astar.artifacts import write_json, bundle_results
from llm_astar.models import Model, build_prompt, chat_messages, workspace_path

TERMINAL = {'completed', 'failed', 'expired', 'cancelled'}


def jobs_for(config, queries, prompts):
    config = Model(config).config
    jobs = {}
    for sample, query in queries:
        for style in prompts:
            prompt = build_prompt(query, style, config['prompt_family'], config.get('prompt_variant', 'upstream'))
            key = hashlib.sha256(json.dumps(dict(config=config, prompt=prompt), sort_keys=True).encode()).hexdigest()
            jobs[key] = dict(config=config, prompt=prompt, sample=sample, style=style)
    return jobs


def import_outputs(text, jobs, cache, batch_id):
    for line in text.splitlines():
        row = json.loads(line)
        key = row['custom_id']
        if key not in jobs:
            raise ValueError('Batch returned an unknown request ID')
        response = row.get('response')
        if row.get('error') or not response or response['status_code'] != 200:
            continue
        body = response['body']
        choice = body['choices'][0]
        if (cache/(key+'.json')).exists():
            continue
        job = jobs[key]
        write_json(cache/(key+'.json'), dict(text=choice['message'].get('content') or '',
            usage=body.get('usage'), finish_reason=choice['finish_reason'], returned_model=body['model'],
            config=job['config'], prompt=job['prompt'], cache_key=key, cache_hit=False,
            generation_seconds=None, timing_source='batch_per_request_latency_unavailable',
            synthetic=False, batch_id=batch_id, request_id=response.get('request_id')))


def run(args, client=None):
    config = json.loads(Path(args.config).read_text())
    if config.get('provider') != 'openai' or config.get('completion_api'):
        raise ValueError('Batch runner requires an OpenAI chat-completions config')
    queries = list(load_queries(args.input))
    if args.limit is not None:
        if not 1 <= args.limit <= len(queries):
            raise ValueError('Invalid --limit')
        queries = queries[:args.limit]
    jobs = jobs_for(config, queries, args.prompts)
    output = workspace_path(args.output)
    if output == ROOT:
        raise ValueError('Use a dedicated output folder')
    cache = output/'responses'
    source = workspace_path(args.reuse_cache)
    if source == cache:
        raise ValueError('Reuse cache must differ from the Batch output cache')
    plan = dict(model=config['model'], samples=len(queries), generation_jobs=len(queries)*len(args.prompts),
                unique_requests=len(jobs), expected_rows=len(queries)*(1+2*len(args.prompts)))
    if args.dry_run:
        plan['already_cached'] = sum((cache/(k+'.json')).exists() or (source/(k+'.json')).exists() for k in jobs)
        plan['remaining_requests'] = len(jobs)-plan['already_cached']
        print(json.dumps(plan, indent=2), flush=True)
        return plan
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signature = dict(config=config, input_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
            samples=[s for s,q in queries], prompts=args.prompts,
            source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [*sorted((ROOT/'llm_astar').rglob('*.py')), Path(__file__).resolve()]})
        sf = output/'experiment-code.json'
        if sf.exists() and json.loads(sf.read_text()) != signature:
            raise ValueError('Code/config/input changed; choose a new output folder')
        write_json(sf, signature)
        cache.mkdir(exist_ok=True)
        reused = 0
        for key, job in jobs.items():
            dst = cache/(key+'.json')
            src = source/dst.name
            if not dst.exists() and src.exists():
                saved = json.loads(src.read_text())
                if saved['config'] != job['config'] or saved['prompt'] != job['prompt'] or saved['cache_key'] != key:
                    raise ValueError('Cached response does not match experiment')
                shutil.copyfile(src, dst)
                reused += 1
        batch_dir = output/'logs'/'batch'
        batch_dir.mkdir(parents=True, exist_ok=True)
        state_file = batch_dir/'state.json'
        state = json.loads(state_file.read_text()) if state_file.exists() else {'attempts': [], 'reused_sync_responses': 0}
        state['reused_sync_responses'] += reused
        write_json(state_file, state)
        pending = [k for k in jobs if not (cache/(k+'.json')).exists()]
        print(json.dumps(dict(plan, cached=len(jobs)-len(pending), remaining_requests=len(pending)), indent=2), flush=True)
        if pending:
            if client is None:
                if not os.environ.get(config.get('api_key_env', 'OPENAI_API_KEY')):
                    raise ValueError('Set OPENAI_API_KEY in the terminal')
                from openai import OpenAI
                client = OpenAI(api_key=os.environ[config.get('api_key_env', 'OPENAI_API_KEY')],
                                base_url=config.get('base_url'), max_retries=0, timeout=120)
            last = state['attempts'][-1] if state['attempts'] else None
            if last and last.get('creation_uncertain'):
                raise ValueError('Submission was interrupted before saving its ID. Check the OpenAI Batch dashboard before resubmitting to avoid duplicate requests.')
            if last and last.get('collected') and not args.retry_failed:
                raise ValueError('Some requests failed; inspect logs/batch, then use --retry-failed to submit only missing responses')
            if not last or last.get('collected'):
                number = len(state['attempts'])+1
                request_file = batch_dir/f'requests-{number}.jsonl'
                with request_file.open('w') as stream:
                    for key in pending:
                        job = jobs[key]
                        body = dict(model=config['model'], temperature=job['config']['temperature'],
                            max_tokens=job['config']['max_tokens'], messages=chat_messages(job['prompt'], job['config']['prompt_family']))
                        stream.write(json.dumps(dict(custom_id=key, method='POST', url='/v1/chat/completions', body=body))+'\n')
                with request_file.open('rb') as stream:
                    uploaded = client.files.create(file=stream, purpose='batch')
                last = dict(input_file_id=uploaded.id, creation_uncertain=True, requests=len(pending))
                state['attempts'].append(last)
                write_json(state_file, state)
                batch = client.batches.create(input_file_id=uploaded.id, endpoint='/v1/chat/completions', completion_window='24h')
                last.update(batch_id=batch.id, creation_uncertain=False)
                write_json(state_file, state)
                print('Submitted Batch:', batch.id, flush=True)
            while True:
                batch = client.batches.retrieve(last['batch_id'])
                write_json(batch_dir/(batch.id+'.json'), batch.model_dump())
                write_json(output/'status.json', dict(plan, state=batch.status, batch_id=batch.id,
                    reused_sync_responses=state['reused_sync_responses']))
                print('Batch:', batch.id, batch.status, flush=True)
                if batch.status in TERMINAL:
                    break
                if args.submit_only:
                    print('Run the same command without --submit-only later to collect results.', flush=True)
                    return {'state': batch.status, 'batch_id': batch.id}
                time.sleep(20)
            for field in ['output_file_id', 'error_file_id']:
                file_id = getattr(batch, field, None)
                if file_id:
                    text = client.files.content(file_id).text
                    (batch_dir/(batch.id+'-'+field+'.jsonl')).write_text(text)
                    import_outputs(text, jobs, cache, batch.id)
            last['collected'] = True
            write_json(state_file, state)
            missing = [k for k in jobs if not (cache/(k+'.json')).exists()]
            if missing:
                raise ValueError(f'{len(missing)} requests remain; inspect logs/batch. --retry-failed submits only missing requests.')
        # Never fall back to regular paid requests while evaluating the Batch cache.
        with patch.object(Model, '_api', side_effect=RuntimeError('Missing Batch cache: regular API fallback disabled')):
            evaluate(argparse.Namespace(input=args.input, config=[args.config], output=str(output),
                cache_dir=str(cache), limit=args.limit, offset=0, methods=['astar','llm','llm-astar'],
                prompts=args.prompts, search_mode='upstream'))
        shutil.copyfile(args.input, output/'input.json')
        write_json(output/'status.json', dict(plan, state='completed',
            reused_sync_responses=state['reused_sync_responses'], batch_attempts=len(state['attempts'])))
        if args.zip:
            print('Result ZIP:', bundle_results(output), flush=True)
        return {'state': 'completed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'configs/gpt4omini_table1_upstream.json'))
    parser.add_argument('--input', default=str(ROOT/'data/environment_50_30.json'))
    parser.add_argument('--output', default='outputs/gpt4omini-table1-upstream-batch-v1')
    parser.add_argument('--reuse-cache', default='outputs/gpt4omini-table1-upstream-v1/responses')
    parser.add_argument('--prompts', nargs='+', choices=['standard','cot','repe','vot','repe2','repe2_noround','repe2_origdemos'], default=['standard','cot','repe'])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--submit-only', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--zip', action='store_true')
    try:
        run(parser.parse_args())
    except KeyboardInterrupt:
        print('\nLocal waiting stopped. Submitted Batches continue remotely; rerun the same command to collect.', file=sys.stderr)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
