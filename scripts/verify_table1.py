"""Offline validation only: never loads a model or calls an API."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from llm_astar.__main__ import load_queries,summarize
from llm_astar.models import workspace_path
from llm_astar.search import Environment,path_result

# Table 1, Llama 3 rows. Metrics are percentages in the order shown below.
PAPER={
 ('llm','standard'):(None,None,111.05,12.60),
 ('llm','cot'):(None,None,114.89,12.00),
 ('llm','repe'):(None,None,138.32,16.40),
 ('llm-astar','standard'):(44.59,64.02,102.47,100),
 ('llm-astar','cot'):(47.60,66.27,102.46,100),
 ('llm-astar','repe'):(64.08,80.19,102.54,100),
}
METRICS=('operation_ratio','storage_ratio','relative_path_length','valid_path_ratio')


def verify(output):
    manifest=json.loads((output/'manifest.json').read_text())
    records=[json.loads(line) for line in (output/'results.jsonl').read_text().splitlines() if line.strip()]
    dataset=ROOT/'data/environment_50_30.json'
    errors=[]
    status_file=output/'status.json'
    if status_file.exists() and json.loads(status_file.read_text()).get('state')!='completed':
        errors.append('Run status is not completed')
    if hashlib.sha256(dataset.read_bytes()).hexdigest()!=manifest['input_sha256']:
        errors.append('Input dataset hash mismatch')
    queries=dict(load_queries(dataset))
    expected={(s,'astar','-','-') for s in manifest['samples']}
    expected.update((s,m,c['name'],p) for s in manifest['samples'] for c in manifest['configs']
                    for p in manifest['prompts'] for m in manifest['methods'] if m!='astar')
    keys=[(r['sample'],r['method'],r['model'],r['prompt']) for r in records]
    if len(keys)!=len(set(keys)): errors.append('Duplicate result rows')
    if set(keys)!=expected: errors.append(f'Missing {len(expected-set(keys))}; unexpected {len(set(keys)-expected)} rows')
    if any(r.get('synthetic') for r in records): errors.append('Synthetic predictions found')
    checked=set()
    for row in records:
        if row['sample'] not in queries:
            errors.append('Unknown sample: '+row['sample']);continue
        env=Environment(queries[row['sample']],collision_backend=manifest.get('collision_backend','orientation'))
        path=[tuple(p) for p in row['path']]
        actual=path_result(env,path)
        if actual['valid']!=row['valid']:
            errors.append('Path validity mismatch: '+str((row['sample'],row['method'],row['prompt'])))
        if actual['valid'] and not math.isclose(actual['length'],row['length'],rel_tol=1e-10,abs_tol=1e-10):
            errors.append('Path length mismatch: '+row['sample'])
        if row['method']!='astar':
            key=row['cache_key']
            if key in checked: continue
            checked.add(key)
            cache=output/'responses'/f'{key}.json'
            if not cache.exists():
                errors.append('Missing response: '+key);continue
            response=json.loads(cache.read_text())
            if response.get('synthetic') or response['config']['provider']!='transformers':
                errors.append('Response is not real local inference: '+key)
            expected_key=hashlib.sha256(json.dumps(dict(config=response['config'],prompt=response['prompt']),sort_keys=True).encode()).hexdigest()
            if expected_key!=key: errors.append('Response fingerprint mismatch: '+key)
            if response['config']['model']!='meta-llama/Meta-Llama-3-8B-Instruct':
                errors.append('Wrong model: '+key)
    summary=summarize(records)
    stored=json.loads((output/'summary.json').read_text())
    if summary!=stored: errors.append('Stored summary differs from recomputed summary')
    comparisons=[]
    for row in summary:
        reference=PAPER.get((row['method'],row['prompt']))
        if reference is None: continue
        comparisons.append(dict(method=row['method'],prompt=row['prompt'],samples=row['samples'],
            **{metric:dict(observed=row[metric],paper=value,
                difference_percentage_points=row[metric]-value if row[metric] is not None and value is not None else None)
               for metric,value in zip(METRICS,reference)}))
    report=dict(passed=not errors,errors=errors,samples=len(manifest['samples']),
                expected_rows=len(expected),actual_rows=len(records),unique_responses=len(checked),
                full_table1_dataset=len(manifest['samples'])==1000 and set(manifest['samples'])==set(queries),
                source='https://arxiv.org/html/2407.02511v1',comparisons=comparisons,
                note='Checks artifact consistency, not equality with paper numbers. Differences require analysis.')
    (output/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',nargs='?',default='outputs/llama3-table1')
    args=parser.parse_args()
    report=verify(workspace_path(args.output))
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':
    main()
