"""python -m llm_astar: single queries and Table 1 experiment matrix."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from .models import Model, ROOT, parse_path, workspace_path
from .search import Environment, path_result, search


def load_queries(path):
    data = json.loads(Path(path).read_text())
    maps = data if isinstance(data,list) else [data]
    for i,env in enumerate(maps):
        if 'start_goal' in env:
            for j,pair in enumerate(env['start_goal']):
                yield f"{env.get('id',i)}:{j}",dict(env,start=pair[0],goal=pair[1])
        else:
            yield str(env.get('id',i)),env


def geometric(values):
    return 100*math.exp(sum(math.log(v) for v in values)/len(values)) if values else None


def summarize(records):
    baselines = {r['sample']:r for r in records if r['method']=='astar'}
    groups = {}
    for row in records:
        key = row['method'],row['model'],row['prompt']
        groups.setdefault(key,[]).append(row)
    summary=[]
    for (method,model,prompt),rows in groups.items():
        ratios={k:[] for k in ('operation','storage','length')}
        for row in rows:
            base=baselines[row['sample']]
            if not row['valid'] or not base['valid']:
                continue
            for metric in ratios:
                a,b=row.get(metric),base.get(metric)
                if a is not None and b is not None and a>0 and b>0:
                    ratios[metric].append(a/b)
        summary.append(dict(method=method,model=model,prompt=prompt,samples=len(rows),
            valid_path_ratio=100*sum(r['valid'] for r in rows)/len(rows),
            operation_ratio=geometric(ratios['operation']),storage_ratio=geometric(ratios['storage']),
            relative_path_length=geometric(ratios['length']),paired_valid_samples=len(ratios['length']),
            parse_failures=sum(bool(r.get('parse_error')) for r in rows),
            truncated=sum(r.get('finish_reason')=='length' for r in rows),
            synthetic=any(r.get('synthetic',False) for r in rows)))
    return summary


def run(args):
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    if args.offset < 0:
        raise ValueError('--offset must not be negative')
    queries=list(load_queries(args.input))[args.offset:]
    if args.limit is not None:
        queries=queries[:args.limit]
    if not queries:
        raise ValueError('No input samples selected')
    configs=[json.loads(Path(p).read_text()) for p in args.config]
    needs_llm=any(m!='astar' for m in args.methods)
    if needs_llm and not configs:
        raise ValueError('--config is required for llm and llm-astar')
    names=[c['name'] for c in configs]
    if len(names)!=len(set(names)):
        raise ValueError('Config names must be unique')
    output=workspace_path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    manifest=dict(input_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        samples=[sid for sid,_ in queries],methods=args.methods,prompts=args.prompts,
        configs=configs,search_mode=args.search_mode,collision_backend="shapely",parser_mode=args.search_mode,version=1)
    manifest_file=output/'manifest.json'
    if manifest_file.exists() and json.loads(manifest_file.read_text())!=manifest:
        raise ValueError('Output directory contains a different experiment; choose a new --output')
    manifest_file.write_text(json.dumps(manifest,indent=2))
    models=[Model(c,args.cache_dir) for c in configs] if needs_llm else []
    records=[]
    # Responses are cached; a restarted run redoes cheap search, not completed LLM calls.
    with (output/'results.jsonl').open('w') as log:
        def save(row):
            records.append(row)
            log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            log.flush()
        for sample,query in queries:
            started=time.perf_counter()
            baseline=search(query,mode=args.search_mode)
            save(dict(baseline,sample=sample,method='astar',model='-',prompt='-',
                      search_seconds=time.perf_counter()-started))
            for model in models:
                for style in args.prompts:
                    response=model.generate(query,style)
                    try:
                        nodes=parse_path(response['text'],mode=args.search_mode)
                        parse_error=None
                    except ValueError as exc:
                        nodes,parse_error=[],str(exc)
                    metadata=dict(sample=sample,model=model.config['name'],prompt=style,
                        cache_key=response['cache_key'],cache_hit=response['cache_hit'],
                        generation_seconds=response['generation_seconds'],usage=response['usage'],
                        finish_reason=response['finish_reason'],parse_error=parse_error,
                        synthetic=response['synthetic'])
                    for method in args.methods:
                        if method=='astar':
                            continue
                        started=time.perf_counter()
                        result=(search(query,nodes,args.search_mode) if method=='llm-astar' else
                                path_result(Environment(query),nodes))
                        save(dict(result,**metadata,method=method,search_seconds=time.perf_counter()-started))
            print(f'Completed {sample}',file=sys.stderr,flush=True)
    summary=summarize(records)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    with (output/'table1.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description='LLM-A* reproduction; A* baseline always included')
    parser.add_argument('--input',default=str(ROOT/'examples/query.json'))
    parser.add_argument('--config',action='append',default=[],help='JSON model config; repeat for model comparison')
    parser.add_argument('--methods',nargs='+',choices=['astar','llm','llm-astar'],default=['astar'])
    parser.add_argument('--prompts',nargs='+',choices=['standard','cot','repe','vot','repe2','repe2_noround','repe2_origdemos'],default=['standard'])
    parser.add_argument('--search-mode',choices=['upstream','paper'],default='upstream')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--offset',type=int,default=0)
    parser.add_argument('--output',default='outputs/run')
    parser.add_argument('--cache-dir',default='outputs/cache')
    args=parser.parse_args()
    try:
        run(args)
    except (ValueError,OSError) as exc:
        parser.exit(2,f'Error: {exc}\n')


if __name__=='__main__':
    main()
