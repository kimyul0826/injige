"""E3 failure check per family: waypoints returned vs turning points of the optimal path (string-pulled),
share of responses with zero surviving waypoints, truncations.
Run: PYTHONDONTWRITEBYTECODE=1 python analysis/e3-bench/failure_by_family.py"""
import sys
sys.dont_write_bytecode = True
import json, statistics
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'exp' / 'claude'))
import offline_improve as o           # noqa: E402
import combine_llm_wallc as cb        # noqa: E402

RUNS = {'standard': ('outputs/gpt4omini-e3-bench-batch-v1', 'standard'),
        'repe2': ('outputs/gpt4omini-e3-bench-batch-v1', 'repe2'),
        'endpoint std': ('outputs/gpt4omini-e3-bench-endpoint-batch-v1', 'standard')}
RESP = {}
for name, (d, p) in RUNS.items():
    for r in map(json.loads, open(ROOT / d / 'results.jsonl')):
        if r['method'] == 'llm-astar' and r['prompt'] == p:
            j = json.loads((ROOT / d / 'responses' / f"{r['cache_key']}.json").read_text())
            RESP[(name, r['sample'])] = (j['text'], j.get('finish_reason'))


def work(args):
    s, q = args
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
    need = len(cb.oracle_chain(env, o.Grid(env))) - 2
    out = {}
    for name in RUNS:
        text, fin = RESP[(name, s)]
        try:
            P = o.parse_path(text, mode='upstream')
        except ValueError:
            P = []
        T = env.targets(P, 'upstream')
        out[name] = (max(len(P) - 2, 0), len(T) - 2, fin == 'length')
    return s.split('-')[0], need, out


if __name__ == '__main__':
    with Pool(16) as pool:
        res = pool.map(work, list(o.load_queries(ROOT / 'data' / 'bench_e3.json')), chunksize=4)
    lines = ['family   need(mean)  | per prompt: waypoints returned (mean) / kept after filter (mean) / zero kept % / truncated']
    for f in ['dense', 'rooms', 'utrap', 'maze', 'scaled2']:
        R = [r for r in res if r[0] == f]
        cells = []
        for name in RUNS:
            ret = [r[2][name][0] for r in R]; kept = [r[2][name][1] for r in R]
            cells.append(f"{name}: {statistics.mean(ret):4.2f}/{statistics.mean(kept):4.2f}/{100*sum(k == 0 for k in kept)/len(R):3.0f}%/{sum(r[2][name][2] for r in R)}")
        lines.append(f"{f:8s} {statistics.mean(r[1] for r in R):5.2f}       | " + '  '.join(cells))
    text = '\n'.join(lines)
    print(text)
    (Path(__file__).parent / 'failure_by_family.txt').write_text(text + '\n')
