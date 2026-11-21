"""Bad cases vs chain detour: chain length (start -> filtered waypoints -> goal) / optimal A* path length.
Run: PYTHONPATH=. python -B outputs/gpt4omini-barrier-endpoints-demos-analysis/chain_detour.py"""
import sys
sys.dont_write_bytecode = True
import json, math
from multiprocessing import Pool
sys.path.insert(0, 'exp/claude')
import offline_improve as o            # noqa: E402
from llm_astar.models import parse_path  # noqa: E402

NEW, OLD = 'outputs/gpt4omini-barrier-endpoints-demos-batch-v1', 'outputs/gpt4omini-table1-upstream-batch-v1'
load = lambda O: {(r['sample'], r['method'], r['prompt']): r for r in map(json.loads, open(O + '/results.jsonl'))}
new, old = load(NEW), load(OLD)


def work(args):
    s, q = args
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
    opt = new[(s, 'astar', '-')]['path_length'] if 'path_length' in new[(s, 'astar', '-')] else None
    if opt is None:
        opt = o.search(env, o.Grid(env), None, dedupe=True)['len']
    out = []
    for O, R in ((OLD, old), (NEW, new)):
        for p in ('standard', 'cot', 'repe'):
            text = json.loads(open(f"{O}/responses/{R[(s, 'llm-astar', p)]['cache_key']}.json").read())['text']
            try:
                P = parse_path(text, 'upstream')
            except Exception:
                P = []
            T = env.targets(P, 'upstream')
            chain = sum(math.dist(a, b) for a, b in zip(T, T[1:]))
            op = R[(s, 'llm-astar', p)]['operation'] / new[(s, 'astar', '-')]['operation']
            out.append((O == NEW, p, chain / opt, op))
    return out


if __name__ == '__main__':
    with Pool(16) as pool:
        rows = [r for rs in pool.map(work, list(o.load_queries(o.DATA)), chunksize=8) for r in rs]
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v)) if v else float('nan')
    for p in ('standard', 'cot', 'repe'):
        for is_new in (False, True):
            R = [r for r in rows if r[0] == is_new and r[1] == p]
            bins = [(0, 1.1), (1.1, 1.3), (1.3, 1.6), (1.6, 99)]
            cells = []
            for lo, hi in bins:
                B = [r[3] for r in R if lo <= r[2] < hi]
                cells.append(f'[{lo},{hi}) n={len(B):3d} op={gm(B):6.1f}')
            print(f"{p:8s} {'new' if is_new else 'old'}  " + '  '.join(cells))
