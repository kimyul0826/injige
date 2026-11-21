"""Are bad cases explained by barrier-end cells that were dropped from the candidate list?
Reasons a cell just past a barrier end is dropped: 'border' (outside the official filter band
range+1 < x < range-2, incl. off-map) or 'on_barrier' (lies on another barrier).
'needed' = the optimal path's string-pulled turning points (oracle chain) pass within 1 cell of a dropped cell.
Run: PYTHONPATH=. python -B outputs/gpt4omini-barrier-endpoints-demos-analysis/dropped_endpoints.py"""
import sys
sys.dont_write_bytecode = True
import json, math
from collections import defaultdict
from multiprocessing import Pool
sys.path.insert(0, 'exp/claude')
import offline_improve as o            # noqa: E402
import combine_llm_wallc as cb         # noqa: E402

NEW, OLD = 'outputs/gpt4omini-barrier-endpoints-demos-batch-v1', 'outputs/gpt4omini-table1-upstream-batch-v1'


def raw_cells(q):
    segs, cells = [], []
    for y, a, b in q.get('horizontal_barriers', []):
        a, b = sorted((a, b)); segs.append(((a, y), (b, y))); cells += [(a - 1, y), (b + 1, y)]
    for x, a, b in q.get('vertical_barriers', []):
        a, b = sorted((a, b)); segs.append(((x, a), (x, b))); cells += [(x, a - 1), (x, b + 1)]
    return segs, list(dict.fromkeys(cells))


def work(args):
    s, q = args
    (x0, x1), (y0, y1) = q['range_x'], q['range_y']
    segs, cells = raw_cells(q)
    dropped = {}
    for c in cells:
        if not (x0 + 1 < c[0] < x1 - 2 and y0 + 1 < c[1] < y1 - 2):
            dropped[c] = 'border'
        elif any(a[0] <= c[0] <= b[0] and a[1] <= c[1] <= b[1] for a, b in segs):
            dropped[c] = 'on_barrier'
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
    T = cb.oracle_chain(env, o.Grid(env))[1:-1]
    needed = {r for c, r in dropped.items() if any(max(abs(c[0] - t[0]), abs(c[1] - t[1])) <= 1 for t in T)}
    return s, {r: sum(v == r for v in dropped.values()) for r in ('border', 'on_barrier')}, needed


def load(O):
    return {(r['sample'], r['method'], r['prompt']): r for r in map(json.loads, open(O + '/results.jsonl'))}


if __name__ == '__main__':
    new, old = load(NEW), load(OLD)
    with Pool(16) as p:
        info = {s: (d, n) for s, d, n in p.map(work, list(o.load_queries(o.DATA)), chunksize=8)}
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v)) if v else float('nan')
    groups = defaultdict(list)
    for s, (d, n) in info.items():
        groups['all'].append(s)
        groups['no dropped cell' if not any(d.values()) else 'some dropped cell'].append(s)
        for r in ('border', 'on_barrier'):
            if d[r]:
                groups[f'dropped {r}'].append(s)
            if r in n:
                groups[f'NEEDED {r}'].append(s)
        if not n:
            groups['no needed dropped cell'].append(s)
    print('dropped cells per map: border %.2f, on_barrier %.2f' % (
        sum(d['border'] for d, _ in info.values()) / 1000, sum(d['on_barrier'] for d, _ in info.values()) / 1000))
    for p in ('standard', 'cot', 'repe'):
        print(f'\n[{p}]  group  n  old -> new  (paired new/old)  #new>200%')
        for g in ('all', 'no dropped cell', 'some dropped cell', 'dropped border', 'dropped on_barrier',
                  'NEEDED border', 'NEEDED on_barrier', 'no needed dropped cell'):
            S = groups[g]
            ra = lambda R: [R[(s, 'llm-astar', p)]['operation'] / new[(s, 'astar', '-')]['operation'] for s in S]
            a, b = ra(old), ra(new)
            print(f'  {g:24s} {len(S):4d}  {gm(a):6.2f} -> {gm(b):6.2f}  ({gm([y / x for x, y in zip(a, b)]):6.1f})  '
                  f'{sum(x > 2 for x in b):3d}')
        worst = sorted(groups['all'], key=lambda s: -new[(s, 'llm-astar', p)]['operation'] / old[(s, 'llm-astar', p)]['operation'])[:20]
        print('  worst 20 new/old -> needed dropped reasons:', [sorted(info[s][1]) for s in worst])
