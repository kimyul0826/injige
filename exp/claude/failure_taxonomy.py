"""Failure taxonomy of LLM waypoint responses and its cost inside the official LLM-A* search (no API calls).

Each response is put in exactly one class (first match):
  parse/trunc   no coordinate list parsed or output truncated (finish_reason=length)
  no waypoint   the model returned start and goal only
  all dropped   waypoints given but the official filter removed all of them -> search == weighted A* w=2
  valid         the model's full path is collision-free (LLM-only success)
  on barrier    some waypoint lies on a barrier (dropped by the filter), path also collides elsewhere
  corner cut    exactly one segment crosses a barrier
  multi cross   two or more segments cross barriers
Orthogonal flag: detour = filtered chain length / optimal path length >= 1.3.
Per class: share, GM operation ratio (LLM-A*) and of weighted A* w=2 on the same problems.
Also: operation ratio by the problem's minimal waypoint count k (0/1/2 exact; 3 over cells near barrier ends; 4 = more).
Writes exp/claude/results/failure_taxonomy.json and failure_cases.json (per-problem rows for plotting).
"""
import sys
sys.dont_write_bytecode = True
import json, math
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from shapely.geometry import Point

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import offline_improve as o           # noqa: E402
import build_repe2_demos as b         # noqa: E402

SETS = [('llama original', 'outputs/llama3-table1-upstream-v2', ('standard', 'cot', 'repe')),
        ('gpt original', 'outputs/gpt4omini-table1-upstream-batch-v1', ('standard', 'cot', 'repe')),
        ('gpt waypoint-limit', 'outputs/gpt4omini-waypoint-limit-batch-v1', ('standard', 'cot', 'repe')),
        ('gpt vot+limit', 'outputs/gpt4omini-vot-waypoint-limit-batch-v1', ('vot',)),
        ('gpt endpoint', 'outputs/gpt4omini-barrier-endpoints-demos-batch-v1', ('standard', 'cot', 'repe')),
        ('gpt repe2', 'outputs/gpt4omini-repe2-batch-v1', ('repe2',))]
for extra in ('repe2_noround', 'repe2_origdemos'):   # E1 ablations, included once collected
    d = ROOT / f'outputs/gpt4omini-{extra}-batch-v1'
    if (d / 'summary.json').exists():
        SETS.append((f'gpt {extra}', str(d.relative_to(ROOT)), (extra,)))


def load(d):
    rows = {(r['sample'], r['method'], r['prompt']): r for r in map(json.loads, open(ROOT / d / 'results.jsonl'))}
    resp = {}
    for (s, m, p), r in rows.items():
        if m == 'llm-astar':
            j = json.loads((ROOT / d / 'responses' / f"{r['cache_key']}.json").read_text())
            resp[(s, p)] = (j['text'], j.get('finish_reason'))
    return rows, resp


DATA = {name: load(d) for name, d, _ in SETS}


def classify(env, q, text, finish):
    try:
        P = o.parse_path(text, mode='upstream')
    except ValueError:
        P = []
    if not P or finish == 'length':
        return 'parse/trunc', P
    S, G = tuple(q['start']), tuple(q['goal'])
    mid = [p for p in P if p not in (S, G)]
    if not mid:
        return 'no waypoint', P
    if len(env.targets(P, 'upstream')) == 2:
        return 'all dropped', P
    bad = sum(env.collision(a, c) for a, c in zip(P, P[1:]))
    if bad == 0:
        return 'valid', P
    if any(any(Point(m).intersects(w) for w in env._segments) for m in mid):
        return 'on barrier', P
    return ('corner cut' if bad == 1 else 'multi cross'), P


def work(args):
    s, q = args
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
    grid = o.Grid(env)
    base = o.search(env, grid, None)
    w2 = o.search(env, grid, None, wg=2.0)['op'] / base['op']
    opt = o.search(env, grid, None, dedupe=True)['len']
    res = b.solve(q)
    k = res[0] if res else 4          # None: no chain with <= 3 waypoints over cells near barrier ends
    out = []
    for name, _, prompts in SETS:
        rows, resp = DATA[name]
        for p in prompts:
            text, finish = resp[(s, p)]
            cls, P = classify(env, q, text, finish)
            T = env.targets(P, 'upstream')
            chain = sum(math.dist(a, c) for a, c in zip(T, T[1:]))
            op = rows[(s, 'llm-astar', p)]['operation'] / base['op']
            out.append(dict(set=name, prompt=p, sample=s, cls=cls, op=op, w2=w2, k=k, detour=chain / opt,
                            path=[list(x) for x in P], targets=[list(x) for x in T]))
    return out


if __name__ == '__main__':
    with Pool(16) as pool:
        rows = [r for part in pool.map(work, list(o.load_queries(o.DATA)), chunksize=8) for r in part]
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v)) if v else float('nan')
    groups = defaultdict(list)
    for r in rows:
        groups[(r['set'], r['prompt'])].append(r)
    classes = ['parse/trunc', 'no waypoint', 'all dropped', 'valid', 'on barrier', 'corner cut', 'multi cross']
    summary = {}
    for (name, p), R in groups.items():
        e = dict(op=gm([r['op'] for r in R]), w2=gm([r['w2'] for r in R]), classes={}, by_k={},
                 detour_share=sum(r['detour'] >= 1.3 for r in R) / len(R))
        for c in classes:
            sub = [r for r in R if r['cls'] == c]
            e['classes'][c] = dict(n=len(sub), op=gm([r['op'] for r in sub]), w2=gm([r['w2'] for r in sub]))
        for kk in (0, 1, 2, 3):
            sub = [r for r in R if min(r['k'], 3) == kk]
            e['by_k'][kk] = dict(n=len(sub), op=gm([r['op'] for r in sub]), w2=gm([r['w2'] for r in sub]))
        sub = [r for r in R if r['detour'] >= 1.3]
        e['detour'] = dict(n=len(sub), op=gm([r['op'] for r in sub]), w2=gm([r['w2'] for r in sub]))
        summary[f'{name}|{p}'] = e
    lines = []
    for key, e in summary.items():
        lines.append(f"\n== {key}: op {e['op']:.2f} (wA*2 {e['w2']:.2f}); detour>=1.3: {e['detour']['n']} op {e['detour']['op']:.1f} vs wA* {e['detour']['w2']:.1f}")
        lines.append('   ' + '  '.join(f"{c} {v['n']}:{v['op']:.0f}/{v['w2']:.0f}" for c, v in e['classes'].items() if v['n']))
        lines.append('   by min k: ' + '  '.join(f"k={kk} n={v['n']} {v['op']:.1f}/{v['w2']:.1f}" for kk, v in e['by_k'].items()))
    text = '\n'.join(lines)
    print(text)
    (HERE / 'results' / 'failure_taxonomy.txt').write_text(text + '\n')
    (HERE / 'results' / 'failure_taxonomy.json').write_text(json.dumps(summary, indent=1))
    (HERE / 'results' / 'failure_cases.json').write_text(json.dumps(rows))
