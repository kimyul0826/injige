"""Offline improvement experiments on stored V2 Llama-3 responses.

No model inference. Reads project data/results read-only; writes only under exp/claude.
Run with PYTHONDONTWRITEBYTECODE=1 so nothing is written into llm_astar/__pycache__.
"""
import sys
sys.dont_write_bytecode = True
import argparse, collections, csv, gzip, hashlib, heapq, json, math, re, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import shapely

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from llm_astar.__main__ import load_queries           # noqa: E402
from llm_astar.models import parse_path               # noqa: E402
from llm_astar.search import Environment, MOTIONS, path_result  # noqa: E402

SRC = ROOT / 'outputs/llama3-table1-upstream-v2'
DATA = ROOT / 'data/environment_50_30.json'
PROMPTS = ('standard', 'cot', 'repe')
D = math.dist
INF = math.inf


# --------------------------------------------------------------------------- collision cache
class Grid:
    """Exact Shapely collision results for every unit move from interior cells (vectorised)."""
    def __init__(self, env):
        self.env = env
        x0, x1 = env.rx[0] + 1, env.rx[1] - 1
        y0, y1 = env.ry[0] + 1, env.ry[1] - 1
        src = [(x, y) for x in range(x0, x1) for y in range(y0, y1)]
        edges = [(s, (s[0] + dx, s[1] + dy)) for s in src for dx, dy in MOTIONS]
        lines = shapely.linestrings(np.array(edges, dtype=float))
        segs = np.array(env._segments, dtype=object)
        hit = shapely.intersects(lines[:, None], segs[None, :]).any(axis=1)
        self.blocked = {e: bool(h) for e, h in zip(edges, hit)}

    def coll(self, a, b):
        r = self.blocked.get((a, b))
        if r is None:
            r = self.env.collision(a, b)
            self.blocked[(a, b)] = r
        return r


# --------------------------------------------------------------------------- generic search
def search(env, grid, T=None, *, start=None, goal=None, dedupe=False, tie='node', wg=1.0, wt=1.0,
           switch='neighbor', timeout=None, via=None):
    """T=None: (weighted) A*.  T=list: LLM-A* target list (T[0]=start, T[-1]=goal).

    dedupe=False/tie='node'/wg=1/wt=1/switch='neighbor' reproduces the released implementation
    (duplicate heap pops counted, blocked neighbours stored with g=inf).
    """
    start = start or env.start
    goal = goal or env.goal
    T = list(T) if T else None
    wts = None
    if T:
        wts = list(wt) if isinstance(wt, (list, tuple)) else [wt] * len(T)
        chain = [0.0] * len(T)
        for k in range(len(T) - 2, -1, -1):
            chain[k] = chain[k + 1] + D(T[k], T[k + 1])
    idx = 1

    def f(p):
        if T and via is not None:
            return g[p] + via * (D(p, T[idx]) + chain[idx])
        h = wg * D(p, goal)
        if T:
            h = h + wts[idx] * D(p, T[idx])
        return g[p] + h       # same float association as the released code: g + (h_goal + h_target)

    def entry(p):
        return (f(p), D(p, goal), p) if tie == 'h' else (f(p), p)

    g, parent = {start: 0.0}, {start: start}
    q = [entry(start)]
    closed = set()
    ops = updates = 0
    since = 0
    found = False
    while q:
        s = heapq.heappop(q)[-1]
        if dedupe and s in closed:
            continue
        ops += 1
        closed.add(s)
        if s == goal:
            found = True
            break
        if T:
            old = idx
            if switch == 'pop':
                if s == T[idx] and s != goal:
                    idx = min(idx + 1, len(T) - 1)
            elif switch == 'progress':
                while idx < len(T) - 1 and (D(s, T[idx]) <= 1.5 or D(s, goal) < D(T[idx], goal)):
                    idx += 1
            since += 1
            if timeout and idx < len(T) - 1 and since > timeout:
                idx += 1
            if idx != old:
                since = 0
                updates += 1
                q = [entry(e[-1]) for e in q]
                heapq.heapify(q)
        for dx, dy in MOTIONS:
            n = (s[0] + dx, s[1] + dy)
            if switch == 'neighbor' and T and n == T[idx] and n != goal:
                idx = min(idx + 1, len(T) - 1)
                q = [entry(e[-1]) for e in q]
                heapq.heapify(q)
                updates += 1
            if n in closed:
                continue
            blocked = grid.coll(s, n)
            if blocked:
                if not dedupe:
                    g.setdefault(n, INF)
                continue
            if not dedupe:
                g.setdefault(n, INF)
            new = g[s] + D(s, n)
            if new < g.get(n, INF):
                g[n], parent[n] = new, s
                heapq.heappush(q, entry(n))
    path = []
    if found:
        path = [goal]
        while path[-1] != start:
            path.append(parent[path[-1]])
        path.reverse()
    length = sum(D(a, b) for a, b in zip(path, path[1:])) if found else None
    return dict(op=ops, sto=len(g), len=length, path=path, updates=updates, found=found)


# --------------------------------------------------------------------------- waypoint processing
def upstream_ok(env, p, obs):
    return (p not in obs and env.rx[0] + 1 < p[0] < env.rx[1] - 2 and env.ry[0] + 1 < p[1] < env.ry[1] - 2)


def obstacle_set(env):
    obs = set()
    for y, a, b in env.query.get('horizontal_barriers', []):
        if all(type(v) is int for v in (y, a, b)):
            obs.update((x, y) for x in range(a, b))
    for x, a, b in env.query.get('vertical_barriers', []):
        if all(type(v) is int for v in (x, a, b)):
            obs.update((x, y) for y in range(a, b))
    return obs


def snap(env, nodes, radius=3):
    """Move invalid waypoints to the nearest valid free cell (official criteria + exact free)."""
    obs = obstacle_set(env)
    out = []
    for p in nodes:
        p = (round(p[0]), round(p[1]))
        if upstream_ok(env, p, obs) and env.free(p):
            out.append(p)
            continue
        best = None
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                c = (p[0] + dx, p[1] + dy)
                if upstream_ok(env, c, obs) and env.free(c):
                    d = math.hypot(dx, dy)
                    if d <= radius and (best is None or d < best[0]):
                        best = (d, c)
        if best:
            out.append(best[1])
    return env.targets(out, 'upstream')


def clamp(env, nodes):
    """Clip out-of-map coordinates into the valid band, then official filter."""
    lo_x, hi_x = env.rx[0] + 2, env.rx[1] - 3
    lo_y, hi_y = env.ry[0] + 2, env.ry[1] - 3
    out = [(min(hi_x, max(lo_x, round(p[0]))), min(hi_y, max(lo_y, round(p[1])))) for p in nodes]
    return env.targets(out, 'upstream')


def exact_filter(env, nodes):
    T = env.targets(nodes, 'upstream')
    return [T[0]] + [p for p in T[1:-1] if env.free(p)] + [T[-1]]


def prune_detour(T, beta=1.5):
    out = [T[0]]
    for i in range(1, len(T) - 1):
        a, w, b = out[-1], T[i], T[i + 1]
        if D(a, w) + D(w, b) <= beta * D(a, b) + 2:
            out.append(w)
    return out + [T[-1]]


def drop_backward(T, slack=2.0):
    out = [T[0]]
    for w in T[1:-1]:
        if D(w, T[-1]) <= D(out[-1], T[-1]) + slack:
            out.append(w)
    return out + [T[-1]]


def string_pull(env, P):
    if len(P) <= 2:
        return list(P)
    out, i = [P[0]], 0
    while i < len(P) - 1:
        j = len(P) - 1
        while j > i + 1 and env.collision(P[i], P[j]):
            j -= 1
        out.append(P[j])
        i = j
    return out


def dedup_consecutive(P):
    out = []
    for p in P:
        if not out or out[-1] != p:
            out.append(p)
    return out


def chain_len(T):
    return sum(D(a, b) for a, b in zip(T, T[1:]))


def los_weights(env, T):
    return [1.0] + [1.0 if k == len(T) - 1 or not env.collision(T[k - 1], T[k]) else 0.0
                    for k in range(1, len(T))]


def selected_points(text, start):
    """RePE: last 'Selected Point' of every iteration block."""
    blocks = re.split(r'-\s*\w+\s+Iteration on', text)
    pts = []
    for b in blocks[1:]:
        m = re.findall(r'Selected Point:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', b)
        if m:
            x, y = m[-1]
            pts.append((float(x) if '.' in x else int(x), float(y) if '.' in y else int(y)))
    return [tuple(start)] + pts if pts else None


# --------------------------------------------------------------------------- variant definitions
# Each LLM-A* variant: (name, group, waypoint processor, search kwargs). acc = accounting/tie keys
# that are also applied to the matching A* baseline so ratios stay fair.
def W(name):  # waypoint processors -> function(env, nodes, ctx) -> T
    return {
        'official': lambda env, n, c: env.targets(n, 'upstream'),
        'exact': lambda env, n, c: exact_filter(env, n),
        'snap': lambda env, n, c: snap(env, n),
        'clamp': lambda env, n, c: clamp(env, n),
        'prune1.5': lambda env, n, c: prune_detour(env.targets(n, 'upstream'), 1.5),
        'prune1.3': lambda env, n, c: prune_detour(env.targets(n, 'upstream'), 1.3),
        'backward': lambda env, n, c: drop_backward(env.targets(n, 'upstream')),
        'pull': lambda env, n, c: string_pull(env, env.targets(n, 'upstream')),
        'selected': lambda env, n, c: env.targets(c['selected'] or n, 'upstream'),
        'skip_if_los': lambda env, n, c: ([env.start, env.goal] if not env.collision(env.start, env.goal)
                                          else env.targets(n, 'upstream')),
        'snap+prune1.5': lambda env, n, c: prune_detour(snap(env, n), 1.5),
        'snap+prune1.5+pull': lambda env, n, c: string_pull(env, prune_detour(snap(env, n), 1.5)),
        'none': lambda env, n, c: [env.start, env.goal],
    }[name]


VARIANTS = [
    # name, group, processor, kwargs
    ('official', 'baseline', 'official', {}),
    # --- waypoint post-processing (official search)
    ('wp:exact_filter', 'waypoint', 'exact', {}),
    ('wp:snap', 'waypoint', 'snap', {}),
    ('wp:clamp', 'waypoint', 'clamp', {}),
    ('wp:prune_detour1.5', 'waypoint', 'prune1.5', {}),
    ('wp:prune_detour1.3', 'waypoint', 'prune1.3', {}),
    ('wp:drop_backward', 'waypoint', 'backward', {}),
    ('wp:string_pull', 'waypoint', 'pull', {}),
    ('wp:repe_selected_points', 'waypoint', 'selected', {}),
    ('wp:skip_llm_if_line_of_sight', 'waypoint', 'skip_if_los', {}),
    # --- search / evaluation function (official waypoints)
    ('s:dedupe', 'search', 'official', dict(dedupe=True)),
    ('s:tie_h', 'search', 'official', dict(tie='h')),
    ('s:dedupe+tie_h', 'search', 'official', dict(dedupe=True, tie='h')),
    ('s:switch_on_pop', 'search', 'official', dict(switch='pop')),
    ('s:switch_progress', 'search', 'official', dict(switch='progress')),
    ('s:timeout30', 'search', 'official', dict(timeout=30)),
    ('s:timeout60', 'search', 'official', dict(timeout=60)),
    ('s:wt0.5', 'search', 'official', dict(wt=0.5)),
    ('s:wt2', 'search', 'official', dict(wt=2.0)),
    ('s:wg2_wt0.5', 'search', 'official', dict(wg=2.0, wt=0.5)),
    ('s:wg0.5_wt1', 'search', 'official', dict(wg=0.5)),
    ('s:via1.5', 'search', 'official', dict(via=1.5)),
    ('s:via2', 'search', 'official', dict(via=2.0)),
    ('s:los_gate', 'search', 'official', dict(wt='los')),
    ('s:fallback_astar', 'search', 'official', dict(fallback=True)),
    # --- combinations
    ('c:snap+progress', 'combo', 'snap', dict(switch='progress')),
    ('c:snap+prune+progress', 'combo', 'snap+prune1.5', dict(switch='progress')),
    ('c:snap+prune+progress+tie_h', 'combo', 'snap+prune1.5', dict(switch='progress', tie='h')),
    ('c:snap+prune+progress+tie_h+dedupe', 'combo', 'snap+prune1.5', dict(switch='progress', tie='h', dedupe=True)),
    ('c:snap+prune+pull+progress+tie_h+dedupe', 'combo', 'snap+prune1.5+pull', dict(switch='progress', tie='h', dedupe=True)),
    ('c:progress+los_gate+tie_h+dedupe', 'combo', 'snap+prune1.5', dict(switch='progress', tie='h', dedupe=True, wt='los')),
    ('c:progress+wg1.5_wt0.5+tie_h+dedupe', 'combo', 'snap+prune1.5', dict(switch='progress', tie='h', dedupe=True, wg=1.5, wt=0.5)),
    ('c:progress+timeout30+tie_h+dedupe', 'combo', 'snap+prune1.5', dict(switch='progress', tie='h', dedupe=True, timeout=30)),
]
WA_WEIGHTS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
ACCS = [dict(dedupe=False, tie='node'), dict(dedupe=True, tie='node'),
        dict(dedupe=False, tie='h'), dict(dedupe=True, tie='h')]


def acc_key(kw):
    return (bool(kw.get('dedupe', False)), kw.get('tie', 'node'))


# --------------------------------------------------------------------------- LLM-only variants
def llm_only_paths(env, grid, nodes, ctx):
    """Return {variant: path}. Hybrid repairs are labelled separately in the report."""
    out = {'raw': list(nodes)}
    if not nodes:
        return {k: [] for k in ('raw', 'official_filter', 'exact_filter', 'snap', 'clamp', 'pull',
                                'snap+pull', 'snap+prune+pull', 'selected', 'hybrid:repair', 'hybrid:repair+pull')}
    out['official_filter'] = env.targets(nodes, 'upstream')
    out['exact_filter'] = exact_filter(env, nodes)
    out['snap'] = snap(env, nodes)
    out['clamp'] = clamp(env, nodes)
    out['pull'] = string_pull(env, out['official_filter'])
    out['snap+pull'] = string_pull(env, out['snap'])
    out['snap+prune+pull'] = string_pull(env, prune_detour(out['snap'], 1.5))
    out['selected'] = env.targets(ctx['selected'], 'upstream') if ctx['selected'] else out['official_filter']
    # Hybrid: keep LLM waypoints, replace only colliding segments by grid A* sub-paths.
    P = dedup_consecutive([p for p in out['snap'] if all(type(v) is int for v in p)])
    rep = [P[0]]
    ok = True
    for a, b in zip(P, P[1:]):
        if env.collision(a, b):
            r = search(env, grid, None, start=a, goal=b, dedupe=True)
            if not r['found']:
                ok = False
                break
            rep += r['path'][1:]
        else:
            rep.append(b)
    out['hybrid:repair'] = rep if ok else []
    out['hybrid:repair+pull'] = string_pull(env, rep) if ok else []
    return out


# --------------------------------------------------------------------------- worker
def load_texts():
    rows = [json.loads(x) for x in (SRC / 'results.jsonl').read_text().splitlines()]
    texts, stored = {}, {}
    for r in rows:
        if r['method'] == 'llm-astar':
            texts[(r['sample'], r['prompt'])] = json.loads(
                (SRC / 'responses' / f"{r['cache_key']}.json").read_text())['text']
        stored[(r['sample'], r['method'], r['prompt'])] = r
    return texts, stored


def work(args):
    map_id, items = args          # items: [(sample, query, {prompt: text}, stored_rows)]
    out = []
    for sample, query, texts, stored in items:
        env = Environment(json.loads(json.dumps(query)), collision_backend='shapely')
        grid = Grid(env)
        base = {}
        for acc in ACCS:
            for w in WA_WEIGHTS:
                r = search(env, grid, None, wg=w, **acc)
                base[(acc_key(acc), w)] = r
                out.append(dict(kind='astar', sample=sample, prompt='-', variant=f'wA*{w}',
                                dedupe=acc['dedupe'], tie=acc['tie'], op=r['op'], sto=r['sto'], len=r['len']))
        # exactness check against stored V2 A*
        a0 = base[((False, 'node'), 1.0)]
        sa = stored[('astar', '-')]
        assert (a0['op'], a0['sto']) == (sa['operation'], sa['storage']) and math.isclose(a0['len'], sa['length']), sample
        per_prompt_T = {}
        for prompt in PROMPTS:
            text = texts[prompt]
            try:
                nodes = parse_path(text, mode='upstream')
            except ValueError:
                nodes = []
            ctx = dict(selected=selected_points(text, query['start']) if prompt == 'repe' else None)
            for name, group, proc, kw in VARIANTS:
                kw = dict(kw)
                T = W(proc)(env, nodes, ctx)
                if name == 'official':
                    per_prompt_T[prompt] = T
                if kw.pop('fallback', False) and len(T) <= 2:
                    T = None
                if kw.get('wt') == 'los':
                    kw['wt'] = los_weights(env, T) if T else 1.0
                r = search(env, grid, T, **kw)
                if name == 'official':
                    st = stored[('llm-astar', prompt)]
                    assert (r['op'], r['sto']) == (st['operation'], st['storage']), (sample, prompt)
                    assert math.isclose(r['len'], st['length']), (sample, prompt)
                out.append(dict(kind='llm-astar', sample=sample, prompt=prompt, variant=name, group=group,
                                dedupe=bool(kw.get('dedupe', False)), tie=kw.get('tie', 'node'),
                                op=r['op'], sto=r['sto'], len=r['len'], updates=r['updates'],
                                n_targets=len(T) if T else 0))
            # LLM-only
            for name, path in llm_only_paths(env, grid, nodes, ctx).items():
                res = path_result(env, [tuple(p) for p in path])
                if name == 'raw':
                    st = stored[('llm', prompt)]
                    assert res['valid'] == st['valid'] and [list(p) for p in path] == st['path'], (sample, prompt)
                out.append(dict(kind='llm', sample=sample, prompt=prompt, variant=name,
                                valid=res['valid'], len=res['length'], n_points=len(path)))
        # Ensemble over the 3 stored responses (costs 3 LLM calls in deployment).
        best = min(PROMPTS, key=lambda p: (chain_len(per_prompt_T[p]), PROMPTS.index(p)))
        for name, kw in [('ens:min_chain_len', {}),
                         ('ens:min_chain_len+progress+tie_h+dedupe', dict(switch='progress', tie='h', dedupe=True))]:
            r = search(env, grid, per_prompt_T[best], **kw)
            out.append(dict(kind='llm-astar', sample=sample, prompt='ensemble', variant=name, group='ensemble',
                            dedupe=bool(kw.get('dedupe', False)), tie=kw.get('tie', 'node'),
                            op=r['op'], sto=r['sto'], len=r['len'], updates=r['updates'],
                            n_targets=len(per_prompt_T[best])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--maps', type=int, default=None, help='limit number of maps (debug)')
    ap.add_argument('--out', default=str(HERE / 'results'))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    texts, stored = load_texts()
    queries = list(load_queries(DATA))
    by_map = collections.defaultdict(list)
    for sample, q in queries:
        m = sample.split(':')[0]
        by_map[m].append((sample, q, {p: texts[(sample, p)] for p in PROMPTS},
                          {(meth, p): stored[(sample, meth, p)] for meth, p in
                           [('astar', '-')] + [(mm, pp) for mm in ('llm', 'llm-astar') for pp in PROMPTS]}))
    tasks = list(by_map.items())[:args.maps]
    rows = []
    with Pool(args.workers) as pool:
        for i, part in enumerate(pool.imap_unordered(work, tasks), 1):
            rows += part
            if i % 10 == 0:
                print(f'{i}/{len(tasks)} maps, {time.time() - t0:.0f}s', flush=True)
    with gzip.open(out_dir / 'per_sample.jsonl.gz', 'wt') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    meta = dict(inference_calls=0, maps=len(tasks), rows=len(rows), seconds=time.time() - t0,
                source_results_sha256=hashlib.sha256((SRC / 'results.jsonl').read_bytes()).hexdigest(),
                exactness='official variant and wA*1.0 (dedupe=False, tie=node) asserted equal to stored V2 '
                          'operation/storage/length for every sample and prompt')
    (out_dir / 'run_meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
