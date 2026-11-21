"""LLM-A* (official, stored V2 Llama) vs wallC (no LLM) on the 1000 assignment problems.

Two accountings, each applied to every method (including the A* denominator):
  upstream : released-code style; every heap pop counted (stale ones too), blocked neighbours stored.
             Matches Table 1 numbers of the reproduction.
  dedupe   : stale pops skipped, blocked neighbours not stored.
wallC needs reopening (admissible, not consistent); reopened expansions are counted.
Also reported: wall-clock search time (this machine, Python) and stored LLM generation time.
"""
import sys
sys.dont_write_bytecode = True
import collections, heapq, json, math, time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o           # noqa: E402
import wall_component_h as wc         # noqa: E402

D = math.dist
INF = math.inf


def astar_h(env, grid, h, w, upstream):
    start, goal = env.start, env.goal
    g, parent = {start: 0.0}, {start: start}
    q, closed, ops = [(w * h(start), start)], set(), 0
    while q:
        _, s = heapq.heappop(q)
        if s in closed and not upstream:
            continue
        ops += 1
        closed.add(s)
        if s == goal:
            break
        for dx, dy in o.MOTIONS:
            n = (s[0] + dx, s[1] + dy)
            if grid.coll(s, n):
                if upstream:
                    g.setdefault(n, INF)
                continue
            if upstream:
                g.setdefault(n, INF)
            ng = g[s] + D(s, n)
            if ng < g.get(n, INF):
                g[n], parent[n] = ng, s
                closed.discard(n)
                heapq.heappush(q, (ng + w * h(n), n))
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return dict(op=ops, sto=len(g), len=sum(D(a, b) for a, b in zip(path, path[1:])))


def work(args):
    sample, query, texts = args
    env = o.Environment(json.loads(json.dumps(query)), collision_backend='shapely')
    grid = o.Grid(env)
    out = []
    for acc, kw in (('upstream', {}), ('dedupe', dict(dedupe=True))):
        base = o.search(env, grid, None, **kw)
        rel = lambda r: (r['op'] / base['op'], r['sto'] / base['sto'], r['len'] / base['len'])
        for p in o.PROMPTS:
            try:
                nodes = o.parse_path(texts[p], mode='upstream')
            except ValueError:
                nodes = []
            T = env.targets(nodes, 'upstream')
            t = time.perf_counter(); r = o.search(env, grid, T, **kw); dt = time.perf_counter() - t
            out.append((acc, f'LLM-A* {p}', *rel(r), dt))
        for w in (2.0,):
            t = time.perf_counter(); r = o.search(env, grid, None, wg=w, **kw); dt = time.perf_counter() - t
            out.append((acc, f'weighted A* w={w}', *rel(r), dt))
        for w in (1.0, 1.5, 2.0):
            t = time.perf_counter(); r = astar_h(env, grid, wc.make_wallC(env), w, acc == 'upstream'); dt = time.perf_counter() - t
            out.append((acc, f'wallC w={w}', *rel(r), dt))
        t = time.perf_counter(); o.search(env, grid, None, **kw); dt = time.perf_counter() - t
        out.append((acc, 'A*', 1.0, 1.0, 1.0, dt))
    return out


def main():
    texts, stored = o.load_texts()
    tasks = [(s, q, {p: texts[(s, p)] for p in o.PROMPTS}) for s, q in o.load_queries(o.DATA)]
    gen = collections.defaultdict(list)
    for (s, m, p), r in stored.items():
        if m == 'llm-astar':
            meta = json.loads((o.SRC / 'responses' / f"{r['cache_key']}.json").read_text())
            gen[p].append(meta.get('generation_seconds') or 0.0)
    agg = collections.defaultdict(list)
    with Pool(16) as pool:
        for part in pool.imap_unordered(work, tasks, chunksize=4):
            for acc, name, *m in part:
                agg[(acc, name)].append(m)
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    res = {'llm_generation_seconds_mean': {p: sum(v) / len(v) for p, v in gen.items()}, 'rows': []}
    print('LLM generation seconds (mean, stored):', {p: round(sum(v) / len(v), 1) for p, v in gen.items()})
    for (acc, name), v in sorted(agg.items()):
        r = dict(accounting=acc, method=name, op=gm([x[0] for x in v]), sto=gm([x[1] for x in v]),
                 len=gm([x[2] for x in v]), max_len=max(x[2] for x in v), op_gt_2x=sum(x[0] > 2 for x in v),
                 search_ms=1000 * sum(x[3] for x in v) / len(v))
        res['rows'].append(r)
        print(f"{acc:8s} {name:20s} op {r['op']:6.2f} sto {r['sto']:6.2f} len {r['len']:7.2f} "
              f"max_len {r['max_len']:.3f} op>2x {r['op_gt_2x']:3d} search {r['search_ms']:.2f} ms")
    (HERE / 'results' / 'llmastar_vs_wallc.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
