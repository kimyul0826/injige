"""E-graph with several LLM routes at once (union of collision-free segments). No inference.

Uses the 3 stored responses (standard / CoT / RePE) of each problem as 3 alternative routes,
as a proxy for asking the LLM for K diverse routes in one call.
"""
import sys
sys.dont_write_bytecode = True
import collections, json, math, random
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o      # noqa: E402
import logic_variants as lv      # noqa: E402

D = math.dist


def egraph_multi(env, grid, chains, eps_e, w):
    # concatenate chains with a separator that egraph_h treats as a collision (skip), by running
    # egraph_h on each chain's segments: build one combined list of segments
    segs = [(a, b) for T in chains for a, b in zip(T, T[1:])]
    # egraph_h expects a chain; emulate by a chain builder that inserts only real segments
    class Multi(list):
        pass
    # simple re-implementation: reuse egraph_h by passing a chain where consecutive pairs = segs
    flat = []
    for a, b in segs:
        flat += [a, b, None]  # None breaks the chain
    return run_with_segments(env, grid, segs, eps_e, w)


def run_with_segments(env, grid, segs, eps_e, w):
    import heapq
    INF = math.inf
    goal = env.goal
    nodes, edges = [goal], []
    for a, b in segs:
        if a == b or env.collision(a, b):
            continue
        n = max(1, math.ceil(D(a, b)))
        base = len(nodes)
        nodes += [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(n + 1)]
        edges += [(base + i, base + i + 1, D(nodes[base + i], nodes[base + i + 1])) for i in range(n)]
    N = len(nodes)
    adj = [dict() for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            adj[i][j] = adj[j][i] = eps_e * D(nodes[i], nodes[j])
    for i, j, c in edges:
        if c < adj[i][j]:
            adj[i][j] = adj[j][i] = c
    hv = [INF] * N
    hv[0] = 0.0
    pq = [(0.0, 0)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > hv[u]:
            continue
        for v, c in adj[u].items():
            if d + c < hv[v]:
                hv[v] = d + c; heapq.heappush(pq, (d + c, v))
    items = list(zip(nodes, hv))
    cache = {}

    def H(p):
        if p not in cache:
            cache[p] = min(eps_e * D(p, v) + h for v, h in items)
        return cache[p]
    start = env.start
    g, parent = {start: 0.0}, {start: start}
    q = [(w * H(start), start)]
    closed = set()
    ops = 0
    while q:
        _, s = heapq.heappop(q)
        if s in closed:
            continue
        ops += 1
        closed.add(s)
        if s == goal:
            return lv.finish(parent, start, goal, ops, g)
        for n, c in lv.neighbours(grid, s):
            if n in closed:
                continue
            new = g[s] + c
            if new < g.get(n, INF):
                g[n], parent[n] = new, s
                heapq.heappush(q, (new + w * H(n), n))
    raise RuntimeError('no path')


CFG = [(1.5, 1.33), (2.0, 1.25)]


def work(args):
    sample, query, texts = args
    env = o.Environment(json.loads(json.dumps(query)), collision_backend='shapely')
    grid = o.Grid(env)
    base = o.search(env, grid, None, dedupe=True)
    chains = {}
    for p in o.PROMPTS:
        try:
            nodes = o.parse_path(texts[p], mode='upstream')
        except ValueError:
            nodes = []
        chains[p] = env.targets(nodes, 'upstream')
    out = []
    for e, w in CFG:
        ctl = run_with_segments(env, grid, [], e, w)
        res = {'noLLM': ctl}
        for p in o.PROMPTS:
            res[p] = run_with_segments(env, grid, list(zip(chains[p], chains[p][1:])), e, w)
        res['union3'] = run_with_segments(env, grid, [s for p in o.PROMPTS for s in zip(chains[p], chains[p][1:])], e, w)
        for k, r in res.items():
            out.append((f'eE{e}_w{w}', k, sample, r['op'] / base['op'], r['sto'] / base['sto'], r['len'] / base['len'],
                        r['op'] / ctl['op']))
    return out


def main():
    texts, _ = o.load_texts()
    tasks = [(s, q, {p: texts[(s, p)] for p in o.PROMPTS}) for s, q in o.load_queries(o.DATA)]
    agg = collections.defaultdict(list)
    with Pool(16) as pool:
        for part in pool.imap_unordered(work, tasks, chunksize=4):
            for cfg, k, s, *m in part:
                agg[(cfg, k)].append((s, *m))
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    res = []
    for (cfg, k), v in sorted(agg.items()):
        by = collections.defaultdict(list)
        for s, *_m, rc in v:
            by[s.split(':')[0]].append(math.log(rc))
        maps = list(by); rng = random.Random(0)
        bs = sorted((lambda pick: sum(x for m in pick for x in by[m]) / sum(len(by[m]) for m in pick))(
            [rng.choice(maps) for _ in maps]) for _ in range(2000))
        r = dict(cfg=cfg, routes=k, op=gm([x[1] for x in v]), sto=gm([x[2] for x in v]), len=gm([x[3] for x in v]),
                 op_vs_noLLM=gm([x[4] for x in v]), ci=[100 * math.exp(bs[50]), 100 * math.exp(bs[1949])],
                 max_len=max(x[3] for x in v), op_gt_2x=sum(x[1] > 2 for x in v))
        res.append(r)
        print(f"{cfg:12s} {k:8s} op {r['op']:6.2f} sto {r['sto']:6.2f} len {r['len']:7.2f} "
              f"vs noLLM {r['op_vs_noLLM']:6.1f} [{r['ci'][0]:.1f},{r['ci'][1]:.1f}] max_len {r['max_len']:.3f} op>2x {r['op_gt_2x']}")
    (HERE / 'results' / 'egraph_multi.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
