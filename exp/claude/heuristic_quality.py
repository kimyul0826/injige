"""Decompose a heuristic into strength (scale) and precision (shape) against the true cost-to-go h*.

For every problem: h* = exact grid distance to the goal (reverse Dijkstra, same 8-connected moves and
collision check). Evaluated on the states expanded by plain A* (the region search actually visits).
  scale     = median(h / h*)                       -> how "strong" (inflated) the heuristic is
  shape_err = median(|h/scale - h*| / h*)          -> error left after removing the scale = imprecision
  spearman  = rank correlation(h, h*)              -> does h order states like h* does
Heuristics (target fixed to the first waypoint T[1]):
  euclid     : d(s, goal)                          (A*)
  2euclid    : 2 d(s, goal)                        (weighted A* w=2 = LLM-A* without waypoints)
  llmastar   : d(s, goal) + d(s, T1)               (LLM-A* term)
  route      : d(s, T1) + chain(T1 -> goal)        (remaining route length via the waypoints)
No inference; stored V2 Llama responses and oracle waypoints.
"""
import sys
sys.dont_write_bytecode = True
import collections, heapq, json, math, statistics
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o      # noqa: E402
import astar_family as af        # noqa: E402

D = math.dist


def true_cost(env, grid):
    goal = env.goal
    dist = {goal: 0.0}
    pq = [(0.0, goal)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for dx, dy in o.MOTIONS:
            v = (u[0] + dx, u[1] + dy)
            if not env.inside(v) or grid.coll(u, v):
                continue
            nd = d + D(u, v)
            if nd < dist.get(v, math.inf):
                dist[v] = nd; heapq.heappush(pq, (nd, v))
    return dist


def spearman(a, b):
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for k, i in enumerate(order):
            r[i] = k
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra)); vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va and vb else 0.0


def work(args):
    sample, query, texts = args
    env = o.Environment(json.loads(json.dumps(query)), collision_backend='shapely')
    grid = o.Grid(env)
    hstar = true_cost(env, grid)
    # states expanded by plain A* (closed set)
    base, orc = af.oracle_T(env, grid)
    states = []
    g, q, closed = {env.start: 0.0}, [(D(env.start, env.goal), env.start)], set()
    while q:
        _, s = heapq.heappop(q)
        if s in closed:
            continue
        closed.add(s)
        if s == env.goal:
            break
        for dx, dy in o.MOTIONS:
            n = (s[0] + dx, s[1] + dy)
            if n in closed or grid.coll(s, n):
                continue
            if g[s] + D(s, n) < g.get(n, math.inf):
                g[n] = g[s] + D(s, n); heapq.heappush(q, (g[n] + D(n, env.goal), n))
    states = [s for s in closed if s != env.goal and hstar.get(s, 0) > 0]
    chains = {'oracle': orc}
    for p in o.PROMPTS:
        try:
            nodes = o.parse_path(texts[p], mode='upstream')
        except ValueError:
            nodes = []
        chains[p] = env.targets(nodes, 'upstream')
    hs = [hstar[s] for s in states]
    out = []

    def measure(name, vals):
        ratio = [v / t for v, t in zip(vals, hs)]
        scale = statistics.median(ratio)
        err = statistics.median([abs(v / scale - t) / t for v, t in zip(vals, hs)])
        out.append((name, scale, err, spearman(vals, hs)))
    measure('euclid', [D(s, env.goal) for s in states])
    measure('2euclid', [2 * D(s, env.goal) for s in states])
    for k, T in chains.items():
        t1 = T[1]
        rest = sum(D(a, b) for a, b in zip(T[1:], T[2:]))
        measure(f'llmastar|{k}', [D(s, env.goal) + D(s, t1) for s in states])
        measure(f'route|{k}', [D(s, t1) + rest for s in states])
    return out


def main():
    texts, _ = o.load_texts()
    tasks = [(s, q, {p: texts[(s, p)] for p in o.PROMPTS}) for s, q in o.load_queries(o.DATA)]
    agg = collections.defaultdict(list)
    with Pool(16) as pool:
        for part in pool.imap_unordered(work, tasks, chunksize=4):
            for name, *m in part:
                agg[name].append(m)
    res = {}
    for k, v in agg.items():
        res[k] = dict(scale=statistics.median(x[0] for x in v), shape_err=statistics.median(x[1] for x in v),
                      spearman=statistics.median(x[2] for x in v))
        print(f"{k:20s} scale {res[k]['scale']:.3f}  shape_err {100 * res[k]['shape_err']:5.1f}%  spearman {res[k]['spearman']:.3f}")
    (HERE / 'results' / 'heuristic_quality.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
