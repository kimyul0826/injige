"""User idea: penalise the angle between the search step direction and the waypoint chain line.

Offline, stored V2 Llama responses, no inference.
  key(n) = g(n) + [c(n)] + h(n) [+ step penalty]
  h      : 'euclid' -> w * d(n, goal)                       (weighted A*)
           'route'  -> w * (d(n, t_k) + chain(t_k -> goal))  (route heuristic)
  angle  : theta = angle between step (s -> n) and a reference direction
           ref 'seg' -> chain segment t_{k-1} -> t_k   (the line joining waypoints)
           ref 'tgt' -> n's parent s -> t_k             (heading to the current waypoint)
           pen = lam * (1 - cos theta) * |s n|
  mode   : 'edge' -> accumulated into path cost c(n) = c(s) + pen  (cost shaping; true length reported)
           'step' -> only the last step's pen added to the key (not accumulated)
Target switching: on pop within 1.5 cells, or after 60 pops (same as route heuristic runs).
Accounting: dedupe heap pops (op), len(g) (sto), ratios vs A* (dedupe).
"""
import sys
sys.dont_write_bytecode = True
import collections, heapq, json, math
from multiprocessing import Pool
from pathlib import Path

from shapely.geometry import LineString, Point

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o           # noqa: E402
import astar_family as af             # noqa: E402
import gate_rule as gr                # noqa: E402

D = math.dist
INF = math.inf


def unit(a, b):
    d = D(a, b)
    return ((b[0] - a[0]) / d, (b[1] - a[1]) / d) if d else (0.0, 0.0)


def search(env, grid, T, h='euclid', w=2.0, lam=0.0, mode='none', ref='seg', timeout=60):
    start, goal = env.start, env.goal
    chain = [0.0] * len(T)
    for k in range(len(T) - 2, -1, -1):
        chain[k] = chain[k + 1] + D(T[k], T[k + 1])
    st = {'idx': 1}

    def H(p):
        i = st['idx']
        return w * (D(p, T[i]) + chain[i]) if h == 'route' else w * D(p, goal)

    def pen(s, n):
        if lam == 0 or mode == 'none':
            return 0.0
        i = st['idx']
        r = unit(T[i - 1], T[i]) if ref == 'seg' else unit(s, T[i])
        if r == (0.0, 0.0):
            return 0.0
        u = unit(s, n)
        return lam * (1 - (u[0] * r[0] + u[1] * r[1])) * D(s, n)

    g, c, last, parent = {start: 0.0}, {start: 0.0}, {start: 0.0}, {start: start}
    key = lambda p: g[p] + c[p] + H(p) + last[p]
    q = [(key(start), start)]
    closed, ops, since = set(), 0, 0
    found = False
    while q:
        _, s = heapq.heappop(q)
        if s in closed:
            continue
        ops += 1
        closed.add(s)
        if s == goal:
            found = True
            break
        old = st['idx']
        while st['idx'] < len(T) - 1 and D(s, T[st['idx']]) <= 1.5:
            st['idx'] += 1
        since += 1
        if timeout and st['idx'] < len(T) - 1 and since > timeout:
            st['idx'] += 1
        if st['idx'] != old:
            since = 0
            q = [(key(e[1]), e[1]) for e in q]
            heapq.heapify(q)
        for dx, dy in o.MOTIONS:
            n = (s[0] + dx, s[1] + dy)
            if n in closed or grid.coll(s, n):
                continue
            p = pen(s, n)
            ng = g[s] + D(s, n)
            nc = c[s] + (p if mode == 'edge' else 0.0)
            if ng + nc < g.get(n, INF) + c.get(n, INF):
                g[n], c[n], parent[n] = ng, nc, s
                last[n] = p if mode == 'step' else 0.0
                heapq.heappush(q, (key(n), n))
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return dict(op=ops, sto=len(g), len=sum(D(a, b) for a, b in zip(path, path[1:])))


CONFIGS = [('euclid', 2.0, 0.0, 'none', 'seg')]
for hh, ws in (('euclid', (1.0, 2.0)), ('route', (2.0,))):
    for w in ws:
        if hh == 'route':
            CONFIGS.append((hh, w, 0.0, 'none', 'seg'))
        for lam in (0.5, 1.0, 2.0, 4.0):
            for mode in ('edge', 'step'):
                for ref in ('seg', 'tgt'):
                    CONFIGS.append((hh, w, lam, mode, ref))


def work(args):
    sample, query, text = args
    env = o.Environment(json.loads(json.dumps(query)), collision_backend='shapely')
    grid = o.Grid(env)
    base, orc = af.oracle_T(env, grid)
    try:
        nodes = o.parse_path(text, mode='upstream')
    except ValueError:
        nodes = []
    raw = env.targets(nodes, 'upstream')
    line = LineString(raw) if len(set(raw)) > 1 else LineString([env.start, env.goal])
    near = lambda a, b, cands: min(cands, key=lambda gg: line.distance(Point(gg)))
    chains = {'oracle': orc, 'llm_topo_standard': gr.build(env, env.start, env.goal, near),
              'llm_raw_standard': raw}
    out = []
    for name, T in chains.items():
        for cfg in CONFIGS:
            if cfg == CONFIGS[0] and name != 'oracle':
                continue
            r = search(env, grid, T, *cfg)
            out.append(('wA*' if cfg == CONFIGS[0] else name, cfg, r['op'] / base['op'],
                        r['sto'] / base['sto'], r['len'] / base['len']))
    return out


def main():
    texts, _ = o.load_texts()
    tasks = [(s, q, texts[(s, 'standard')]) for s, q in o.load_queries(o.DATA)]
    agg = collections.defaultdict(list)
    with Pool(16) as pool:
        for part in pool.imap_unordered(work, tasks, chunksize=4):
            for name, cfg, *m in part:
                agg[(name, cfg)].append(m)
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    res = []
    for (name, cfg), v in sorted(agg.items(), key=lambda x: (x[0][0], str(x[0][1]))):
        r = dict(chain=name, h=cfg[0], w=cfg[1], lam=cfg[2], mode=cfg[3], ref=cfg[4],
                 op=gm([x[0] for x in v]), sto=gm([x[1] for x in v]), len=gm([x[2] for x in v]),
                 op_gt_2x=sum(x[0] > 2 for x in v), max_len=max(x[2] for x in v))
        res.append(r)
        print(f"{name:18s} {cfg[0]:6s} w={cfg[1]} lam={cfg[2]:<3} {cfg[3]:4s} {cfg[4]:3s} "
              f"op {r['op']:6.2f} sto {r['sto']:6.2f} len {r['len']:7.2f} >2x {r['op_gt_2x']:3d} maxlen {r['max_len']:.3f}")
    (HERE / 'results' / 'angle_penalty.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
