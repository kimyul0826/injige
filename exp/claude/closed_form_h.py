"""Heuristics computed per node in closed form: no pre-search, no grid Dijkstra, no visibility-graph search.

  euclid        : d(s, goal)
  octile        : 8-connected free-space distance (exact without walls, admissible, >= euclid)
  wall1         : max over walls W crossing segment s->goal of min_e [d(s,e) + d(e,goal)], e = passable end of W
                  (shortest way around a single segment = admissible lower bound; max of admissible = admissible)
  wall1_oct     : same with octile distances (approximate)
  wall_sum      : euclid + sum over crossing walls of their detour (inadmissible, stronger)
Per-node cost: O(#walls) segment-intersection tests (reported as tests per expansion).
Reference: rule-route (one geometric chain per problem, gate_rule.build), not per node.
All with weight w: f = g + w h.  Ratios vs plain A* (euclid, dedupe).
"""
import sys
sys.dont_write_bytecode = True
import collections, heapq, json, math
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import offline_improve as o           # noqa: E402
import gate_rule as gr                # noqa: E402
from llm_astar.search import intersects   # noqa: E402

D = math.dist
INF = math.inf
WS = (1.0, 1.5, 2.0, 3.0)


def octile(a, b):
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)


def make_h(env, kind):
    goal = env.goal
    x0, x1, y0, y1 = env.rx[0], env.rx[1] - 1, env.ry[0], env.ry[1] - 1
    on_border = lambda p: p[0] <= x0 or p[0] >= x1 or p[1] <= y0 or p[1] >= y1
    walls = [(a, b, [e for e in (a, b) if not on_border(e)]) for a, b in env.walls]
    dist = octile if kind in ('octile', 'wall1_oct') else D
    if kind in ('euclid', 'octile'):
        return lambda s: dist(s, goal)

    def h(s):
        base = dist(s, goal)
        best, total = base, base
        for a, b, ends in walls:
            if not ends or not intersects(s, goal, a, b):
                continue
            det = min(dist(s, e) + dist(e, goal) for e in ends)
            best = max(best, det)
            total += det - base
        return total if kind == 'wall_sum' else best
    return h


def astar(env, grid, h, w):
    start, goal = env.start, env.goal
    g, parent = {start: 0.0}, {start: start}
    q, closed, ops, hc = [(w * h(start), start)], set(), 0, 1
    while q:
        _, s = heapq.heappop(q)
        if s in closed:
            continue
        ops += 1
        closed.add(s)
        if s == goal:
            break
        for dx, dy in o.MOTIONS:
            n = (s[0] + dx, s[1] + dy)
            if n in closed or grid.coll(s, n):
                continue
            ng = g[s] + D(s, n)
            if ng < g.get(n, INF):
                g[n], parent[n] = ng, s
                hc += 1
                heapq.heappush(q, (ng + w * h(n), n))
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return dict(op=ops, sto=len(g), len=sum(D(a, b) for a, b in zip(path, path[1:])), hcalls=hc)


def work(args):
    sample, query = args
    env = o.Environment(json.loads(json.dumps(query)), collision_backend='shapely')
    grid = o.Grid(env)
    base = astar(env, grid, make_h(env, 'euclid'), 1.0)
    out = []
    for kind in ('euclid', 'octile', 'wall1', 'wall1_oct', 'wall_sum'):
        h = make_h(env, kind)
        for w in WS:
            r = astar(env, grid, h, w)
            out.append((kind, w, r['op'] / base['op'], r['sto'] / base['sto'], r['len'] / base['len']))
    rule = gr.build(env, env.start, env.goal, lambda a, b, c: min(c, key=lambda g: D(a, g) + D(g, b)))
    for w in WS:
        r = o.search(env, grid, rule, dedupe=True, via=w, switch='pop', timeout=60)
        out.append(('rule_route', w, r['op'] / base['op'], r['sto'] / base['sto'], r['len'] / base['len']))
    out.append(('n_walls', 0, len(env.walls), 1, 1))
    return out


def main():
    tasks = list(o.load_queries(o.DATA))
    agg = collections.defaultdict(list)
    with Pool(16) as pool:
        for part in pool.imap_unordered(work, tasks, chunksize=8):
            for k, w, *m in part:
                agg[(k, w)].append(m)
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    res = []
    print('mean walls per map', sum(x[0] for x in agg[('n_walls', 0)]) / 1000)
    for (k, w), v in sorted(agg.items()):
        if k == 'n_walls':
            continue
        r = dict(h=k, w=w, op=gm([x[0] for x in v]), sto=gm([x[1] for x in v]), len=gm([x[2] for x in v]),
                 max_len=max(x[2] for x in v), op_gt_2x=sum(x[0] > 2 for x in v))
        res.append(r)
        print(f"{k:11s} w={w:<4} op {r['op']:6.2f} sto {r['sto']:6.2f} len {r['len']:7.2f} max_len {r['max_len']:.3f}")
    (HERE / 'results' / 'closed_form_h.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
