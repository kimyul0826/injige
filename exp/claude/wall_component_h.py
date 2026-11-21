"""Touching / overlapping walls: treat connected walls as ONE obstacle (no pre-search, per-node closed form).

wall1 (closed_form_h) goes around each wall segment separately via its two endpoints. When walls touch
(L, T, U shapes, or a wall attached to the map border) an endpoint may be unpassable and the real detour
goes around the whole connected structure.

wallC: union walls that intersect into components (border-attached components are flagged).
For every component crossed by segment s->goal, detour = shortest way around the component's convex hull:
  min over hull vertices a (visible from s), b (visible from goal), both walking directions of
  d(s,a) + hull-boundary(a->b) + d(b,goal); walks through hull vertices that lie on the map border are
  forbidden (cannot pass between a wall and the border).
If s or goal is inside the hull (in a pocket), fall back to the wall1 bound of that component.
h = max(euclid, max over components). For s, goal outside the hull this is the exact detour around one
connected obstacle, hence admissible; max of admissible bounds is admissible.
Per-node cost: O(sum of hull sizes of crossed components) small geometric tests; cached per node.
"""
import sys
sys.dont_write_bytecode = True
import collections, json, math
from multiprocessing import Pool
from pathlib import Path

from shapely.geometry import LineString, MultiLineString, Point, Polygon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import offline_improve as o           # noqa: E402
import closed_form_h as cf            # noqa: E402
from llm_astar.search import intersects   # noqa: E402

D = math.dist


def components(env):
    W = env.walls
    parent = list(range(len(W)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(W)):
        for j in range(i + 1, len(W)):
            if intersects(W[i][0], W[i][1], W[j][0], W[j][1]):
                parent[find(i)] = find(j)
    groups = collections.defaultdict(list)
    for i in range(len(W)):
        groups[find(i)].append(W[i])
    return list(groups.values())


def make_wallC(env):
    goal = env.goal
    x0, x1, y0, y1 = env.rx[0], env.rx[1] - 1, env.ry[0], env.ry[1] - 1
    on_border = lambda p: p[0] <= x0 or p[0] >= x1 or p[1] <= y0 or p[1] >= y1
    comps = []
    for segs in components(env):
        hull = MultiLineString([list(t) for t in segs]).convex_hull
        # per-segment passable ends: only map-border ends are excluded (crossing a segment's supporting line
        # beyond a wall-touching end is still possible further out, so that end remains a valid lower bound)
        c = dict(segs=[(a, b, [e for e in (a, b) if not on_border(e)]) for a, b in segs])
        if isinstance(hull, Polygon):
            ring = list(hull.exterior.coords)[:-1]
            if hull.exterior.is_ccw is False:
                ring.reverse()
            c.update(poly=hull, core=hull.buffer(-1e-6), V=[tuple(v) for v in ring],
                     border=[on_border(v) for v in ring])
            pre = [0.0]
            for i in range(len(ring)):
                pre.append(pre[-1] + D(ring[i], ring[(i + 1) % len(ring)]))
            c['pre'] = pre
        comps.append(c)

    def crosses(s, c):
        return any(intersects(s, goal, a, b) for a, b, _ in c['segs'])

    def wall1_bound(s, c):
        # max over crossed segments of the single-segment detour (= wall1 restricted to this component)
        v = D(s, goal)
        for a, b, ends in c['segs']:
            if ends and intersects(s, goal, a, b):
                v = max(v, min(D(s, e) + D(e, goal) for e in ends))
        return v

    def hull_bound(s, c):
        poly, core, V, B, pre = c['poly'], c['core'], c['V'], c['border'], c['pre']
        if poly.contains(Point(s)) or poly.contains(Point(goal)):
            return wall1_bound(s, c)
        n = len(V)
        vis = lambda p: [i for i in range(n) if not LineString([p, V[i]]).intersects(core)]
        A, Bv = vis(s), vis(goal)
        best = math.inf
        for a in A:
            for b in Bv:
                for step in (1, -1):
                    # walk a -> b along the ring in direction `step`; forbid border vertices on the way
                    i, ok = a, not B[a]
                    while ok and i != b:
                        i = (i + step) % n
                        ok = not B[i]
                    if not ok:
                        continue
                    if step == 1:
                        walk = pre[b] - pre[a] if b >= a else pre[n] - pre[a] + pre[b]
                    else:
                        walk = pre[a] - pre[b] if a >= b else pre[n] - pre[b] + pre[a]
                    best = min(best, D(s, V[a]) + walk + D(V[b], goal))
        return max(best, wall1_bound(s, c)) if best < math.inf else wall1_bound(s, c)

    cache = {}

    def h(s):
        v = cache.get(s)
        if v is not None:
            return v
        v = D(s, goal)
        for c in comps:
            if crosses(s, c):
                v = max(v, hull_bound(s, c) if 'poly' in c else wall1_bound(s, c))
        cache[s] = v
        return v
    return h


def astar_reopen(env, grid, h, w):
    import heapq
    start, goal = env.start, env.goal
    g, parent = {start: 0.0}, {start: start}
    q, closed, ops = [(w * h(start), start)], set(), 0
    while q:
        f, s = heapq.heappop(q)
        if s in closed:
            continue
        ops += 1
        closed.add(s)
        if s == goal:
            break
        for dx, dy in o.MOTIONS:
            n = (s[0] + dx, s[1] + dy)
            if grid.coll(s, n):
                continue
            ng = g[s] + D(s, n)
            if ng < g.get(n, math.inf):
                g[n], parent[n] = ng, s
                closed.discard(n)
                heapq.heappush(q, (ng + w * h(n), n))
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return dict(op=ops, sto=len(g), len=sum(D(a, b) for a, b in zip(path, path[1:])))


FAMS = ('original', 'dense', 'rooms', 'utrap', 'maze')
CFG = [('euclid', 2.0), ('wall1', 1.0), ('wallC', 1.0), ('wall1', 1.5), ('wallC', 1.5), ('wall1', 2.0), ('wallC', 2.0)]


def work(args):
    fam, q = args
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
    grid = o.Grid(env)
    base = cf.astar(env, grid, cf.make_h(env, 'euclid'), 1.0)
    out = []
    for k, w in CFG:
        h = make_wallC(env) if k == 'wallC' else cf.make_h(env, k)
        r = astar_reopen(env, grid, h, w) if k == 'wallC' else cf.astar(env, grid, h, w)
        out.append((k, w, r['op'] / base['op'], r['len'] / base['len']))
    return fam, out


def main():
    orig = [q for _, q in o.load_queries(o.DATA)]
    tasks = [('original', q) for q in orig] + \
            [(f, q) for f in FAMS[1:] for q in json.loads((HERE / 'bench' / f'{f}.json').read_text())]
    agg = collections.defaultdict(list)
    with Pool(16) as pool:
        for fam, rows in pool.imap_unordered(work, tasks, chunksize=2):
            for k, w, op, ln in rows:
                agg[(fam, k, w)].append((op, ln))
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    res = {}
    print('family    ' + ''.join(f'{k}{w}'.rjust(16) for k, w in CFG))
    for f in FAMS:
        cells = []
        for k, w in CFG:
            v = agg[(f, k, w)]
            r = dict(op=gm([x[0] for x in v]), len=gm([x[1] for x in v]), max_len=max(x[1] for x in v))
            res[f'{f}|{k}|{w}'] = r
            cells.append(f"{r['op']:6.1f}/{r['len']:6.2f}".rjust(16))
        print(f'{f:9s} ' + ''.join(cells))
    for f in FAMS:
        print(f, 'w=1 max_len wall1 %.4f wallC %.4f' % (res[f'{f}|wall1|1.0']['max_len'], res[f'{f}|wallC|1.0']['max_len']))
    (HERE / 'results' / 'wall_component_h.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
