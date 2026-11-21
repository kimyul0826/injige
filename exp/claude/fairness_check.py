"""Fairness check for wallC vs weighted A* on the 1000 original queries (upstream accounting).
1) same search function: astar_h(euclid, w) must reproduce o.search(wg=w)  -> implementation parity
2) same weight: wallC vs euclid at w = 1, 1.5, 2
3) path validity of wallC paths (every step collision-free under env.collision)
4) wall-clock per search (heuristic cost included), single process"""
import sys
sys.dont_write_bytecode = True
import json, math, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o            # noqa: E402
import wall_component_h as wc          # noqa: E402
import llmastar_vs_wallc as lv         # noqa: E402
import heapq
D, INF = math.dist, math.inf

def astar_path(env, grid, h, w):
    start, goal = env.start, env.goal
    g, parent = {start: 0.0}, {start: start}
    q, closed = [(w * h(start), start)], set()
    while q:
        _, s = heapq.heappop(q)
        if s == goal: break
        if s in closed: continue
        closed.add(s)
        for dx, dy in o.MOTIONS:
            n = (s[0] + dx, s[1] + dy)
            if grid.coll(s, n): continue
            ng = g[s] + D(s, n)
            if ng < g.get(n, INF):
                g[n], parent[n] = ng, s; closed.discard(n)
                heapq.heappush(q, (ng + w * h(n), n))
    path = [goal]
    while path[-1] != start: path.append(parent[path[-1]])
    return path[::-1]

def main():
    qs = list(o.load_queries(o.DATA))
    acc = {}
    add = lambda k, v: acc.setdefault(k, []).append(v)
    bad = 0
    for i, (s, q) in enumerate(qs):
        env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
        grid = o.Grid(env)
        base = o.search(env, grid, None)
        eu = lambda p: D(p, env.goal)
        r = o.search(env, grid, None, wg=2.0); add('o.search wA* w=2', (r['op'] / base['op'], r['len'] / base['len'], 0))
        for name, hf in (('euclid', lambda: eu), ('wallC', lambda: wc.make_wallC(env))):
            for w in (1.0, 1.5, 2.0):
                h = hf()
                t = time.perf_counter(); r = lv.astar_h(env, grid, h, w, True); dt = time.perf_counter() - t
                add(f'astar_h {name} w={w}', (r['op'] / base['op'], r['len'] / base['len'], dt))
        p = astar_path(env, grid, wc.make_wallC(env), 1.5)
        bad += any(env.collision(a, b) for a, b in zip(p, p[1:])) or p[0] != env.start or p[-1] != env.goal
        t = time.perf_counter(); o.search(env, grid, None); add('A* time', (1, 1, time.perf_counter() - t))
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    res = {}
    print(f"{'method':26} {'op':>7} {'len':>8} {'ms/search':>10}")
    for k, v in acc.items():
        res[k] = dict(op=gm([a for a, _, _ in v]), len=gm([b for _, b, _ in v]), ms=1000 * sum(c for *_, c in v) / len(v))
        print(f"{k:26} {res[k]['op']:7.2f} {res[k]['len']:8.2f} {res[k]['ms']:10.2f}")
    print('wallC w=1.5 invalid paths:', bad, '/', len(qs))
    res['invalid_wallC_paths'] = bad
    (HERE / 'results' / 'fairness_check.json').write_text(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
