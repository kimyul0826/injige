"""Dynamic weighted A* (paper baseline: initial weight 2, decay 0.99) in the same upstream accounting.
The paper does not define the schedule exactly; two readings, weight updated per expansion t:
  mult : w_t = max(1, 2 * 0.99^t)      plus : w_t = 1 + 0.99^t
f is computed with the weight at push time (heap not re-sorted). Fixed wA* w=1.5/2 for reference."""
import sys
sys.dont_write_bytecode = True
import heapq, json, math
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o   # noqa: E402
D, INF = math.dist, math.inf


def dwa(env, grid, wfun):
    start, goal = env.start, env.goal
    g, parent = {start: 0.0}, {start: start}
    q, closed, ops = [(2.0 * D(start, goal), start)], set(), 0
    while q:
        s = heapq.heappop(q)[-1]
        ops += 1; closed.add(s)
        if s == goal:
            break
        w = wfun(ops)
        for dx, dy in o.MOTIONS:
            n = (s[0] + dx, s[1] + dy)
            if n in closed:
                continue
            g.setdefault(n, INF)
            if grid.coll(s, n):
                continue
            new = g[s] + D(s, n)
            if new < g[n]:
                g[n], parent[n] = new, s
                heapq.heappush(q, (new + w * D(n, goal), n))
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return dict(op=ops, sto=len(g), len=sum(D(a, b) for a, b in zip(path, path[1:])))


def work(q):
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely'); grid = o.Grid(env)
    base = o.search(env, grid, None)
    rs = dict(mult=dwa(env, grid, lambda t: max(1.0, 2 * 0.99 ** t)), plus=dwa(env, grid, lambda t: 1 + 0.99 ** t),
              w15=o.search(env, grid, None, wg=1.5), w2=o.search(env, grid, None, wg=2.0))
    return {k: (r['op'] / base['op'], r['sto'] / base['sto'], r['len'] / base['len']) for k, r in rs.items()}


def main():
    qs = [q for _, q in o.load_queries(o.DATA)]
    with Pool(10) as pool:
        rows = pool.map(work, qs, chunksize=8)
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    res = {k: dict(op=gm([r[k][0] for r in rows]), sto=gm([r[k][1] for r in rows]), len=gm([r[k][2] for r in rows]))
           for k in rows[0]}
    for k, r in res.items():
        print(f"{k:5} op {r['op']:6.2f} sto {r['sto']:6.2f} len {r['len']:7.2f}")
    (HERE / 'results' / 'dwa_check.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
