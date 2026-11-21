"""User idea: shrink the map, run A* on the coarse grid, restore the path to full scale, pick waypoints
from the restored path, then run the full-scale search guided by those waypoints.

Coarse grid factor k: coarse cell c covers fine cells [c*k, c*k+k). Node = cell centre (fine coords).
Coarse edge allowed iff the fine straight segment between the two centres is collision-free.
Waypoints = turning points of the restored path, string-pulled (start/goal swapped in), at most 3.
Fine search: route heuristic f = g + w*(d(s,t_k) + chain), switch on pop, timeout 60; if the chain
collides, weighted A* w.
Cost reported honestly: total op = coarse pops + fine pops (also sto = coarse nodes + fine nodes).
Collision checks of the coarse search and string pull are counted separately (coll_checks).
"""
import sys
sys.dont_write_bytecode = True
import collections, heapq, json, math
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o      # noqa: E402
import closed_form_h as cf       # noqa: E402

D = math.dist
INF = math.inf
KS = (2, 3, 5)
WS = (1.5, 2.0)


def coarse_path(env, k):
    centre = lambda c: (c[0] * k + (k - 1) / 2, c[1] * k + (k - 1) / 2)
    cell = lambda p: (p[0] // k, p[1] // k)
    cs, cg = cell(env.start), cell(env.goal)
    pt = lambda c: env.start if c == cs else env.goal if c == cg else centre(c)
    g, parent = {cs: 0.0}, {cs: cs}
    q, closed, ops, checks = [(D(pt(cs), env.goal), cs)], set(), 0, 0
    while q:
        _, c = heapq.heappop(q)
        if c in closed:
            continue
        ops += 1
        closed.add(c)
        if c == cg:
            break
        for dx, dy in o.MOTIONS:
            n = (c[0] + dx, c[1] + dy)
            p = pt(n)
            if n in closed or not env.inside((round(p[0]), round(p[1]))):
                continue
            checks += 1
            if env.collision(pt(c), p):
                continue
            ng = g[c] + D(pt(c), p)
            if ng < g.get(n, INF):
                g[n], parent[n] = ng, c
                heapq.heappush(q, (ng + D(p, env.goal), n))
    if cg not in closed:
        return None, ops, len(g), checks
    path = [cg]
    while path[-1] != cs:
        path.append(parent[path[-1]])
    return [pt(c) for c in reversed(path)], ops, len(g), checks


def waypoints(env, P):
    # snap to integer free cells, then keep turning points and string-pull with exact collision checks
    snap = [env.start] + [(round(x), round(y)) for x, y in P[1:-1]] + [env.goal]
    turns = [snap[i] for i in range(1, len(snap) - 1)
             if (snap[i][0] - snap[i-1][0], snap[i][1] - snap[i-1][1]) != (snap[i+1][0] - snap[i][0], snap[i+1][1] - snap[i][1])]
    pulled = o.string_pull(env, [env.start] + turns + [env.goal])
    return [env.start] + pulled[1:-1][:3] + [env.goal]


def work(args):
    sample, query = args
    env = o.Environment(json.loads(json.dumps(query)), collision_backend='shapely')
    grid = o.Grid(env)
    base = o.search(env, grid, None, dedupe=True)
    out = []
    for w in WS:
        r = o.search(env, grid, None, dedupe=True, wg=w)
        out.append(('wA*', w, r['op'] / base['op'], r['sto'] / base['sto'], r['len'] / base['len'], 0, 1))
        r = cf.astar(env, grid, cf.make_h(env, 'wall1'), w)
        out.append(('wall1', w, r['op'] / base['op'], r['sto'] / base['sto'], r['len'] / base['len'], 0, 1))
    for k in KS:
        P, cops, csto, checks = coarse_path(env, k)
        T = waypoints(env, P) if P else None
        valid = bool(T) and all(not env.collision(a, b) for a, b in zip(T, T[1:]))
        for w in WS:
            r = o.search(env, grid, T, dedupe=True, via=w, switch='pop', timeout=60) if valid else \
                o.search(env, grid, None, dedupe=True, wg=w)
            fine = (r['op'] / base['op'], r['sto'] / base['sto'], r['len'] / base['len'])
            out.append((f'coarse{k} fine-only', w, *fine, checks, valid))
            out.append((f'coarse{k} total', w, (r['op'] + cops) / base['op'], (r['sto'] + csto) / base['sto'],
                        fine[2], checks, valid))
    return out


def main():
    tasks = list(o.load_queries(o.DATA))
    agg = collections.defaultdict(list)
    with Pool(16) as pool:
        for part in pool.imap_unordered(work, tasks, chunksize=8):
            for name, w, *m in part:
                agg[(name, w)].append(m)
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    res = []
    for (name, w), v in sorted(agg.items()):
        r = dict(method=name, w=w, op=gm([x[0] for x in v]), sto=gm([x[1] for x in v]), len=gm([x[2] for x in v]),
                 coll_checks=sum(x[3] for x in v) / len(v), valid=sum(bool(x[4]) for x in v) / 10)
        res.append(r)
        print(f"{name:20s} w={w:<4} op {r['op']:6.2f} sto {r['sto']:6.2f} len {r['len']:7.2f} "
              f"coarse_checks {r['coll_checks']:6.1f} valid {r['valid']:5.1f}%")
    (HERE / 'results' / 'coarse_then_fine.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
