"""Build RePE2 demonstrations from FRESH random maps (not in the evaluation set).

Maps imitate data/environment_50_30.json: 51x31 grid, per-map barrier counts and barrier lengths sampled from the
dataset's empirical distributions, integer coordinates. For each problem the exact minimal waypoint count k is
computed over every interior cell the official LLM-A* filter keeps (2 <= x <= 48, 2 <= y <= 28, not on a barrier):
  k=0 straight line clear; k=1 some cell sees start and goal; k=2 some visible pair (a in vis(S), b in vis(G));
  k=3 otherwise if a 3-waypoint chain exists over cells near barrier ends (minimality of 3 is exact).
The shortest chain with k waypoints is fed into the official LLM-A* search (tsearch, upstream accounting);
op ratio vs A* measures how efficient that chain is. Demos are picked per k with the lowest op ratio among chains
whose length is within 3% of the optimal path. Output: exp/claude/results/repe2_demos.json
"""
import sys
sys.dont_write_bytecode = True
import json, math, random
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import shapely

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import offline_improve as o           # noqa: E402
import combine_llm_wallc as cb        # noqa: E402

D = math.dist
DATA = json.loads(o.DATA.read_text())
LENGTHS = [abs(b - a) for m in DATA for k in ('horizontal_barriers', 'vertical_barriers') for _, a, b in m[k]]
NH = [len(m['horizontal_barriers']) for m in DATA]
NV = [len(m['vertical_barriers']) for m in DATA]
EVAL_BARRIERS = {(json.dumps(m['horizontal_barriers']), json.dumps(m['vertical_barriers'])) for m in DATA}


def random_map(rng):
    q = dict(range_x=[0, 51], range_y=[0, 31], horizontal_barriers=[], vertical_barriers=[])
    for key, n, fmax, lmax in (('horizontal_barriers', rng.choice(NH), 28, 50), ('vertical_barriers', rng.choice(NV), 48, 30)):
        for _ in range(n):
            fixed = rng.randint(3, fmax - 1) if key[0] == 'h' else rng.randint(3, fmax)
            length = max(1, min(rng.choice(LENGTHS), lmax))
            r = rng.random()
            lo = 0 if r < 0.25 else (lmax - length if r < 0.5 else rng.randint(1, lmax - length - 1 if lmax - length > 2 else 1))
            q[key].append([fixed, lo, lo + length])
    return q


class Vis:
    def __init__(self, env):
        self.segs = np.array(env._segments, dtype=object)

    def ok(self, P, Q):
        """vectorised: True where segment P[i]-Q[i] touches no barrier"""
        lines = shapely.linestrings(np.stack([np.asarray(P, float), np.asarray(Q, float)], axis=1))
        return ~shapely.intersects(lines[:, None], self.segs[None, :]).any(axis=1)


def solve(q):
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
    S, G = env.start, env.goal
    vis = Vis(env)
    cells = [(x, y) for x in range(2, 49) for y in range(2, 29) if env.free((x, y))]
    cells = [c for c in cells if not any(shapely.intersects(shapely.Point(c), s) for s in vis.segs)]
    C = np.array(cells, float)
    if not env.collision(S, G):
        return 0, [S, G]
    vs, vg = vis.ok([S] * len(C), C), vis.ok(C, [G] * len(C))
    one = [(D(S, c) + D(c, G), c) for c, a, b in zip(cells, vs, vg) if a and b]
    if one:
        return 1, [S, min(one)[1], G]
    A = [c for c, a in zip(cells, vs) if a]; B = [c for c, b in zip(cells, vg) if b]
    if A and B:
        pairs = list(product(A, B))
        okp = vis.ok([p[0] for p in pairs], [p[1] for p in pairs])
        two = [(D(S, a) + D(a, b) + D(b, G), a, b) for (a, b), k in zip(pairs, okp) if k]
        if two:
            _, a, b = min(two)
            return 2, [S, a, b, G]
    # 3 waypoints over cells around barrier ends (2-waypoint impossibility above is exact)
    near = sorted({(e[0] + dx, e[1] + dy) for w in env.walls for e in w for dx in range(-2, 3) for dy in range(-2, 3)}
                  & set(cells))
    nodes = [S] + near + [G]
    n = len(nodes)
    I, J = np.triu_indices(n, 1)
    okm = np.zeros((n, n), bool)
    okm[I, J] = vis.ok([nodes[i] for i in I], [nodes[j] for j in J]); okm |= okm.T
    best = {0: {0: (0.0, None)}}
    for layer in range(1, 5):
        best[layer] = {}
        for j in range(n):
            if (layer < 4 and j in (0, n - 1)) or (layer == 4 and j != n - 1):
                continue
            c = [(v[0] + D(nodes[i], nodes[j]), i) for i, v in best[layer - 1].items() if okm[i, j]]
            if c:
                best[layer][j] = min(c)
    if n - 1 not in best[4]:
        return None
    chain, j = [], n - 1
    for layer in range(4, -1, -1):
        chain.append(nodes[j]); j = best[layer][j][1]
    return 3, chain[::-1]


def work(seed):
    try:
        return _work(seed)
    except (KeyError, ValueError):   # goal unreachable on a random map, etc.
        return None


def _work(seed):
    rng = random.Random(seed)
    q = random_map(rng)
    if (json.dumps(q['horizontal_barriers']), json.dumps(q['vertical_barriers'])) in EVAL_BARRIERS:
        return None
    env0 = o.Environment(json.loads(json.dumps(dict(q, start=[1, 1], goal=[1, 1]))), collision_backend='shapely')
    free = [(x, y) for x in range(2, 49) for y in range(2, 29) if env0.free((x, y))]
    for _ in range(20):
        s, g = rng.choice(free), rng.choice(free)
        if D(s, g) >= 18:
            break
    else:
        return None
    q.update(start=list(s), goal=list(g))
    res = solve(q)
    if res is None:
        return None
    k, chain = res
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
    grid = o.Grid(env)
    base = o.search(env, grid, None)
    opt = o.search(env, grid, None, dedupe=True)['len']
    T = env.targets(chain, 'upstream')
    if T != [tuple(c) for c in chain]:
        return None
    r = cb.tsearch(env, grid, T, lambda p: D(p, env.goal))
    chain_len = sum(D(a, b) for a, b in zip(chain, chain[1:]))
    seg = shapely.LineString([s, g])
    return dict(seed=seed, query=q, k=k, chain=[list(c) for c in chain], op=r['op'] / base['op'],
                chain_over_opt=chain_len / opt, barriers=len(env.walls), opt_over_straight=opt / D(s, g),
                straight=D(s, g), min_barrier_len=min(abs(b[2] - b[1]) for kk in ('horizontal_barriers', 'vertical_barriers') for b in q[kk]),
                clearance=min(seg.distance(w) for w in env._segments))


if __name__ == '__main__':
    with Pool(16) as p:
        rows = [r for r in p.map(work, range(12000), chunksize=8) if r]
    from collections import Counter
    print('problems', len(rows), 'k dist', Counter(r['k'] for r in rows))
    # realism filters: dataset-like barrier count, no tiny barriers, long enough trip;
    # k>=1: a real detour (optimal path >= 10% longer than the straight line); k=0: the straight line passes
    # within 2 cells of a barrier, so the Round 0 check is not trivial
    def realistic(r):
        base = 5 <= r['barriers'] <= 7 and r['min_barrier_len'] >= 3 and r['straight'] >= 25
        return base and (r['clearance'] <= 2 if r['k'] == 0 else r['opt_over_straight'] >= 1.1)
    picks, used = {}, set()
    for k, count in ((0, 1), (1, 2), (2, 1), (3, 1)):
        pool = sorted([r for r in rows if r['k'] == k and r['chain_over_opt'] <= 1.03 and realistic(r)],
                      key=lambda r: r['op'])
        pool = [r for r in pool if json.dumps(r['query']['horizontal_barriers']) not in used]
        picks[k] = pool[:count]
        used |= {json.dumps(r['query']['horizontal_barriers']) for r in picks[k]}
        print('k', k, 'eligible', len(pool), 'picked op', [round(100 * r['op'], 1) for r in picks[k]],
              'eligible median op', round(100 * pool[len(pool) // 2]['op'], 1) if pool else None)
    out = [r for k in (0, 1, 2, 3) for r in picks[k]]
    (HERE / 'results' / 'repe2_demos.json').write_text(json.dumps(out, indent=1))
    for r in out:
        print(r['k'], r['query'], r['chain'], 'op %.1f chain/opt %.3f' % (100 * r['op'], r['chain_over_opt']))
