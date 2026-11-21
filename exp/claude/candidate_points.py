"""B-plan feasibility: label free cells next to wall ends as candidate waypoints.

Candidates: for every wall endpoint not on the map border, the 4 diagonal neighbours that are inside the
map interior, not on any wall, and not adjacent-blocked; deduplicated. Reports candidate counts and the
ceiling of 'LLM picks candidates': shortest collision-free chain S -> candidates -> G on the candidate
visibility graph (Dijkstra), fed into wallC+visroute w=1.5 (full gating). No LLM, no API.
"""
import sys
sys.dont_write_bytecode = True
import heapq, json, math, statistics
from multiprocessing import Pool
from pathlib import Path

from shapely.geometry import Point

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o           # noqa: E402
import wall_component_h as wc         # noqa: E402
import llmastar_vs_wallc as lv        # noqa: E402
import combine_llm_wallc as cb        # noqa: E402

D = math.dist



def candidates(env):
    x0, x1, y0, y1 = env.rx[0] + 1, env.rx[1] - 2, env.ry[0] + 1, env.ry[1] - 2
    border = lambda p: p[0] <= env.rx[0] or p[0] >= env.rx[1] - 1 or p[1] <= env.ry[0] or p[1] >= env.ry[1] - 1
    out = []
    for a, b in env.walls:
        for e in (tuple(a), tuple(b)):
            if border(e):
                continue
            for dx in (-1, 1):
                for dy in (-1, 1):
                    c = (e[0] + dx, e[1] + dy)
                    if not (x0 <= c[0] <= x1 and y0 <= c[1] <= y1):
                        continue
                    if c in out or any(Point(c).intersects(w) for w in env._segments):
                        continue
                    out.append(c)
    return sorted(out)


def best_chain(env, C):
    nodes = [env.start] + C + [env.goal]
    n = len(nodes)
    dist = [math.inf] * n
    prev = [-1] * n
    dist[0] = 0.0
    q = [(0.0, 0)]
    while q:
        d, i = heapq.heappop(q)
        if d > dist[i]:
            continue
        if i == n - 1:
            break
        for j in range(1, n):
            if j == i:
                continue
            nd = d + D(nodes[i], nodes[j])
            if nd < dist[j] and not env.collision(nodes[i], nodes[j]):
                dist[j], prev[j] = nd, i
                heapq.heappush(q, (nd, j))
    if dist[-1] == math.inf:
        return [env.start, env.goal]
    path, k = [], n - 1
    while k != -1:
        path.append(nodes[k])
        k = prev[k]
    return path[::-1]


def work(q):
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
    grid = o.Grid(env)
    base = o.search(env, grid, None)
    C = candidates(env)
    T = best_chain(env, C)
    hw = wc.make_wallC(env)
    r0 = lv.astar_h(env, grid, hw, 1.5, True)
    r1 = lv.astar_h(env, grid, cb.combined(hw, cb.make_route(env, T)), 1.5, True)
    return len(C), len(T) - 2, r0['op'] / base['op'], r1['op'] / base['op'], r1['len'] / base['len']


if __name__ == '__main__':
    qs = [q for _, q in o.load_queries(o.DATA)]
    with Pool(16) as p:
        rows = p.map(work, qs, chunksize=4)
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    nc = [r[0] for r in rows]; nt = [r[1] for r in rows]
    res = dict(n=len(rows), cand_mean=statistics.mean(nc), cand_median=statistics.median(nc), cand_max=max(nc),
               chain_wp_mean=statistics.mean(nt), wallC_op=gm([r[2] for r in rows]),
               cand_oracle_op=gm([r[3] for r in rows]), cand_oracle_len=gm([r[4] for r in rows]),
               cand_oracle_maxlen=100 * max(r[4] for r in rows))
    print(json.dumps(res, indent=2))
    (HERE / 'results' / 'candidate_points.json').write_text(json.dumps(res, indent=2))
