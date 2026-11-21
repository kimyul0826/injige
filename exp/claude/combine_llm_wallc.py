"""Combine stored Llama-3 LLM-A* waypoints with the wallC geometric heuristic (offline, no inference).

Data: 1000 original 50x30 queries, stored V2 responses (standard / cot / repe). Upstream accounting
(duplicate pops counted, same as Table 1 reproduction). Ratios = geometric mean vs A* (100 = A*).

Waypoint processing
  raw  : official filter (env.targets, upstream)
  vp   : verified prefix -- keep waypoints from the front while the straight segment from the previous
         kept point is collision-free; stop at the first colliding segment; goal appended.
  full : chain used only if EVERY segment (incl. last -> goal) is collision-free, else no waypoints.
  oracle: string-pulled turning points of the optimal A* path (upper reference, not an LLM).

Methods
  LLM-A*            f = g + d(s,goal) + d(s,target)            (official, target switching)
  LLM-A*+wallC      f = g + wallC(s) + d(s,target)             (goal term replaced by wallC)
  wallC             f = g + w*wallC(s)                          (no LLM)
  wallC+visroute    f = g + w*max(wallC(s), route(s))           route(s) = min over chain vertices t_k VISIBLE
                    from s of d(s,t_k) + chain(t_k..goal); non-committing (no target state). Without the
                    visibility condition route collapses to d(s,goal) (goal is a chain vertex) -- first run showed this.
"""
import sys
sys.dont_write_bytecode = True
import collections, heapq, json, math, time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o           # noqa: E402
import wall_component_h as wc         # noqa: E402
import llmastar_vs_wallc as lv        # noqa: E402
import os
if os.environ.get('LLM_SRC'):          # read-only alternative response set, e.g. GPT-4o mini batch run
    o.SRC = Path(os.environ['LLM_SRC'])
TAG = os.environ.get('TAG', '')

D, INF = math.dist, math.inf


def tsearch(env, grid, T, hg):
    """Copy of o.search upstream semantics (neighbor target switch, no dedupe) with goal term hg(p)."""
    start, goal = env.start, env.goal
    idx = 1
    f = lambda p: g[p] + (hg(p) + D(p, T[idx]))
    g, parent = {start: 0.0}, {start: start}
    q, closed, ops = [(f(start), start)], set(), 0
    while q:
        s = heapq.heappop(q)[-1]
        ops += 1
        closed.add(s)
        if s == goal:
            break
        for dx, dy in o.MOTIONS:
            n = (s[0] + dx, s[1] + dy)
            if T and n == T[idx] and n != goal:
                idx = min(idx + 1, len(T) - 1)
                q = [(f(e[-1]), e[-1]) for e in q]
                heapq.heapify(q)
            if n in closed:
                continue
            if grid.coll(s, n):
                g.setdefault(n, INF)
                continue
            g.setdefault(n, INF)
            new = g[s] + D(s, n)
            if new < g[n]:
                g[n], parent[n] = new, s
                heapq.heappush(q, (f(n), n))
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return dict(op=ops, sto=len(g), len=sum(D(a, b) for a, b in zip(path, path[1:])))


def verified_prefix(env, T):
    out = [T[0]]
    for p in T[1:-1]:
        if env.collision(out[-1], p):
            break
        out.append(p)
    return out + [T[-1]]


def fully_valid(env, T):
    return T if all(not env.collision(a, b) for a, b in zip(T, T[1:])) else [T[0], T[-1]]


def oracle_chain(env, grid):
    opt = o.search(env, grid, None, dedupe=True)['path']
    turns = [opt[i] for i in range(1, len(opt) - 1)
             if (opt[i][0] - opt[i-1][0], opt[i][1] - opt[i-1][1]) != (opt[i+1][0] - opt[i][0], opt[i+1][1] - opt[i][1])]
    return o.string_pull(env, [env.start] + turns + [env.goal])


def make_route(env, T):
    """Upper-bound route via the chain: shortest way to ANY chain vertex visible from s (straight,
    collision-free), then along the chain to the goal. Without the visibility condition the goal itself is
    always a rejoin point and the term collapses to d(s,goal). No visible vertex -> 0 (term inactive)."""
    chain = [0.0] * len(T)
    for k in range(len(T) - 2, -1, -1):
        chain[k] = chain[k + 1] + D(T[k], T[k + 1])
    pts = list(zip(T[1:], chain[1:]))
    def route(s):
        v = [D(s, t) + c for t, c in pts if not env.collision(s, t)]
        return min(v) if v else 0.0
    return route


def combined(hw, route):
    return lambda s: max(hw(s), route(s))


def work(args):
    sample, query, texts = args
    env = o.Environment(json.loads(json.dumps(query)), collision_backend='shapely')
    grid = o.Grid(env)
    base = o.search(env, grid, None)
    out = []

    def rec(name, r, extra=None):
        out.append((name, r['op'] / base['op'], r['len'] / base['len'], extra))

    rec('wA* w=2', o.search(env, grid, None, wg=2.0))
    hw = wc.make_wallC(env)
    for w in (1.0, 1.5):
        rec(f'wallC w={w}', lv.astar_h(env, grid, hw, w, True))
    chains = {}
    for p in o.PROMPTS:
        try:
            nodes = o.parse_path(texts[p], mode='upstream')
        except ValueError:
            nodes = []
        raw = env.targets(nodes, 'upstream')
        chains[p] = dict(raw=raw, vp=verified_prefix(env, raw), full=fully_valid(env, raw))
    chains['oracle'] = dict(raw=oracle_chain(env, grid))
    for p, C in chains.items():
        for kind, T in C.items():
            tag = f'{p}|{kind}'
            n_wp = len(T) - 2
            if p != 'oracle' or kind == 'raw':
                if p != 'oracle':
                    rec(f'LLM-A* {tag}', tsearch(env, grid, T, lambda s: D(s, env.goal)), n_wp)
                rec(f'LLM-A*+wallC {tag}', tsearch(env, grid, T, hw), n_wp)
            route = make_route(env, T)
            for w in (1.0, 1.5):
                rec(f'wallC+visroute w={w} {tag}', lv.astar_h(env, grid, combined(hw, route), w, True), n_wp)
    return out


def main():
    texts, _ = o.load_texts()
    tasks = [(s, q, {p: texts[(s, p)] for p in o.PROMPTS}) for s, q in o.load_queries(o.DATA)]
    agg = collections.defaultdict(list)
    t0 = time.time()
    with Pool(10) as pool:
        for part in pool.imap_unordered(work, tasks, chunksize=4):
            for name, op, ln, nwp in part:
                agg[name].append((op, ln, nwp))
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    res = {}
    for name in sorted(agg):
        v = agg[name]
        r = dict(n=len(v), op=gm([x[0] for x in v]), len=gm([x[1] for x in v]), max_len=100 * max(x[1] for x in v),
                 op_gt_2x=sum(x[0] > 2 for x in v),
                 mean_wp=(sum(x[2] for x in v) / len(v)) if v[0][2] is not None else None)
        res[name] = r
    wa = agg['wA* w=2']
    for name in res:
        res[name]['beats_wA2_pct'] = 100 * sum(x[0] < y[0] for x, y in zip(agg[name], wa)) / len(wa)
    for name, r in res.items():
        wp = f"wp {r['mean_wp']:.2f}" if r['mean_wp'] is not None else ''
        print(f"{name:42s} op {r['op']:6.2f} len {r['len']:7.2f} max_len {r['max_len']:7.2f} "
              f"op>2x {r['op_gt_2x']:4d} beat_wA2 {r['beats_wA2_pct']:5.1f}%  {wp}")
    print(f'elapsed {time.time() - t0:.0f}s')
    (HERE / 'results' / f'combine_llm_wallc{TAG}.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
