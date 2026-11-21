"""Random waypoints inside the start-goal bounding rectangle, run through the official LLM-A* search.
k = 1, 2, 3 waypoints; R draws per query (fixed seeds). Waypoints are drawn only from cells that pass the
official filter (so exactly k survive) and are free; ordered by projection on the start->goal direction
('sorted', like an LLM path) and also in draw order ('unsorted').
Reference: stored Llama-3 LLM-A* grouped by the number of waypoints that survived the official filter.
Same-query wA*2 is reported for every group. Upstream accounting, 1000 original queries."""
import sys
sys.dont_write_bytecode = True
import collections, json, math, random
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o   # noqa: E402

R = 5
KS = (1, 2, 3)


def work(args):
    sample, q, texts = args
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely'); grid = o.Grid(env)
    base = o.search(env, grid, None); wa = o.search(env, grid, None, wg=2.0)
    S, G = env.start, env.goal
    obs = o.obstacle_set(env)
    cand = [(x, y) for x in range(min(S[0], G[0]), max(S[0], G[0]) + 1)
            for y in range(min(S[1], G[1]), max(S[1], G[1]) + 1)
            if (x, y) not in (S, G) and o.upstream_ok(env, (x, y), obs) and env.free((x, y))]
    ux, uy = G[0] - S[0], G[1] - S[1]
    proj = lambda p: (p[0] - S[0]) * ux + (p[1] - S[1]) * uy
    rows = []
    rel = lambda r: (r['op'] / base['op'], r['len'] / base['len'], wa['op'] / base['op'])
    for k in KS:
        for d in range(R):
            rng = random.Random(f'{sample}-{k}-{d}')
            pts = rng.sample(cand, min(k, len(cand)))
            for order, P in (('sorted', sorted(pts, key=proj)), ('unsorted', pts)):
                T = env.targets(P, 'upstream')
                assert len(T) - 2 == len(P)
                valid = all(not env.collision(a, b) for a, b in zip(T, T[1:]))
                rows.append((f'random k={k} {order}', *rel(o.search(env, grid, T)), valid))
    for p in o.PROMPTS:
        try: nodes = o.parse_path(texts[p], mode='upstream')
        except ValueError: nodes = []
        T = env.targets(nodes, 'upstream'); n = len(T) - 2
        valid = all(not env.collision(a, b) for a, b in zip(T, T[1:]))
        rows.append((f'LLM {p} k={min(n, 4) if n < 4 else ">=4"}', *rel(o.search(env, grid, T)), valid))
    return rows


def main():
    texts, _ = o.load_texts()
    tasks = [(s, q, {p: texts[(s, p)] for p in o.PROMPTS}) for s, q in o.load_queries(o.DATA)]
    agg = collections.defaultdict(list)
    with Pool(10) as pool:
        for rows in pool.imap_unordered(work, tasks, chunksize=4):
            for name, *m in rows:
                agg[name].append(m)
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    res = {}
    for name in sorted(agg):
        v = agg[name]
        r = dict(n=len(v), op=gm([x[0] for x in v]), len=gm([x[1] for x in v]), wA2_op=gm([x[2] for x in v]),
                 worse_than_Astar_pct=100 * sum(x[0] > 1 for x in v) / len(v),
                 better_than_wA2_pct=100 * sum(x[0] < x[2] for x in v) / len(v),
                 worse_than_wA2_pct=100 * sum(x[0] > x[2] for x in v) / len(v),
                 gt2x_pct=100 * sum(x[0] > 2 for x in v) / len(v),
                 valid_chain_pct=100 * sum(x[3] for x in v) / len(v))
        res[name] = r
        print(f"{name:26s} n {r['n']:5d} | op {r['op']:6.2f} len {r['len']:6.2f} | same-query wA*2 {r['wA2_op']:6.2f} | "
              f">A* {r['worse_than_Astar_pct']:5.1f}%  <wA2 {r['better_than_wA2_pct']:5.1f}%  >wA2 {r['worse_than_wA2_pct']:5.1f}%  "
              f">2x {r['gt2x_pct']:5.1f}%  valid {r['valid_chain_pct']:5.1f}%")
    (HERE / 'results' / 'random_waypoints.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
