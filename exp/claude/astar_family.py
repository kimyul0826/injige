"""A*-based ways to use waypoints (no inference), with Llama waypoints and oracle waypoints.

  subgoal  : run A* separately between consecutive waypoints (unreachable waypoints skipped)
  tiebreak : plain admissible A* (optimal length guaranteed); ties in f broken by the waypoint route
             estimate d(n, t_k) + remaining chain
Reference rows (LLM-A* official / progress, MHA*, E-graph) are in logic_per_sample.jsonl.gz.
"""
import sys
sys.dont_write_bytecode = True
import collections, heapq, json, math
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o      # noqa: E402
import logic_variants as lv      # noqa: E402

D = math.dist
INF = math.inf


def subgoal(env, grid, T):
    ops, stored, path = 0, set(), [T[0]]
    cur = T[0]
    for t in T[1:]:
        if t == cur:
            continue
        r = o.search(env, grid, None, start=cur, goal=t, dedupe=True)
        ops += r['op']
        if not r['found']:
            continue  # skip unreachable waypoint
        stored |= set()  # storage approximated below
        path += r['path'][1:]
        stored_n = r['sto']
        stored.add((cur, t, stored_n))
        cur = t
    if cur != env.goal:  # last target unreachable: finish with A*
        r = o.search(env, grid, None, start=cur, goal=env.goal, dedupe=True)
        ops += r['op']; path += r['path'][1:]; stored.add((cur, env.goal, r['sto']))
    return dict(op=ops, sto=max(s for *_x, s in stored), len=sum(D(a, b) for a, b in zip(path, path[1:])))


def tiebreak(env, grid, T):
    start, goal = env.start, env.goal
    route = lv.Route(T, goal)
    g, parent = {start: 0.0}, {start: start}
    q = [(D(start, goal), route.h(start), start)]
    closed = set()
    ops = 0
    while q:
        f, _, s = heapq.heappop(q)
        if s in closed:
            continue
        ops += 1
        closed.add(s)
        if s == goal:
            return lv.finish(parent, start, goal, ops, g)
        if route.update(s):
            q = [(g[p] + D(p, goal), route.h(p), p) for _, _, p in q]
            heapq.heapify(q)
        for n, c in lv.neighbours(grid, s):
            if n in closed:
                continue
            new = g[s] + c
            if new < g.get(n, INF):
                g[n], parent[n] = new, s
                heapq.heappush(q, (new + D(n, goal), route.h(n), n))
    raise RuntimeError('no path')


def oracle_T(env, grid):
    base = o.search(env, grid, None, dedupe=True)
    opt = base['path']
    turns = [opt[i] for i in range(1, len(opt) - 1)
             if (opt[i][0] - opt[i-1][0], opt[i][1] - opt[i-1][1]) != (opt[i+1][0] - opt[i][0], opt[i+1][1] - opt[i][1])]
    return base, [env.start] + o.string_pull(env, [env.start] + turns + [env.goal])[1:-1][:3] + [env.goal]


def work(args):
    sample, query, texts = args
    env = o.Environment(json.loads(json.dumps(query)), collision_backend='shapely')
    grid = o.Grid(env)
    base, orc = oracle_T(env, grid)
    chains = {'oracle': orc, 'noLLM': [env.start, env.goal]}
    for p in o.PROMPTS:
        try:
            nodes = o.parse_path(texts[p], mode='upstream')
        except ValueError:
            nodes = []
        chains[p] = env.targets(nodes, 'upstream')
    out = []
    for k, T in chains.items():
        for name, fn in (('subgoal', subgoal), ('tiebreak', tiebreak)):
            r = fn(env, grid, T)
            out.append((name, k, r['op'] / base['op'], r['sto'] / base['sto'], r['len'] / base['len']))
    return out


def main():
    texts, _ = o.load_texts()
    tasks = [(s, q, {p: texts[(s, p)] for p in o.PROMPTS}) for s, q in o.load_queries(o.DATA)]
    agg = collections.defaultdict(list)
    with Pool(16) as pool:
        for part in pool.imap_unordered(work, tasks, chunksize=4):
            for name, k, *m in part:
                agg[(name, k)].append(m)
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    res = []
    for (name, k), v in sorted(agg.items()):
        r = dict(method=name, waypoints=k, op=gm([x[0] for x in v]), sto=gm([x[1] for x in v]),
                 len=gm([x[2] for x in v]), op_gt_2x=sum(x[0] > 2 for x in v), max_len=max(x[2] for x in v))
        res.append(r)
        print(f"{name:9s} {k:8s} op {r['op']:6.2f} sto {r['sto']:6.2f} len {r['len']:7.2f} op>2x {r['op_gt_2x']:3d} max_len {r['max_len']:.3f}")
    (HERE / 'results' / 'astar_family.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
