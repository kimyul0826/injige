"""LLM decides topology, code builds geometry. Offline test with stored Llama responses (no inference).

For each blocking wall met by the incremental gate builder (gate_rule.build), choose the gate
  rule        : shortest detour d(a,g)+d(g,b)                    (no LLM)
  llm_topo_*  : gate closest to the LLM polyline (Llama's route used only as a topology hint)
  oracle_gate : gate closest to the optimal A* path
and compare with the raw Llama chain and the oracle turning points.
Measures: shape error of the route heuristic vs h*, topology agreement with oracle, and search
with f = g + w * route (switch on reaching the target, timeout 60), gated by chain validity.
"""
import sys
sys.dont_write_bytecode = True
import collections, json, math, statistics
from multiprocessing import Pool
from pathlib import Path

from shapely.geometry import LineString, Point

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o           # noqa: E402
import astar_family as af             # noqa: E402
import gate_rule as gr                # noqa: E402
import heuristic_quality as hq        # noqa: E402

D = math.dist


def closed_states(env, grid):
    import heapq
    g, q, closed = {env.start: 0.0}, [(D(env.start, env.goal), env.start)], set()
    while q:
        _, s = heapq.heappop(q)
        if s in closed:
            continue
        closed.add(s)
        if s == env.goal:
            break
        for dx, dy in o.MOTIONS:
            n = (s[0] + dx, s[1] + dy)
            if n in closed or grid.coll(s, n):
                continue
            if g[s] + D(s, n) < g.get(n, math.inf):
                g[n] = g[s] + D(s, n); heapq.heappush(q, (g[n] + D(n, env.goal), n))
    return closed


def route_shape_error(T, states, hstar):
    """Route heuristic with each state assigned to its nearest chain segment (target = that segment's end)."""
    chain = [0.0] * len(T)
    for k in range(len(T) - 2, -1, -1):
        chain[k] = chain[k + 1] + D(T[k], T[k + 1])
    segs = [LineString([T[i], T[i + 1]]) if T[i] != T[i + 1] else Point(T[i]) for i in range(len(T) - 1)]
    hs, vals = [], []
    for s in states:
        t = hstar.get(s)
        if not t:
            continue
        p = Point(s)
        i = min(range(len(segs)), key=lambda j: segs[j].distance(p))
        vals.append(D(s, T[i + 1]) + chain[i + 1])
        hs.append(t)
    scale = statistics.median(v / t for v, t in zip(vals, hs))
    return scale, statistics.median(abs(v / scale - t) / t for v, t in zip(vals, hs))


def work(args):
    sample, query, texts = args
    env = o.Environment(json.loads(json.dumps(query)), collision_backend='shapely')
    grid = o.Grid(env)
    hstar = hq.true_cost(env, grid)
    states = [s for s in closed_states(env, grid) if s != env.goal]
    base, orc = af.oracle_T(env, grid)
    opt = LineString(base['path'])
    near = lambda line: (lambda a, b, c: min(c, key=lambda g: line.distance(Point(g))))
    chains = {
        'rule': gr.build(env, env.start, env.goal, lambda a, b, c: min(c, key=lambda g: D(a, g) + D(g, b))),
        'oracle_gate': gr.build(env, env.start, env.goal, near(opt)),
        'oracle_turns': orc,
    }
    for p in o.PROMPTS:
        try:
            nodes = o.parse_path(texts[p], mode='upstream')
        except ValueError:
            nodes = []
        T = env.targets(nodes, 'upstream')
        chains[f'llm_raw_{p}'] = T
        line = LineString(T) if len(set(T)) > 1 else LineString([env.start, env.goal])
        chains[f'llm_topo_{p}'] = gr.build(env, env.start, env.goal, near(line))
    out = []
    for k, T in chains.items():
        scale, err = route_shape_error(T, states, hstar)
        valid = all(not env.collision(a, b) for a, b in zip(T, T[1:]))
        same = T == chains['oracle_gate']
        row = [k, scale, err, valid, same]
        for w in (1.5, 2.0):
            r = o.search(env, grid, T, dedupe=True, via=w, switch='pop', timeout=60) if valid else \
                o.search(env, grid, None, dedupe=True, wg=w)
            row += [r['op'] / base['op'], r['len'] / base['len']]
        out.append(row)
    for w in (1.5, 2.0):
        r = o.search(env, grid, None, dedupe=True, wg=w)
        out.append([f'wA*{w}', None, None, None, None, r['op'] / base['op'], r['len'] / base['len'],
                    r['op'] / base['op'], r['len'] / base['len']])
    return out


def main():
    texts, _ = o.load_texts()
    tasks = [(s, q, {p: texts[(s, p)] for p in o.PROMPTS}) for s, q in o.load_queries(o.DATA)]
    agg = collections.defaultdict(list)
    with Pool(16) as pool:
        for part in pool.imap_unordered(work, tasks, chunksize=4):
            for k, *m in part:
                agg[k].append(m)
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    res = {}
    for k, v in agg.items():
        r = dict(op_w1_5=gm([x[4] for x in v]), len_w1_5=gm([x[5] for x in v]),
                 op_w2=gm([x[6] for x in v]), len_w2=gm([x[7] for x in v]))
        if v[0][0] is not None:
            r.update(scale=statistics.median(x[0] for x in v), shape_err=statistics.median(x[1] for x in v),
                     valid=sum(x[2] for x in v) / 10, same_as_oracle_gate=sum(x[3] for x in v) / 10)
        res[k] = r
        extra = (f"shape_err {100 * r['shape_err']:5.1f}%  valid {r['valid']:5.1f}%  topo=oracle {r['same_as_oracle_gate']:5.1f}%  "
                 if 'scale' in r else ' ' * 56)
        print(f"{k:20s} {extra}w1.5 {r['op_w1_5']:6.2f}/{r['len_w1_5']:6.2f}  w2 {r['op_w2']:6.2f}/{r['len_w2']:6.2f}")
    (HERE / 'results' / 'topology_extract.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
