"""LLM proposes topologies, code verifies: pick the shortest collision-free tight route among candidates."""
import sys
sys.dont_write_bytecode = True
import collections, json, math
from multiprocessing import Pool
from pathlib import Path
from shapely.geometry import LineString, Point
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o, astar_family as af, gate_rule as gr  # noqa: E401,E402
D = math.dist
L = lambda T: sum(D(a, b) for a, b in zip(T, T[1:]))
ok = lambda env, T: all(not env.collision(a, b) for a, b in zip(T, T[1:]))


def work(args):
    sample, query, texts = args
    env = o.Environment(json.loads(json.dumps(query)), collision_backend='shapely')
    grid = o.Grid(env)
    base, _ = af.oracle_T(env, grid)
    near = lambda line: (lambda a, b, c: min(c, key=lambda g: line.distance(Point(g))))
    rule = gr.build(env, env.start, env.goal, lambda a, b, c: min(c, key=lambda g: D(a, g) + D(g, b)))
    orc = gr.build(env, env.start, env.goal, near(LineString(base['path'])))
    topo = {}
    for p in o.PROMPTS:
        try:
            nodes = o.parse_path(texts[p], mode='upstream')
        except ValueError:
            nodes = []
        T = env.targets(nodes, 'upstream')
        line = LineString(T) if len(set(T)) > 1 else LineString([env.start, env.goal])
        topo[p] = gr.build(env, env.start, env.goal, near(line))
    def pick(cands):
        v = [T for T in cands if ok(env, T)]
        return min(v, key=L) if v else None
    sets = {'rule': [rule], 'llm_repe': [topo['repe']], 'llm3': list(topo.values()),
            'rule+llm_repe': [rule, topo['repe']], 'rule+llm3': [rule] + list(topo.values()), 'oracle_gate': [orc]}
    out = []
    for k, c in sets.items():
        T = pick(c)
        for w in (1.5, 2.0):
            r = o.search(env, grid, T, dedupe=True, via=w, switch='pop', timeout=60) if T else \
                o.search(env, grid, None, dedupe=True, wg=w)
            out.append((k, w, r['op'] / base['op'], r['len'] / base['len'], T == orc))
    return out


def main():
    texts, _ = o.load_texts()
    tasks = [(s, q, {p: texts[(s, p)] for p in o.PROMPTS}) for s, q in o.load_queries(o.DATA)]
    agg = collections.defaultdict(list)
    with Pool(16) as pool:
        for part in pool.imap_unordered(work, tasks, chunksize=4):
            for k, w, *m in part:
                agg[(k, w)].append(m)
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    res = []
    for (k, w), v in agg.items():
        r = dict(cands=k, w=w, op=gm([x[0] for x in v]), len=gm([x[1] for x in v]), topo_eq_oracle=sum(x[2] for x in v) / 10)
        res.append(r)
        print(f"{k:14s} w={w} op {r['op']:6.2f} len {r['len']:7.2f} topo=oracle {r['topo_eq_oracle']:5.1f}%")
    (HERE / 'results' / 'topology_select.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
