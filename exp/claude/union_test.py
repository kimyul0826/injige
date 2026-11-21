"""Is it safe to add an LLM route on top of the rule route in E-graph? (1000 problems, stored Llama, no inference)"""
import sys
sys.dont_write_bytecode = True
import collections, json, math
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o      # noqa: E402
import egraph_multi as em        # noqa: E402
import gate_rule as gr           # noqa: E402
import astar_family as af        # noqa: E402

D = math.dist
segs = lambda T: list(zip(T, T[1:]))


def work(args):
    sample, query, texts = args
    env = o.Environment(json.loads(json.dumps(query)), collision_backend='shapely')
    grid = o.Grid(env)
    base, orc = af.oracle_T(env, grid)
    rule = gr.build(env, env.start, env.goal, lambda a, b, c: min(c, key=lambda g: D(a, g) + D(g, b)))
    out = []
    sets = {'rule': segs(rule), 'oracle': segs(orc), 'rule+oracle': segs(rule) + segs(orc)}
    for p in o.PROMPTS:
        try:
            nodes = o.parse_path(texts[p], mode='upstream')
        except ValueError:
            nodes = []
        T = env.targets(nodes, 'upstream')
        sets[f'llm_{p}'] = segs(T)
        sets[f'rule+llm_{p}'] = segs(rule) + segs(T)
    for k, S in sets.items():
        r = em.run_with_segments(env, grid, S, 1.5, 1.33)
        out.append((k, r['op'] / base['op'], r['len'] / base['len']))
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
        res[k] = dict(op=gm([x[0] for x in v]), len=gm([x[1] for x in v]), op_gt_2x=sum(x[0] > 2 for x in v))
        print(f"{k:20s} op {res[k]['op']:6.2f} len {res[k]['len']:7.2f} op>2x {res[k]['op_gt_2x']}")
    (HERE / 'results' / 'union_test.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
