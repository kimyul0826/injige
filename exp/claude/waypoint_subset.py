"""LLM-A* restricted to queries where at least one waypoint survives the official filter
(zero surviving waypoints => LLM-A* == wA*2 exactly). Compared with A* and wA*2 on the SAME subset.
Also for verified-prefix (vp) and fully-valid (full) chains. Upstream accounting, stored Llama-3 responses."""
import sys
sys.dont_write_bytecode = True
import collections, json, math
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o, combine_llm_wallc as cb   # noqa: E402


def work(args):
    s, q, texts = args
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely'); grid = o.Grid(env)
    base = o.search(env, grid, None); wa = o.search(env, grid, None, wg=2.0)
    rows = []
    for p in o.PROMPTS:
        try: nodes = o.parse_path(texts[p], mode='upstream')
        except ValueError: nodes = []
        raw = env.targets(nodes, 'upstream')
        for kind, T in (('raw', raw), ('vp', cb.verified_prefix(env, raw)), ('full', cb.fully_valid(env, raw))):
            r = o.search(env, grid, T)
            rows.append((p, kind, len(T) - 2, r['op'] / base['op'], r['len'] / base['len'],
                         wa['op'] / base['op'], wa['len'] / base['len'], r['op'] == wa['op']))
    return rows


def main():
    texts, _ = o.load_texts()
    tasks = [(s, q, {p: texts[(s, p)] for p in o.PROMPTS}) for s, q in o.load_queries(o.DATA)]
    agg = collections.defaultdict(list)
    with Pool(10) as pool:
        for rows in pool.imap_unordered(work, tasks, chunksize=4):
            for p, kind, *m in rows:
                agg[(p, kind)].append(m)
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v)) if v else float('nan')
    res = {}
    for (p, kind), v in sorted(agg.items()):
        has = [x for x in v if x[0] > 0]; none = [x for x in v if x[0] == 0]
        r = dict(n_with_wp=len(has), n_zero=len(none),
                 zero_equals_wA2=sum(x[5] for x in none),
                 with_wp=dict(llm_op=gm([x[1] for x in has]), llm_len=gm([x[2] for x in has]),
                              wA2_op=gm([x[3] for x in has]), wA2_len=gm([x[4] for x in has]),
                              llm_worse_than_Astar=sum(x[1] > 1 for x in has),
                              llm_better_than_wA2=sum(x[1] < x[3] for x in has),
                              llm_worse_than_wA2=sum(x[1] > x[3] for x in has),
                              llm_op_gt_2x=sum(x[1] > 2 for x in has)),
                 all=dict(llm_op=gm([x[1] for x in v])))
        res[f'{p}|{kind}'] = r
        w = r['with_wp']
        print(f"{p:8} {kind:4} with-wp {len(has):4d} zero {len(none):4d} (==wA*2 {r['zero_equals_wA2']:4d}) | "
              f"subset: LLM-A* op {w['llm_op']:6.2f} len {w['llm_len']:6.2f} | wA*2 op {w['wA2_op']:6.2f} len {w['wA2_len']:6.2f} | "
              f"LLM>A* {w['llm_worse_than_Astar']:3d}  LLM<wA2 {w['llm_better_than_wA2']:3d}  LLM>wA2 {w['llm_worse_than_wA2']:3d}  >2x {w['llm_op_gt_2x']:3d}"
              f" | all {r['all']['llm_op']:6.2f}")
    (HERE / 'results' / 'waypoint_subset.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
