"""Headroom if the LLM only picks evaluation-function settings per problem (no waypoints, no inference).

Per problem, run a small portfolio and take the oracle-best choice:
  W  : weighted A* weight w in {1.25, 1.5, 2, 3, 5}
  H  : heuristic in {euclid w=2, rule-route w=2}  (rule-route = code gate chain, no LLM)
Oracle picks min op subject to len <= cap (cap = 1.02 * A* length, i.e. the wA*2 average level).
Upper bound for "LLM chooses w / chooses heuristic" style outputs.
"""
import sys
sys.dont_write_bytecode = True
import collections, json, math
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o      # noqa: E402
import gate_rule as gr           # noqa: E402

D = math.dist
WS = (1.25, 1.5, 2.0, 3.0, 5.0)


def work(args):
    sample, query = args
    env = o.Environment(json.loads(json.dumps(query)), collision_backend='shapely')
    grid = o.Grid(env)
    base = o.search(env, grid, None, dedupe=True)
    rule = gr.build(env, env.start, env.goal, lambda a, b, c: min(c, key=lambda g: D(a, g) + D(g, b)))
    rows = {}
    for w in WS:
        r = o.search(env, grid, None, dedupe=True, wg=w)
        rows[f'wA*{w}'] = (r['op'] / base['op'], r['len'] / base['len'])
        r = o.search(env, grid, rule, dedupe=True, via=w, switch='pop', timeout=60)
        rows[f'rule{w}'] = (r['op'] / base['op'], r['len'] / base['len'])
    return rows


def main():
    tasks = list(o.load_queries(o.DATA))
    allrows = []
    with Pool(16) as pool:
        allrows = pool.map(work, tasks, chunksize=8)
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    res = {}
    def report(name, picks):
        res[name] = dict(op=gm([p[0] for p in picks]), len=gm([p[1] for p in picks]))
        print(f"{name:34s} op {res[name]['op']:6.2f} len {res[name]['len']:7.2f}")
    for k in ('wA*2.0', 'rule2.0', 'wA*1.5', 'rule1.5'):
        report(f'fixed {k}', [r[k] for r in allrows])
    for cap in (1.0, 1.02, 1.05):
        for fam, keys in (('w only (wA*)', [f'wA*{w}' for w in WS]),
                          ('heuristic only (w=2)', ['wA*2.0', 'rule2.0']),
                          ('w x heuristic', list(allrows[0]))):
            picks = []
            for r in allrows:
                ok = [r[k] for k in keys if r[k][1] <= cap + 1e-9] or [min((r[k] for k in keys), key=lambda x: x[1])]
                picks.append(min(ok))
            report(f'oracle {fam} len<={cap}', picks)
    (HERE / 'results' / 'eval_choice_headroom.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
