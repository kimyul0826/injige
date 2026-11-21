"""Is Llama's topology real information or just a plausible guess? (stored V2 responses, no inference)

All chains built by the same incremental gate builder (gate_rule.build); only the gate-picking rule differs:
  oracle    : gate nearest the optimal A* path                     (target)
  rule      : shortest detour d(a,g)+d(g,b)
  straight  : gate nearest the straight start-goal line             (trivial "plausible" default)
  random    : uniform random gate (5 seeds averaged)
  llm_<p>   : gate nearest the Llama polyline
Reported: agreement with oracle overall, and on the "hard" problems where rule != oracle
(does the LLM know something the trivial rules do not?).
"""
import sys
sys.dont_write_bytecode = True
import collections, json, math, random
from multiprocessing import Pool
from pathlib import Path

from shapely.geometry import LineString, Point

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o      # noqa: E402
import gate_rule as gr           # noqa: E402

D = math.dist


def work(args):
    sample, query, texts = args
    env = o.Environment(json.loads(json.dumps(query)), collision_backend='shapely')
    grid = o.Grid(env)
    base = o.search(env, grid, None, dedupe=True)
    near = lambda line: (lambda a, b, c: min(c, key=lambda g: line.distance(Point(g))))
    build = lambda pick: gr.build(env, env.start, env.goal, pick)
    orc = build(near(LineString(base['path'])))
    ch = {'rule': build(lambda a, b, c: min(c, key=lambda g: D(a, g) + D(g, b))),
          'straight': build(near(LineString([env.start, env.goal])))}
    for p in o.PROMPTS:
        try:
            nodes = o.parse_path(texts[p], mode='upstream')
        except ValueError:
            nodes = []
        T = env.targets(nodes, 'upstream')
        ch[f'llm_{p}'] = build(near(LineString(T) if len(set(T)) > 1 else LineString([env.start, env.goal])))
    same = {k: T == orc for k, T in ch.items()}
    rnd = 0.0
    for seed in range(5):
        rng = random.Random(hash((sample, seed)) & 0xffffffff)
        rnd += build(lambda a, b, c: rng.choice(list(c))) == orc
    same['random'] = rnd / 5
    trivial = orc == [env.start, env.goal]
    return sample, same, trivial


def main():
    texts, _ = o.load_texts()
    tasks = [(s, q, {p: texts[(s, p)] for p in o.PROMPTS}) for s, q in o.load_queries(o.DATA)]
    with Pool(16) as pool:
        rows = pool.map(work, tasks, chunksize=4)
    keys = list(rows[0][1])
    nontriv = [r for r in rows if not r[2]]
    hard = [r for r in nontriv if not r[1]['rule']]
    hard_s = [r for r in nontriv if not r[1]['straight']]
    pct = lambda rs, k: 100 * sum(r[1][k] for r in rs) / max(len(rs), 1)
    res = {'n': len(rows), 'n_nontrivial': len(nontriv), 'n_rule_wrong': len(hard), 'n_straight_wrong': len(hard_s)}
    print(f"problems {len(rows)}, needing a detour {len(nontriv)}, rule wrong {len(hard)}, straight wrong {len(hard_s)}")
    print(f"{'chooser':16s} {'all':>7s} {'detour':>7s} {'rule wrong':>11s} {'straight wrong':>15s}")
    for k in keys:
        a, b, c, d = pct(rows, k), pct(nontriv, k), pct(hard, k), pct(hard_s, k)
        res[k] = dict(all=a, nontrivial=b, rule_wrong=c, straight_wrong=d)
        print(f"{k:16s} {a:7.1f} {b:7.1f} {c:11.1f} {d:15.1f}")
    # does any LLM response fix rule's mistakes beyond what random would?
    anyllm = 100 * sum(any(r[1][f'llm_{p}'] for p in o.PROMPTS) for r in hard) / max(len(hard), 1)
    print(f"any of 3 LLM correct when rule wrong: {anyllm:.1f}%")
    res['any_llm_rule_wrong'] = anyllm
    (HERE / 'results' / 'topology_vs_trivial.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
