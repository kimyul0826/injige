"""Analyse a candidate_waypoints run: candidate adherence, collision-free chains, and op of
LLM-A* (official) / LLM-A*+wallC full / wallC+visroute w=1.5 full vs wallC alone. Usage: SRC=<output dir>."""
import sys
sys.dont_write_bytecode = True
import collections, json, math, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import offline_improve as o           # noqa: E402
o.SRC = Path(os.environ['SRC'])
import combine_llm_wallc as cb        # noqa: E402
from llm_astar.prompt_variants import candidate_waypoints  # noqa: E402

texts, stored = o.load_texts()
queries = {s: q for s, q in o.load_queries(o.SRC / 'input.json')}
gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
agg, stat = collections.defaultdict(list), collections.defaultdict(collections.Counter)
for s, q in queries.items():
    cand = {tuple(c) for c in candidate_waypoints(q)}
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
    for p in o.PROMPTS:
        try:
            nodes = o.parse_path(texts[(s, p)], mode='upstream')
        except ValueError:
            nodes = []
        mid = [tuple(n) for n in nodes[1:-1]] if nodes[:1] == [tuple(q['start'])] else [tuple(n) for n in nodes]
        mid = [n for n in mid if n not in (tuple(q['start']), tuple(q['goal']))]
        inside = sum(n in cand for n in mid)
        raw = env.targets(nodes, 'upstream')
        full = cb.fully_valid(env, raw)
        st = stat[p]
        st['n'] += 1; st['wp'] += len(mid); st['in_cand'] += inside
        st['all_in_cand'] += inside == len(mid)
        st['full_ok'] += len(full) > 2 or (len(raw) == 2 and not env.collision(env.start, env.goal))
        st['kept_after_filter'] += len(raw) - 2
        print(s, p, 'wp', len(mid), 'in_cand', inside, 'full', len(full) - 2, mid)
    for name, op, ln, _ in cb.work((s, q, {p: texts[(s, p)] for p in o.PROMPTS})):
        agg[name].append((op, ln))
print()
for p, st in stat.items():
    print(p, dict(st))
keys = ['wA* w=2', 'wallC w=1.5'] + [f'{m} {p}|{k}' for p in o.PROMPTS for m, k in
        (('LLM-A*', 'raw'), ('LLM-A*+wallC', 'full'), ('wallC+visroute w=1.5', 'full'))] + \
       ['wallC+visroute w=1.5 oracle|raw']
for k in keys:
    v = agg[k]
    print(f'{k:40s} op {gm([x[0] for x in v]):6.2f} len {gm([x[1] for x in v]):7.2f} max_len {100*max(x[1] for x in v):6.1f}')
