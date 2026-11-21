"""Pick representative failure cases from failure_cases.json and record the states each search expands.
Stage 1 (llm-astar env): writes exp/claude/results/failure_case_plots.json; stage 2: plot_failure_cases.py.
Cases (picked by rule, not by hand):
  runaway      gpt original standard, largest LLM-A*/wA*2 operation ratio
  valid_hurts  gpt original standard, LLM path collision-free but LLM-A* op largest vs wA*2
  detour       gpt endpoint standard, largest detour (chain / optimal) among chains >= 1.6
  false_valid  gpt repe2, 'corner cut' response with the largest LLM-A*/wA*2 ratio
  on_barrier   gpt repe2, 'on barrier' response with the largest ratio
"""
import sys
sys.dont_write_bytecode = True
import heapq, json, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o           # noqa: E402

D, INF = math.dist, math.inf
rows = json.loads((HERE / 'results' / 'failure_cases.json').read_text())
qs = dict(o.load_queries(o.DATA))


def recording_search(env, grid, T):
    """Released LLM-A* semantics (cf. combine_llm_wallc.tsearch) that also returns the popped states."""
    start, goal = env.start, env.goal
    idx = 1
    f = lambda p: g[p] + D(p, goal) + D(p, T[idx])
    g, parent = {start: 0.0}, {start: start}
    q, closed, pops = [(f(start), start)], set(), []
    while q:
        s = heapq.heappop(q)[-1]
        pops.append(s)
        closed.add(s)
        if s == goal:
            break
        for dx, dy in o.MOTIONS:
            n = (s[0] + dx, s[1] + dy)
            if n == T[idx] and n != goal:
                idx = min(idx + 1, len(T) - 1)
                q = [(f(e[-1]), e[-1]) for e in q]
                heapq.heapify(q)
            if n in closed:
                continue
            g.setdefault(n, INF)
            if grid.coll(s, n):
                continue
            new = g[s] + D(s, n)
            if new < g[n]:
                g[n], parent[n] = new, s
                heapq.heappush(q, (f(n), n))
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return pops, path[::-1]


def pick(set_name, prompt, cond, key):
    cand = [r for r in rows if r['set'] == set_name and r['prompt'] == prompt and cond(r)]
    return max(cand, key=key)


ratio = lambda r: r['op'] / r['w2']
cases = {
    'runaway': pick('gpt original', 'standard', lambda r: True, ratio),
    'valid_hurts': pick('gpt original', 'standard', lambda r: r['cls'] == 'valid', ratio),
    'detour': pick('gpt endpoint', 'standard', lambda r: r['detour'] >= 1.6, lambda r: r['detour']),
    'false_valid': pick('gpt repe2', 'repe2', lambda r: r['cls'] == 'corner cut', ratio),
    'on_barrier': pick('gpt repe2', 'repe2', lambda r: r['cls'] == 'on barrier', ratio),
}
out = {}
for name, r in cases.items():
    q = qs[r['sample']]
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
    grid = o.Grid(env)
    T = [tuple(t) for t in r['targets']]
    pops_llm, path_llm = recording_search(env, grid, T)
    pops_w, path_w = recording_search(env, grid, [env.start, env.goal])
    assert len(pops_w) == o.search(env, grid, None, wg=2.0)['op']
    bad = [[list(a), list(b)] for a, b in zip(map(tuple, r['path']), map(tuple, r['path'][1:])) if env.collision(a, b)]
    out[name] = dict(sample=r['sample'], set=r['set'], prompt=r['prompt'], cls=r['cls'], op=r['op'], w2=r['w2'],
                     detour=r['detour'], query=q, llm_path=r['path'], targets=r['targets'], bad_segments=bad,
                     pops_llm=[list(p) for p in pops_llm], pops_w2=[list(p) for p in pops_w],
                     path_llm=[list(p) for p in path_llm], path_w2=[list(p) for p in path_w])
    print(f"{name:12s} {r['sample']:6s} {r['cls']:12s} LLM-A* op {100*r['op']:.0f}% vs wA*2 {100*r['w2']:.0f}%  "
          f"pops {len(pops_llm)} vs {len(pops_w)}  detour {r['detour']:.2f}  path {r['path']}")
(HERE / 'results' / 'failure_case_plots.json').write_text(json.dumps(out))
