"""Post-inference search variants on stored GPT-4o mini responses (no API calls).

For every response set, on all 1000 problems, Table-1 (upstream) accounting:
  official        released LLM-A* (asserted equal to the stored run)
  progress        target switch: drop a target once the current node is closer to the goal than it
  progress+w      progress, goal term x2, target term x0.5
  wallC+route     f = g + 1.5*max(wallC(s), route(s)); chain used only if every segment is collision-free
references: weighted A* w=2, wallC w=1.5 (no LLM).
Run: PYTHONDONTWRITEBYTECODE=1 python exp/claude/post_inference_gpt.py
"""
import sys
sys.dont_write_bytecode = True
import json, math, random
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o           # noqa: E402
import combine_llm_wallc as cb        # noqa: E402
import wall_component_h as wc         # noqa: E402
import llmastar_vs_wallc as lv        # noqa: E402

ROOT = HERE.parents[1]
SETS = [('original', 'outputs/gpt4omini-table1-upstream-batch-v1', ('standard', 'cot', 'repe')),
        ('waypoint-limit', 'outputs/gpt4omini-waypoint-limit-batch-v1', ('standard', 'cot', 'repe')),
        ('endpoint', 'outputs/gpt4omini-barrier-endpoints-demos-batch-v1', ('standard', 'cot', 'repe')),
        ('repe2', 'outputs/gpt4omini-repe2-batch-v1', ('repe2',)),
        ('E1a noround', 'outputs/gpt4omini-repe2_noround-batch-v1', ('repe2_noround',)),
        ('E1b origdemos', 'outputs/gpt4omini-repe2_origdemos-batch-v1', ('repe2_origdemos',))]


def load(d):
    rows = {(r['sample'], r['method'], r['prompt']): r for r in map(json.loads, open(ROOT / d / 'results.jsonl'))}
    texts = {}
    for (s, m, p), r in rows.items():
        if m == 'llm-astar':
            texts[(s, p)] = json.loads((ROOT / d / 'responses' / f"{r['cache_key']}.json").read_text())['text']
    return rows, texts


DATA = {name: load(d) for name, d, _ in SETS}


def work(args):
    s, q = args
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
    grid = o.Grid(env)
    base = o.search(env, grid, None)
    out = {('ref', 'wA* w=2'): o.search(env, grid, None, wg=2.0)['op'] / base['op']}
    hw = wc.make_wallC(env)
    out[('ref', 'wallC w=1.5')] = lv.astar_h(env, grid, hw, 1.5, True)['op'] / base['op']
    lens = {}
    nollm = o.search(env, grid, [env.start, env.goal], switch='progress', wg=2.0, wt=0.5)  # = weighted A* w=2.5
    out[('ref', 'noLLM progress+w')] = nollm['op'] / base['op']
    lens[('ref', 'noLLM progress+w')] = nollm['len'] / base['len']
    for name, _, prompts in SETS:
        rows, texts = DATA[name]
        for p in prompts:
            try:
                nodes = o.parse_path(texts[(s, p)], mode='upstream')
            except ValueError:
                nodes = []
            T = env.targets(nodes, 'upstream')
            r = o.search(env, grid, T)
            st = rows[(s, 'llm-astar', p)]
            assert (r['op'], r['sto']) == (st['operation'], st['storage']), (name, s, p)
            res = {'official': r,
                   'progress': o.search(env, grid, T, switch='progress'),
                   'progress+w': o.search(env, grid, T, switch='progress', wg=2.0, wt=0.5),
                   'wallC+route': lv.astar_h(env, grid, cb.combined(hw, cb.make_route(env, cb.fully_valid(env, T))), 1.5, True)}
            for k, v in res.items():
                out[(f'{name}:{p}', k)] = v['op'] / base['op']
                lens[(f'{name}:{p}', k)] = v['len'] / base['len']
    return out, lens


if __name__ == '__main__':
    qs = list(o.load_queries(o.DATA))
    with Pool(16) as pool:
        res = pool.map(work, qs, chunksize=8)
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    keys = list(res[0][0])
    summary = {f'{a}|{b}': dict(op=gm([r[0][(a, b)] for r in res]),
                                len=gm([r[1][(a, b)] for r in res]) if (a, b) in res[0][1] else None,
                                over2=sum(r[0][(a, b)] > 2 for r in res)) for a, b in keys}
    random.seed(0)
    # paired against the matching no-LLM control (same search, waypoints replaced by [start, goal])
    control = {'official': ('ref', 'wA* w=2'), 'progress': ('ref', 'wA* w=2'),
               'progress+w': ('ref', 'noLLM progress+w'), 'wallC+route': ('ref', 'wallC w=1.5')}
    for a, b in keys:
        if a == 'ref':
            continue
        pr = [math.log(r[0][(a, b)] / r[0][control[b]]) for r in res]
        bs = sorted(math.exp(sum(random.choice(pr) for _ in pr) / len(pr)) * 100 for _ in range(1000))
        summary[f'{a}|{b}']['vs_noLLM'] = [round(math.exp(sum(pr) / len(pr)) * 100, 1), round(bs[25], 1), round(bs[975], 1)]
    for k, v in summary.items():
        print(f"{k:38s} op {v['op']:6.2f}  len {v['len'] or 0:6.2f}  >200% {v['over2']:3d}  LLM/noLLM {v.get('vs_noLLM', '')}")
    (HERE / 'results' / 'post_inference_gpt.json').write_text(json.dumps(summary, indent=1))
