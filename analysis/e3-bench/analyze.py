"""E3: prompts on other map families (data/bench_e3.json = first 100 problems of dense, rooms, utrap, maze,
scaled2 from exp/claude/bench). Per family, operation ratio vs A* (GM x100), upstream accounting for the
released search; E-Graph uses dedupe accounting (as LOGIC_REPORT.md).
Run: PYTHONDONTWRITEBYTECODE=1 python analysis/e3-bench/analyze.py"""
import sys
sys.dont_write_bytecode = True
import json, math, random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'exp' / 'claude'))
import offline_improve as o           # noqa: E402
import logic_variants as lv           # noqa: E402
import wall_component_h as wc         # noqa: E402
import llmastar_vs_wallc as lvw       # noqa: E402

RUNS = {'standard': ('outputs/gpt4omini-e3-bench-batch-v1', 'standard'),
        'repe2': ('outputs/gpt4omini-e3-bench-batch-v1', 'repe2'),
        'endpoint std': ('outputs/gpt4omini-e3-bench-endpoint-batch-v1', 'standard')}
DATA = {}
for name, (d, p) in RUNS.items():
    rows = {(r['sample'], r['method'], r['prompt']): r for r in map(json.loads, open(ROOT / d / 'results.jsonl'))}
    DATA[name] = (rows, d, p)


def text(name, s):
    rows, d, p = DATA[name]
    return json.loads((ROOT / d / 'responses' / f"{rows[(s, 'llm-astar', p)]['cache_key']}.json").read_text())['text']


def work(args):
    s, q = args
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
    grid = o.Grid(env)
    base = o.search(env, grid, None)
    based = o.search(env, grid, None, dedupe=True)
    out = {'wA*2': o.search(env, grid, None, wg=2.0)['op'] / base['op'],
           'wallC 1.5': lvw.astar_h(env, grid, wc.make_wallC(env), 1.5, True)['op'] / base['op'],
           'egraph noLLM': lv.egraph(env, grid, [env.start, env.goal], eps_e=1.5, w=1.33)['op'] / based['op']}
    valid = {}
    for name, (rows, _, p) in DATA.items():
        assert rows[(s, 'astar', '-')]['operation'] == base['op'], s
        out[f'LLM-A* {name}'] = rows[(s, 'llm-astar', p)]['operation'] / base['op']
        valid[name] = rows[(s, 'llm', p)]['valid']
        try:
            nodes = o.parse_path(text(name, s), mode='upstream')
        except ValueError:
            nodes = []
        out[f'egraph {name}'] = lv.egraph(env, grid, env.targets(nodes, 'upstream'), eps_e=1.5, w=1.33)['op'] / based['op']
    return s.split('-')[0], out, valid


if __name__ == '__main__':
    qs = list(o.load_queries(ROOT / 'data' / 'bench_e3.json'))
    with Pool(16) as pool:
        res = pool.map(work, qs, chunksize=4)
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    random.seed(0)
    fams = sorted({f for f, _, _ in res}, key=['dense', 'rooms', 'utrap', 'maze', 'scaled2'].index)
    keys = list(res[0][1])
    lines = ['family   ' + ''.join(k.rjust(18) for k in keys) + '   LLM-only valid % (std / repe2 / endpoint)']
    for f in fams + ['all']:
        R = [r for r in res if f in ('all', r[0])]
        cells = ''.join(f'{gm([r[1][k] for r in R]):18.2f}' for k in keys)
        val = ' / '.join(f"{100 * sum(r[2][n] for r in R) / len(R):.0f}" for n in RUNS)
        lines.append(f'{f:8s} {cells}   {val}')
    lines.append('\npaired (95% CI over problems):')
    for f in fams + ['all']:
        R = [r for r in res if f in ('all', r[0])]
        for a, b in [('LLM-A* repe2', 'LLM-A* standard'), ('LLM-A* repe2', 'wA*2'), ('egraph endpoint std', 'egraph noLLM')]:
            pr = [math.log(r[1][a] / r[1][b]) for r in R]
            bs = sorted(math.exp(sum(random.choice(pr) for _ in pr) / len(pr)) * 100 for _ in range(1000))
            lines.append(f'  {f:8s} {a} / {b}: {math.exp(sum(pr) / len(pr)) * 100:.1f} [{bs[25]:.1f}, {bs[975]:.1f}]')
    text_out = '\n'.join(lines)
    print(text_out)
    (Path(__file__).parent / 'analysis.txt').write_text(text_out + '\n')
