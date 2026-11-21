"""E-Graph search (Phillips et al. 2012, logic_variants.egraph) on stored GPT-4o mini responses, no API calls.
Dedupe accounting (as LOGIC_REPORT.md). Each LLM run is paired with the same search given no waypoints
([start, goal]) and with oracle waypoints (string-pulled optimal path, first 3).
Run: PYTHONDONTWRITEBYTECODE=1 python exp/claude/egraph_gpt.py"""
import sys
sys.dont_write_bytecode = True
import json, math, random
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o           # noqa: E402
import logic_variants as lv           # noqa: E402
import post_inference_gpt as pg       # noqa: E402  (response sets + loader)

CFG = [('egraph e1.5 w1.33', dict(eps_e=1.5, w=1.33)), ('egraph e2 w1.25', dict(eps_e=2.0, w=1.25))]


def work(args):
    s, q = args
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
    grid = o.Grid(env)
    base = o.search(env, grid, None, dedupe=True)
    opt = base['path']
    turns = [opt[i] for i in range(1, len(opt) - 1)
             if (opt[i][0] - opt[i-1][0], opt[i][1] - opt[i-1][1]) != (opt[i+1][0] - opt[i][0], opt[i+1][1] - opt[i][1])]
    oracle = [env.start] + o.string_pull(env, [env.start] + turns + [env.goal])[1:-1][:3] + [env.goal]
    out, lens = {}, {}

    def rec(key, r):
        out[key] = r['op'] / base['op']
        lens[key] = r['len'] / base['len']
    for cname, kw in CFG:
        rec(('noLLM', cname), lv.egraph(env, grid, [env.start, env.goal], **kw))
        rec(('oracle', cname), lv.egraph(env, grid, oracle, **kw))
        for name, _, prompts in pg.SETS:
            _, texts = pg.DATA[name]
            for p in prompts:
                try:
                    nodes = o.parse_path(texts[(s, p)], mode='upstream')
                except ValueError:
                    nodes = []
                rec((f'{name}:{p}', cname), lv.egraph(env, grid, env.targets(nodes, 'upstream'), **kw))
    rec(('ref', 'wA*2 dedupe'), o.search(env, grid, None, dedupe=True, wg=2.0))
    return s, out, lens


if __name__ == '__main__':
    with Pool(16) as pool:
        res = pool.map(work, list(o.load_queries(o.DATA)), chunksize=8)
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    random.seed(0)
    summary, lines = {}, []
    for key in res[0][1]:
        e = dict(op=gm([r[1][key] for r in res]), len=gm([r[2][key] for r in res]),
                 max_len=100 * max(r[2][key] for r in res))
        if key[0] not in ('noLLM', 'ref'):
            pr = [math.log(r[1][key] / r[1][('noLLM', key[1])]) for r in res]
            bs = sorted(math.exp(sum(random.choice(pr) for _ in pr) / len(pr)) * 100 for _ in range(1000))
            e['vs_noLLM'] = [round(math.exp(sum(pr) / len(pr)) * 100, 1), round(bs[25], 1), round(bs[975], 1)]
            for half in (0, 1):   # map-id parity split
                hp = [x for x, r in zip(pr, res) if int(r[0].split(':')[0]) % 2 == half]
                e[f'vs_noLLM_maps{"even" if half == 0 else "odd"}'] = round(math.exp(sum(hp) / len(hp)) * 100, 1)
        summary['|'.join(key)] = e
        lines.append(f"{'|'.join(key):45s} op {e['op']:6.2f} len {e['len']:6.2f} max {e['max_len']:5.1f}  "
                     f"LLM/noLLM {e.get('vs_noLLM', '')} even/odd maps {e.get('vs_noLLM_mapseven', '')}/{e.get('vs_noLLM_mapsodd', '')}")
    print('\n'.join(lines))
    (HERE / 'results' / 'egraph_gpt.txt').write_text('\n'.join(lines) + '\n')
    (HERE / 'results' / 'egraph_gpt.json').write_text(json.dumps(summary, indent=1))
