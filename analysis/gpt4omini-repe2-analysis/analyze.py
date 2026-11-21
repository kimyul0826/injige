"""RePE2 batch vs original GPT-4o mini prompts, endpoint RePE and weighted A* (w=2).
Run: PYTHONPATH=. python -B outputs/gpt4omini-repe2-analysis/analyze.py"""
import sys
sys.dont_write_bytecode = True
import json, math, random, re
from collections import Counter
from multiprocessing import Pool
sys.path.insert(0, 'exp/claude')
import offline_improve as o                 # noqa: E402
import combine_llm_wallc as cb              # noqa: E402
from llm_astar.models import parse_path     # noqa: E402

NEW = 'outputs/gpt4omini-repe2-batch-v1'
OLD = 'outputs/gpt4omini-table1-upstream-batch-v1'
EP = 'outputs/gpt4omini-barrier-endpoints-demos-batch-v1'
load = lambda O: {(r['sample'], r['method'], r['prompt']): r for r in map(json.loads, open(O + '/results.jsonl'))}
new, old, ep = load(NEW), load(OLD), load(EP)
qs = dict(o.load_queries(o.DATA))
S = sorted(qs)
gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v)) if v else float('nan')
random.seed(0)


def boot(v, n=2000):
    logs = [math.log(x) for x in v]
    m = sorted(math.exp(sum(random.choice(logs) for _ in logs) / len(logs)) * 100 for _ in range(n))
    return m[int(.025 * n)], m[int(.975 * n)]


def per_problem(s):
    q = qs[s]
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
    grid = o.Grid(env)
    base = new[(s, 'astar', '-')]['operation']
    w2 = o.search(env, grid, None, wg=2.0)['op'] / base
    r = new[(s, 'llm-astar', 'repe2')]
    resp = json.loads(open(f"{NEW}/responses/{r['cache_key']}.json").read())
    text = resp['text']
    try:
        P = parse_path(text, 'upstream')
    except Exception:
        P = []
    T = env.targets(P, 'upstream')
    rounds = [int(x) for x in re.findall(r'Round (\d+)', text)]
    verdicts = re.findall(r'\b(Not valid|Valid)\b', text)
    opt = cb.oracle_chain(env, grid)
    chain = sum(math.dist(a, b) for a, b in zip(T, T[1:]))
    optlen = o.search(env, grid, None, dedupe=True)['len']
    return dict(s=s, w2=w2, op=r['operation'] / base, wp=len(P) - 2, kept=len(T) - 2,
                max_round=max(rounds) if rounds else -1, last=verdicts[-1] if verdicts else '-',
                valid=not any(env.collision(a, b) for a, b in zip(P, P[1:])) if len(P) > 1 else False,
                kept_valid=not any(env.collision(a, b) for a, b in zip(T, T[1:])),
                tok=resp['usage'].get('completion_tokens', 0), opt_wp=len(opt) - 2, detour=chain / optlen)


if __name__ == '__main__':
    with Pool(16) as p:
        rows = p.map(per_problem, S, chunksize=8)
    R = {r['s']: r for r in rows}
    out = ['## LLM-A* operation ratio (A*=100), 1000 problems, paired against RePE2']
    ref = [R[s]['op'] for s in S]
    comps = [('weighted A* w=2', [R[s]['w2'] for s in S])]
    comps += [(f'original {p}', [old[(s, 'llm-astar', p)]['operation'] / new[(s, 'astar', '-')]['operation'] for s in S])
              for p in ('standard', 'cot', 'repe')]
    comps += [('endpoint repe', [ep[(s, 'llm-astar', 'repe')]['operation'] / new[(s, 'astar', '-')]['operation'] for s in S])]
    out.append(f'RePE2                op {gm(ref):6.2f}   op>200%: {sum(x > 2 for x in ref)}')
    for name, v in comps:
        pr = [a / b for a, b in zip(ref, v)]
        lo, hi = boot(pr)
        out.append(f'{name:20s} op {gm(v):6.2f}   RePE2/this {gm(pr):6.1f} [{lo:.1f}, {hi:.1f}]  '
                   f'RePE2 better {sum(a < b for a, b in zip(ref, v))} worse {sum(a > b for a, b in zip(ref, v))}  op>200%: {sum(x > 2 for x in v)}')
    out.append('\n## behaviour')
    out.append(f"output tokens mean {sum(r['tok'] for r in rows) / 1000:.0f}, max {max(r['tok'] for r in rows)}")
    out.append('final waypoints     ' + str(dict(sorted(Counter(min(r['wp'], 6) for r in rows).items()))) + '  (6 = 6+)')
    out.append('kept after filter   ' + str(dict(sorted(Counter(min(r['kept'], 6) for r in rows).items()))))
    out.append('optimal turn points ' + str(dict(sorted(Counter(r['opt_wp'] for r in rows).items()))))
    out.append('highest round       ' + str(dict(sorted(Counter(r['max_round'] for r in rows).items()))))
    out.append('last verdict        ' + str(dict(Counter(r['last'] for r in rows))))
    out.append(f"final path really collision-free: {sum(r['valid'] for r in rows)}  "
               f"(filtered chain collision-free: {sum(r['kept_valid'] for r in rows)})")
    out.append('\n## op by kept waypoints (RePE2 vs weighted A* on the same problems)')
    for k in range(0, 5):
        sub = [r for r in rows if min(r['kept'], 4) == k]
        if sub:
            out.append(f"kept {k}{'+' if k == 4 else ' '} n={len(sub):4d}  RePE2 {gm([r['op'] for r in sub]):6.2f}  wA* {gm([r['w2'] for r in sub]):6.2f}")
    out.append('\n## op by chain detour (filtered chain length / optimal path length)')
    for lo, hi in ((0, 1.05), (1.05, 1.2), (1.2, 1.5), (1.5, 99)):
        sub = [r for r in rows if lo <= r['detour'] < hi]
        out.append(f'[{lo}, {hi}) n={len(sub):4d}  RePE2 {gm([r["op"] for r in sub]):6.2f}  wA* {gm([r["w2"] for r in sub]):6.2f}')
    text = '\n'.join(out)
    print(text)
    open('outputs/gpt4omini-repe2-analysis/analysis.txt', 'w').write(text + '\n')
