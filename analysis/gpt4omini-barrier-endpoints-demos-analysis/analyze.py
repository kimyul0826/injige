"""Endpoint-candidate prompt (endpoint_demos+barrier_endpoints) vs original GPT-4o mini Table 1 batch.
Run: PYTHONPATH=. python -B outputs/gpt4omini-barrier-endpoints-demos-analysis/analyze.py"""
import json, math, random
from collections import Counter, defaultdict
from pathlib import Path

from llm_astar.__main__ import load_queries
from llm_astar.models import parse_path
from llm_astar.prompt_variants import barrier_endpoints

NEW = Path('outputs/gpt4omini-barrier-endpoints-demos-batch-v1')
OLD = Path('outputs/gpt4omini-table1-upstream-batch-v1')
PROMPTS = ('standard', 'cot', 'repe')
qs = dict(load_queries(NEW / 'input.json'))


def load(O):
    rows = [json.loads(l) for l in (O / 'results.jsonl').read_text().splitlines()]
    return {(r['sample'], r['method'], r['prompt']): r for r in rows}


new, old = load(NEW), load(OLD)
base = {s: new[(s, 'astar', '-')] for s in qs} if (next(iter(qs)), 'astar', '-') in new else None
if base is None:
    base = {k[0]: r for k, r in new.items() if k[1] == 'astar'}
ratio = lambda R, s, p, m='operation': R[(s, 'llm-astar', p)][m] / base[s][m]
gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
S = sorted(qs)
random.seed(0)
out = []


def boot(v, n=2000):
    logs = [math.log(x) for x in v]
    m = [math.exp(sum(random.choice(logs) for _ in logs) / len(logs)) * 100 for _ in range(n)]
    m.sort()
    return m[int(.025 * n)], m[int(.975 * n)]


out.append('## LLM-A* (1000 problems, A*=100)')
for p in PROMPTS:
    on = [ratio(old, s, p) for s in S]; nn = [ratio(new, s, p) for s in S]
    paired = [b / a for a, b in zip(on, nn)]
    lo, hi = boot(paired)
    better = sum(b < a for a, b in zip(on, nn)); worse = sum(b > a for a, b in zip(on, nn))
    sto_o = gm([ratio(old, s, p, 'storage') for s in S]); sto_n = gm([ratio(new, s, p, 'storage') for s in S])
    out.append(f'{p:8s} op {gm(on):6.2f} -> {gm(nn):6.2f}  paired new/old {gm(paired):6.1f} [{lo:.1f}, {hi:.1f}]  '
               f'better {better} worse {worse} tie {len(S)-better-worse}  sto {sto_o:.2f} -> {sto_n:.2f}')

out.append('\n## LLM only success')
for p in PROMPTS:
    v = lambda R: sum(R[(s, 'llm', p)]['valid'] for s in S) / 10
    out.append(f'{p:8s} {v(old):.1f}% -> {v(new):.1f}%')

out.append('\n## endpoint adherence (new run)')
cases = defaultdict(list)
for p in PROMPTS:
    c = Counter()
    for s in S:
        q = qs[s]; E = {tuple(e) for e in barrier_endpoints(q)}
        r = new[(s, 'llm-astar', p)]
        text = json.loads((NEW / 'responses' / f"{r['cache_key']}.json").read_text())['text']
        try:
            P = parse_path(text, 'upstream')
        except Exception:
            P = []; c['parse_fail'] += 1
        mid = [x for x in P if x not in (tuple(q['start']), tuple(q['goal']))]
        ok = all(x in E for x in mid)
        c['wp'] += len(mid); c['in'] += sum(x in E for x in mid); c['all_in'] += ok; c['zero'] += not mid
        c['n_E'] += len(E)
        cases[p].append((ratio(new, s, p), ratio(old, s, p), s, ok, len(mid)))
    out.append(f"{p:8s} wp/resp {c['wp']/1000:.2f}  in-list {c['in']}/{c['wp']} ({100*c['in']/max(1,c['wp']):.1f}%)  "
               f"all-in-list {c['all_in']/10:.1f}%  no-waypoint {c['zero']}  candidates/map {c['n_E']/1000:.2f}")
    a = [x for x in cases[p] if x[3]]; b = [x for x in cases[p] if not x[3]]
    out.append(f"         op when all in list {gm([x[0] for x in a]):.2f} (n={len(a)}, old {gm([x[1] for x in a]):.2f})  "
               f"otherwise {gm([x[0] for x in b]):.2f} (n={len(b)}, old {gm([x[1] for x in b]):.2f})")
    worst = sorted(cases[p], key=lambda x: -x[0] / x[1])[:5]
    out.append('         worst new/old: ' + ', '.join(f'{s} {100*n:.0f}% (old {100*o:.0f}%)' for n, o, s, _, _ in worst))
    big = sum(x[0] > 2 for x in cases[p]); big_o = sum(x[1] > 2 for x in cases[p])
    out.append(f'         op > 200% of A*: new {big}, old {big_o}')

text = '\n'.join(out)
print(text)
Path('outputs/gpt4omini-barrier-endpoints-demos-analysis/analysis.txt').write_text(text + '\n')
