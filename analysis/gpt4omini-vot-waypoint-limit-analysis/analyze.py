"""VoT (no map) + obstacle-count waypoint cap vs other GPT-4o mini prompts and weighted A* (w=2).
Run from the project root: PYTHONPATH=. python -B outputs/gpt4omini-vot-waypoint-limit-analysis/analyze.py"""
import json, math, random, re, sys
from collections import defaultdict
from llm_astar.__main__ import load_queries
from llm_astar.search import search
Q = dict(load_queries('data/environment_50_30.json'))
def load(path, prompt=None):
    A, L, H = {}, {}, {}
    for l in open(path):
        r = json.loads(l)
        if prompt and r['method'] != 'astar' and r['prompt'] != prompt: continue
        {'astar': A, 'llm': L, 'llm-astar': H}[r['method']][r['sample']] = r
    return A, L, H
U = 'outputs/gpt4omini-table1-upstream-batch-v1/results.jsonl'
W = 'outputs/gpt4omini-waypoint-limit-batch-v1/results.jsonl'
runs = {'vot+lim': load('outputs/gpt4omini-vot-waypoint-limit-batch-v1/results.jsonl'),
        'standard': load(U, 'standard'), 'cot': load(U, 'cot'), 'repe': load(U, 'repe'),
        'std+lim': load(W, 'standard'), 'cot+lim': load(W, 'cot'), 'repe+lim': load(W, 'repe'),
        'cot+votline': load('outputs/gpt4omini-cot-vot-batch-v1/results.jsonl', 'cot')}
A = runs['vot+lim'][0]; S = list(A)
# weighted A* w=2 == LLM-A* with all waypoints filtered
wa = {s: search(Q[s], []) for s in S}
g = lambda xs: 100 * math.exp(sum(map(math.log, xs)) / len(xs))
def boot(xs_by_map, n=2000, seed=0):
    rnd = random.Random(seed); maps = list(xs_by_map); out = []
    for _ in range(n):
        pick = [x for m in (rnd.choice(maps) for _ in maps) for x in xs_by_map[m]]
        out.append(g(pick))
    out.sort(); return out[int(.025*n)], out[int(.975*n)]
def wps(r): return max(len(r['path']) - 2, 0)
print('== overall (LLM-A*; op/storage/len geo-mean % of A*)')
print(f"{'run':12} {'op':>6} {'stor':>6} {'len':>6} {'>A*':>5} {'>2xA*':>6} {'trunc':>5} {'rawvalid':>8} {'wps med':>7} {'wps mean':>8} {'allfiltered':>11}")
for k, (_, L, H) in runs.items():
    ops = [H[s]['operation']/A[s]['operation'] for s in S]
    print(f"{k:12} {g(ops):6.1f} {g([H[s]['storage']/A[s]['storage'] for s in S]):6.1f} {g([H[s]['length']/A[s]['length'] for s in S]):6.1f} "
          f"{sum(o>1 for o in ops):5} {sum(o>2 for o in ops):6} {sum(L[s].get('finish_reason')=='length' for s in S):5} {sum(L[s]['valid'] for s in S):8} "
          f"{sorted(wps(L[s]) for s in S)[500]:7} {sum(wps(L[s]) for s in S)/1000:8.1f} {sum(len(H[s]['llm_output'])==2 for s in S):11}")
ops = [wa[s]['operation']/A[s]['operation'] for s in S]
print(f"{'wA* w=2':12} {g(ops):6.1f} {g([wa[s]['storage']/A[s]['storage'] for s in S]):6.1f} {g([wa[s]['length']/A[s]['length'] for s in S]):6.1f} {sum(o>1 for o in ops):5} {sum(o>2 for o in ops):6}")

print('\n== paired vs vot+lim: geo-mean ratio of op (vot+lim / other), map-bootstrap 95% CI; <100 = vot+lim better')
_, VL, VH = runs['vot+lim']
others = {k: v[2] for k, v in runs.items() if k != 'vot+lim'}; others['wA* w=2'] = wa
for k, H in others.items():
    bym = defaultdict(list)
    for s in S: bym[s.split(':')[0]].append(VH[s]['operation']/H[s]['operation'])
    allr = [x for v in bym.values() for x in v]
    lo, hi = boot(bym)
    print(f"  vs {k:12} {g(allr):6.1f} [{lo:5.1f}, {hi:5.1f}]  wins {sum(r<1 for r in allr)} ties {sum(r==1 for r in allr)} losses {sum(r>1 for r in allr)}")

print('\n== cap compliance (vot+lim)')
cap = {s: len(Q[s]['horizontal_barriers'])+len(Q[s]['vertical_barriers']) for s in S}
ok = [s for s in S if VL[s].get('finish_reason') != 'length' and not VL[s].get('parse_error')]
print('  parsed, not truncated:', len(ok), ' within cap:', sum(wps(VL[s]) <= cap[s] for s in ok), ' over cap:', sum(wps(VL[s]) > cap[s] for s in ok))
print('  over cap by >10 wps:', sum(wps(VL[s]) > cap[s]+10 for s in ok))
for kk in ('std+lim', 'cot+lim', 'repe+lim'):
    _, LL, _ = runs[kk]
    okk = [s for s in S if LL[s].get('finish_reason') != 'length' and not LL[s].get('parse_error')]
    print(f'  {kk}: within cap {sum(wps(LL[s]) <= cap[s] for s in okk)}/{len(okk)}, trunc {sum(LL[s].get("finish_reason")=="length" for s in S)}')

print('\n== by cap value (vot+lim op geo, n, trunc, std+lim op, wA*)')
byc = defaultdict(list)
for s in S: byc[cap[s]].append(s)
SH = runs['std+lim'][2]
for c in sorted(byc):
    ss = byc[c]
    print(f"  cap {c:2}: n={len(ss):3} vot+lim {g([VH[s]['operation']/A[s]['operation'] for s in ss]):6.1f}  std+lim {g([SH[s]['operation']/A[s]['operation'] for s in ss]):6.1f}  wA* {g([wa[s]['operation']/A[s]['operation'] for s in ss]):6.1f}  trunc {sum(VL[s].get('finish_reason')=='length' for s in ss)}")

print('\n== vot+lim split by outcome')
groups = {'truncated': [s for s in S if VL[s].get('finish_reason')=='length'],
          'all filtered (=wA*)': [s for s in S if VL[s].get('finish_reason')!='length' and len(VH[s]['llm_output'])==2],
          'some kept, <=cap': [s for s in S if VL[s].get('finish_reason')!='length' and len(VH[s]['llm_output'])>2 and wps(VL[s])<=cap[s]],
          'some kept, >cap': [s for s in S if VL[s].get('finish_reason')!='length' and len(VH[s]['llm_output'])>2 and wps(VL[s])>cap[s]]}
for k, ss in groups.items():
    if ss: print(f"  {k:22} n={len(ss):4} op {g([VH[s]['operation']/A[s]['operation'] for s in ss]):6.1f}  wA* same samples {g([wa[s]['operation']/A[s]['operation'] for s in ss]):6.1f}  len {g([VH[s]['length']/A[s]['length'] for s in ss]):6.1f}  >2xA* {sum(VH[s]['operation']>2*A[s]['operation'] for s in ss)}")

print('\n== truncation details')
toks = []
for s in groups['truncated']:
    r = json.load(open(f"outputs/gpt4omini-vot-waypoint-limit-batch-v1/responses/{VH[s]['cache_key']}.json"))
    t = r['text']; toks.append(s)
    grid = sum(bool(re.fullmatch(r'\s*\d*\s*[#.SG*o@x+| ]{15,}', x)) for x in t.splitlines())
    lines = t.splitlines(); uniq = len(set(lines))
    if len(toks) <= 6: print(f"  {s} cap {cap[s]} lines {len(lines)} unique {uniq} gridrows {grid} tail: {t[-160:]!r}")
rep = 0
for s in groups['truncated']:
    t = json.load(open(f"outputs/gpt4omini-vot-waypoint-limit-batch-v1/responses/{VH[s]['cache_key']}.json"))['text']
    lines = [x for x in t.splitlines() if x.strip()]
    if lines and len(set(lines[-50:])) < 15: rep += 1
print('  truncated with repetitive last 50 lines:', rep, '/', len(groups['truncated']))
print('  truncated by map:', sorted(defaultdict(int, {m: sum(s.split(':')[0]==m for s in groups['truncated']) for m in {s.split(':')[0] for s in groups['truncated']}}).items(), key=lambda x: -x[1])[:8])
