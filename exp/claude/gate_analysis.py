"""Gated LLM use: take LLM-A* only when the (filtered) LLM path is itself collision-free, else weighted A*.

Pure post-hoc combination of per-sample results from offline_improve.py (no new search, no inference).
The gate uses only information available before search (collision check of the LLM waypoint chain).
"""
import sys
sys.dont_write_bytecode = True
import collections, gzip, json, math, random
from pathlib import Path

RES = Path(__file__).resolve().parent / 'results'
rows = [json.loads(l) for l in gzip.open(RES / 'per_sample.jsonl.gz', 'rt')]
A = {(r['sample'], r['dedupe'], r['tie'], r['variant']): r for r in rows if r['kind'] == 'astar'}
V = collections.defaultdict(dict)
L = collections.defaultdict(dict)
for r in rows:
    if r['kind'] == 'llm-astar':
        V[(r['variant'], r['prompt'])][r['sample']] = r
    elif r['kind'] == 'llm':
        L[(r['variant'], r['prompt'])][r['sample']] = r


def gm(v):
    return 100 * math.exp(sum(map(math.log, v)) / len(v))


def boot(diffs, n=2000, seed=0):
    maps = list(diffs)
    rng = random.Random(seed)
    tot = [x for v in diffs.values() for x in v]
    point = sum(tot) / len(tot)
    bs = []
    for _ in range(n):
        vals = [x for m in (rng.choice(maps) for _ in maps) for x in diffs[m]]
        bs.append(sum(vals) / len(vals))
    bs.sort()
    return [100 * math.exp(point), 100 * math.exp(bs[int(.025 * n)]), 100 * math.exp(bs[int(.975 * n)])]


out = []
for variant, fallback in [('official', 'wA*2.0'), ('s:wg2_wt0.5', 'wA*2.0'),
                          ('c:progress+wg1.5_wt0.5+tie_h+dedupe', 'wA*2.0')]:
    for prompt in ('standard', 'cot', 'repe'):
        for gate in ('official_filter', 'snap+pull', 'oracle'):
            op, ln, used = [], [], 0
            dop, dln = collections.defaultdict(list), collections.defaultdict(list)
            for s, r in V[(variant, prompt)].items():
                acc = (r['dedupe'], r['tie'])
                b = A[(s, *acc, 'wA*1.0')]
                a = A[(s, *acc, fallback)]
                if gate == 'oracle':
                    use = (r['op'], r['len']) < (a['op'], a['len'])
                else:
                    use = L[(gate, prompt)][s]['valid']
                used += use
                x = r if use else a
                op.append(x['op'] / b['op']); ln.append(x['len'] / b['len'])
                m = s.split(':')[0]
                dop[m].append(math.log(x['op'] / a['op'])); dln[m].append(math.log(x['len'] / a['len']))
            out.append(dict(variant=variant, prompt=prompt, gate=gate, fallback=fallback, used_llm=used,
                            op=gm(op), len=gm(ln),
                            op_vs_fallback_ci95=boot(dop), len_vs_fallback_ci95=boot(dln)))
(RES / 'gate.json').write_text(json.dumps(out, indent=2))
for o in out:
    print(f"{o['variant']:38s} {o['prompt']:8s} {o['gate']:15s} used={o['used_llm']:4d} op {o['op']:.2f} len {o['len']:.2f} "
          f"op/fallback {o['op_vs_fallback_ci95'][0]:.2f} [{o['op_vs_fallback_ci95'][1]:.2f},{o['op_vs_fallback_ci95'][2]:.2f}] "
          f"len/fallback {o['len_vs_fallback_ci95'][0]:.2f} [{o['len_vs_fallback_ci95'][1]:.2f},{o['len_vs_fallback_ci95'][2]:.2f}]")
