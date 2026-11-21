"""Summarise logic_variants.py output: LLM vs no-LLM control vs weighted-A* curve (dedupe accounting)."""
import sys
sys.dont_write_bytecode = True
import collections, csv, gzip, json, math, random
from pathlib import Path

RES = Path(__file__).resolve().parent / 'results'
name = sys.argv[1] if len(sys.argv) > 1 else 'logic_per_sample.jsonl.gz'
rows = [json.loads(l) for l in gzip.open(RES / name, 'rt')]
base = {r['sample']: r for r in rows if r['method'] == 'astar'}
groups = collections.defaultdict(dict)
for r in rows:
    if r['method'] != 'astar':
        groups[(r['method'], r['prompt'])][r['sample']] = r


def gm(v):
    return 100 * math.exp(sum(map(math.log, v)) / len(v))


def stats(d):
    op = [r['op'] / base[s]['op'] for s, r in d.items()]
    sto = [r['sto'] / base[s]['sto'] for s, r in d.items()]
    ln = [r['len'] / base[s]['len'] for s, r in d.items()]
    return dict(op=gm(op), sto=gm(sto), len=gm(ln), op_gt_astar=sum(x > 1 for x in op),
                op_gt_2x=sum(x > 2 for x in op), max_op=max(op), max_len=max(ln), n=len(op))


curve = sorted((stats(groups[(f'wA*{w}', '-')])['len'], stats(groups[(f'wA*{w}', '-')])['op'], w)
               for w in (1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0))
curve = [(100.0, 100.0, 1.0)] + curve


def curve_op(length):
    if length <= curve[0][0]:
        return curve[0][1]
    for (l0, o0, _), (l1, o1, _) in zip(curve, curve[1:]):
        if l0 <= length <= l1:
            return o0 + (o1 - o0) * (length - l0) / (l1 - l0) if l1 > l0 else min(o0, o1)
    return curve[-1][1]


def boot_vs_control(method, prompt, n=2000, seed=0):
    """Map-cluster bootstrap of geometric-mean op ratio LLM / no-LLM control."""
    llm, ctl = groups[(method, prompt)], groups[(method + '|noLLM', '-')]
    by = collections.defaultdict(list)
    for s, r in llm.items():
        by[s.split(':')[0]].append(math.log(r['op'] / ctl[s]['op']))
    maps = list(by)
    pt = sum(x for v in by.values() for x in v) / sum(len(v) for v in by.values())
    rng = random.Random(seed)
    bs = sorted(sum(x for m in (rng.choice(maps) for _ in maps) for x in by[m]) /
                sum(len(by[m]) for m in (rng.choice(maps) for _ in [0]) ) if False else
                (lambda pick: sum(x for m in pick for x in by[m]) / sum(len(by[m]) for m in pick))
                ([rng.choice(maps) for _ in maps]) for _ in range(n))
    return [100 * math.exp(pt), 100 * math.exp(bs[int(.025 * n)]), 100 * math.exp(bs[int(.975 * n)])]


out = []
print('weighted A* curve (dedupe):', ', '.join(f'w={w}: {o:.1f}/{l:.2f}' for l, o, w in curve))
methods = sorted({m for m, p in groups if not m.startswith('wA*') and not m.endswith('|noLLM')})
for m in methods:
    c = stats(groups[(m + '|noLLM', '-')])
    print(f'\n{m}  control(noLLM): op {c["op"]:.2f} sto {c["sto"]:.2f} len {c["len"]:.2f} '
          f'gap {c["op"] - curve_op(c["len"]):+.2f} max_len {c["max_len"]:.3f}')
    out.append(dict(method=m, prompt='noLLM', **c, curve_gap=c['op'] - curve_op(c['len'])))
    for p in ('standard', 'cot', 'repe', 'oracle'):
        s = stats(groups[(m, p)])
        b = boot_vs_control(m, p)
        print(f'   {p:8s} op {s["op"]:6.2f} sto {s["sto"]:6.2f} len {s["len"]:7.2f} gap {s["op"] - curve_op(s["len"]):+6.2f} '
              f'op>2x {s["op_gt_2x"]:3d} max_op {s["max_op"]:.2f} max_len {s["max_len"]:.3f} '
              f'| LLM/control op {b[0]:.1f} [{b[1]:.1f},{b[2]:.1f}]')
        out.append(dict(method=m, prompt=p, **s, curve_gap=s['op'] - curve_op(s['len']),
                        op_vs_control=b[0], op_vs_control_lo=b[1], op_vs_control_hi=b[2]))
with open(RES / name.replace('.jsonl.gz', '_summary.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[-1]))
    w.writeheader()
    for r in out:
        w.writerow({k: r.get(k) for k in out[-1]})
with open(RES / name.replace('.jsonl.gz', '_curve.json'), 'w') as fh:
    json.dump([dict(w=w, op=o, len=l) for l, o, w in curve], fh, indent=2)
