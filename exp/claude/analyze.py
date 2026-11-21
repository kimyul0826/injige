"""Summaries for offline_improve.py output. Writes only under exp/claude/results."""
import sys
sys.dont_write_bytecode = True
import collections, csv, gzip, json, math, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
rows = [json.loads(l) for l in gzip.open(RES / 'per_sample.jsonl.gz', 'rt')]


def gm(v):
    v = [x for x in v if x is not None]
    return 100 * math.exp(sum(math.log(x) for x in v) / len(v)) if v else None


astar = {(r['sample'], r['dedupe'], r['tie'], float(r['variant'][3:])): r for r in rows if r['kind'] == 'astar'}
WS = sorted({k[3] for k in astar})
ACCS = sorted({(k[1], k[2]) for k in astar})

# ---------------------------------------------------------------- weighted A* curves (fair: vs A* w=1, same acc)
curve = {}
curve_rows = []
for acc in ACCS:
    pts = []
    for w in WS:
        op, sto, ln, aop = [], [], [], []
        for (s, d, t, ww), r in astar.items():
            if (d, t) != acc or ww != w:
                continue
            b = astar[(s, d, t, 1.0)]
            o = astar[(s, False, 'node', 1.0)]
            op.append(r['op'] / b['op']); sto.append(r['sto'] / b['sto']); ln.append(r['len'] / b['len'])
            aop.append(r['op'] / o['op'])
        pts.append((gm(ln), gm(op), w))
        curve_rows.append(dict(dedupe=acc[0], tie=acc[1], w=w, op_fair=gm(op), sto_fair=gm(sto), len=gm(ln),
                               op_vs_official_astar=gm(aop)))
    curve[acc] = sorted(pts)


def wa_op_at(acc, length):
    """Weighted-A* operation ratio interpolated at the same path-length ratio (fair, same acc)."""
    pts = curve[acc]
    if length <= pts[0][0]:
        return pts[0][1]
    for (l0, o0, _), (l1, o1, _) in zip(pts, pts[1:]):
        if l0 <= length <= l1:
            return o0 + (o1 - o0) * (length - l0) / (l1 - l0) if l1 > l0 else min(o0, o1)
    return pts[-1][1]


# ---------------------------------------------------------------- LLM-A* variants
groups = collections.defaultdict(list)
for r in rows:
    if r['kind'] == 'llm-astar':
        groups[(r['variant'], r['prompt'])].append(r)
order = list(dict.fromkeys(r['variant'] for r in rows if r['kind'] == 'llm-astar'))

summary = []
for (v, p), rs in groups.items():
    acc = (rs[0]['dedupe'], rs[0]['tie'])
    op, sto, ln, aop, asto = [], [], [], [], []
    for r in rs:
        b = astar[(r['sample'], acc[0], acc[1], 1.0)]
        o = astar[(r['sample'], False, 'node', 1.0)]
        op.append(r['op'] / b['op']); sto.append(r['sto'] / b['sto']); ln.append(r['len'] / b['len'])
        aop.append(r['op'] / o['op']); asto.append(r['sto'] / o['sto'])
    row = dict(variant=v, prompt=p, group=rs[0]['group'], dedupe=acc[0], tie=acc[1], n=len(rs),
               op=gm(op), sto=gm(sto), len=gm(ln), op_vs_official_astar=gm(aop), sto_vs_official_astar=gm(asto),
               op_gt_astar=sum(x > 1 for x in op), op_gt_2x=sum(x > 2 for x in op),
               len_gt_110=sum(x > 1.1 for x in ln), max_op=max(op), max_len=max(ln))
    row['wA*_op_at_same_len'] = wa_op_at(acc, row['len'])
    row['op_minus_wA*'] = row['op'] - row['wA*_op_at_same_len']
    summary.append(row)
summary.sort(key=lambda r: (order.index(r['variant']), ['standard', 'cot', 'repe', 'ensemble'].index(r['prompt'])))

# ---------------------------------------------------------------- cluster bootstrap (maps) for key contrasts
def bootstrap(variant, prompt, ref, n=2000, seed=0):
    """Paired log-ratio difference of operation (variant vs ref) with map-cluster bootstrap."""
    rs = {r['sample']: r for r in groups[(variant, prompt)]}
    acc = (next(iter(rs.values()))['dedupe'], next(iter(rs.values()))['tie'])
    diffs = collections.defaultdict(list)
    for s, r in rs.items():
        if ref[0] == 'wA*':
            refop = astar[(s, acc[0], acc[1], ref[1])]['op']
        else:
            refop = {x['sample']: x for x in groups[ref]}[s]['op'] if False else None
        if ref[0] != 'wA*':
            refop = ref_lookup[ref][s]
        diffs[s.split(':')[0]].append(math.log(r['op'] / refop))
    maps = list(diffs)
    point = sum(sum(v) for v in diffs.values()) / sum(len(v) for v in diffs.values())
    rng = random.Random(seed)
    boots = []
    for _ in range(n):
        pick = [rng.choice(maps) for _ in maps]
        vals = [x for m in pick for x in diffs[m]]
        boots.append(sum(vals) / len(vals))
    boots.sort()
    return 100 * math.exp(point), 100 * math.exp(boots[int(.025 * n)]), 100 * math.exp(boots[int(.975 * n)])


ref_lookup = {k: {r['sample']: r['op'] for r in v} for k, v in groups.items()}

# ---------------------------------------------------------------- LLM-only
lgroups = collections.defaultdict(list)
for r in rows:
    if r['kind'] == 'llm':
        lgroups[(r['variant'], r['prompt'])].append(r)
raw_valid = {(r['sample'], r['prompt']): r['valid'] for r in rows if r['kind'] == 'llm' and r['variant'] == 'raw'}
llm_summary = []
lorder = list(dict.fromkeys(r['variant'] for r in rows if r['kind'] == 'llm'))
for (v, p), rs in lgroups.items():
    ln, common = [], []
    for r in rs:
        o = astar[(r['sample'], False, 'node', 1.0)]
        if r['valid']:
            ln.append(r['len'] / o['len'])
            if raw_valid[(r['sample'], p)]:
                common.append(r['len'] / o['len'])
    llm_summary.append(dict(variant=v, prompt=p, valid_percent=sum(r['valid'] for r in rs) / len(rs) * 100,
                            len_valid=gm(ln), len_on_raw_valid=gm(common), n_raw_valid_kept=len(common)))
# Ensemble across prompts for LLM-only: shortest valid among the 3 responses (needs 3 LLM calls).
for v in ('snap+pull', 'hybrid:repair+pull'):
    by_s = collections.defaultdict(list)
    for p in ('standard', 'cot', 'repe'):
        for r in lgroups[(v, p)]:
            by_s[r['sample']].append(r)
    ok = []
    for s, rs in by_s.items():
        val = [r['len'] for r in rs if r['valid']]
        ok.append(min(val) / astar[(s, False, 'node', 1.0)]['len'] if val else None)
    llm_summary.append(dict(variant='ens3:' + v, prompt='ensemble', valid_percent=sum(x is not None for x in ok) / 10,
                            len_valid=gm(ok), len_on_raw_valid=None, n_raw_valid_kept=None))
llm_summary.sort(key=lambda r: (lorder.index(r['variant'].replace('ens3:', '')),
                                r['prompt'] == 'ensemble', ['standard', 'cot', 'repe', 'ensemble'].index(r['prompt'])))

# ---------------------------------------------------------------- write
def write_csv(path, data):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(data[0]))
        w.writeheader(); w.writerows(data)


write_csv(RES / 'summary_llm_astar.csv', summary)
write_csv(RES / 'summary_llm_only.csv', llm_summary)
write_csv(RES / 'wastar_curve.csv', curve_rows)

key = {}
for p in ('standard', 'cot', 'repe'):
    for v in ('official', 'c:snap+prune+progress+tie_h+dedupe', 's:switch_progress', 'wp:snap', 'wp:prune_detour1.5'):
        if (v, p) in groups:
            key[f'{v}|{p}|vs_wA*2'] = bootstrap(v, p, ('wA*', 2.0))
            if v != 'official':
                key[f'{v}|{p}|vs_official'] = bootstrap(v, p, ('official', p)) if (v.startswith('wp:') or v == 's:switch_progress') else None
(RES / 'bootstrap.json').write_text(json.dumps({k: v for k, v in key.items() if v}, indent=2))

# plot
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, acc in zip(axes, [(False, 'node'), (True, 'h')]):
        pts = curve[acc]
        ax.plot([x[0] for x in pts], [x[1] for x in pts], 'k-o', label='weighted A* (no LLM)')
        for l, o, w in pts:
            ax.annotate(f'w={w:g}', (l, o), fontsize=7, xytext=(3, 3), textcoords='offset points')
        for r in summary:
            if (r['dedupe'], r['tie']) != acc:
                continue
            mk = {'standard': 'o', 'cot': 's', 'repe': '^', 'ensemble': '*'}[r['prompt']]
            col = {'baseline': 'red', 'waypoint': 'tab:blue', 'search': 'tab:orange', 'combo': 'tab:green',
                   'ensemble': 'tab:purple'}[r['group']]
            ax.scatter(r['len'], r['op'], marker=mk, c=col, s=28, alpha=.8)
        ax.set_xlabel('path length ratio vs A* (%)'); ax.set_ylabel('operation ratio vs A* (%)')
        ax.set_title(f'accounting dedupe={acc[0]}, tie={acc[1]}'); ax.grid(alpha=.3)
    axes[0].legend(handles=[plt.Line2D([], [], color='k', marker='o', label='weighted A*')] +
                   [plt.Line2D([], [], ls='', marker='o', color=c, label=g) for g, c in
                    [('official', 'red'), ('waypoint', 'tab:blue'), ('search', 'tab:orange'), ('combo', 'tab:green'),
                     ('ensemble', 'tab:purple')]], fontsize=8)
    fig.tight_layout(); fig.savefig(RES / 'pareto.png', dpi=130)
except ImportError:
    pass

print('done', len(summary), len(llm_summary))
