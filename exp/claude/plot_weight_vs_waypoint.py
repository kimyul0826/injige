"""Report figure: LLM-A* gain comes from heuristic weighting, not waypoint quality.
(a) operation ratio (GM, % of A*) per method, grouped: no waypoints / random waypoints / LLM waypoints.
(b) decomposition for the standard prompt: A* -> wA*2 (LLM-A* formula with 0 waypoints) -> + same-count random
    waypoints -> LLM waypoints instead of random. Data: results/per_map_compare.json, results/dwa_check.json."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
pm = json.loads((HERE / 'results' / 'per_map_compare.json').read_text())
gp = json.loads((HERE / 'results' / 'per_map_compare_gpt4omini.json').read_text())
dw = json.loads((HERE / 'results' / 'dwa_check.json').read_text())
C = {'none': '#2a78d6', 'rand': '#eb6834', 'llm': '#1baf7a', 'gpt': '#6250d6'}
INK, MUTED = '#0b0b0b', '#52514e'
rows = [('A*', 100.0, 'none'), ('dWA* (paper, reported)', 60.91, 'none'), ('dWA* (our reading)', dw['mult']['op'], 'none'),
        ('wA* w=2  (= LLM-A*, 0 waypoints)', pm['wA2']['op_gm'], 'none'),
        ('random, 1 waypoint', pm['rand1']['op_gm'], 'rand'), ('random, 2 waypoints', pm['rand2']['op_gm'], 'rand'),
        ('random, same count as Llama (standard)', pm['match_standard']['op_gm'], 'rand'),
        ('random, same count as GPT-4o mini (standard)', gp['match_standard']['op_gm'], 'rand'),
        ('LLM-A* Llama-3 8B standard', pm['llm_standard']['op_gm'], 'llm'), ('LLM-A* Llama-3 8B CoT', pm['llm_cot']['op_gm'], 'llm'),
        ('LLM-A* Llama-3 8B RePE', pm['llm_repe']['op_gm'], 'llm'),
        ('LLM-A* GPT-4o mini standard', gp['llm_standard']['op_gm'], 'gpt'), ('LLM-A* GPT-4o mini CoT', gp['llm_cot']['op_gm'], 'gpt'),
        ('LLM-A* GPT-4o mini RePE', gp['llm_repe']['op_gm'], 'gpt')]

fig, (a, b, b2) = plt.subplots(1, 3, figsize=(19, 6.2), gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
for ax in (a, b, b2):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#c9c8c2'); ax.spines['bottom'].set_color('#c9c8c2')
    ax.tick_params(colors=MUTED)
y = list(range(len(rows)))[::-1]
a.barh(y, [r[1] for r in rows], color=[C[r[2]] for r in rows], height=.62, edgecolor='white', linewidth=2)
for yi, (name, v, g) in zip(y, rows):
    a.text(v + 1.2, yi, f'{v:.1f}', va='center', fontsize=9, color=INK)
a.set_yticks(y); a.set_yticklabels([r[0] for r in rows], fontsize=9, color=INK)
a.set_xlabel('operations, % of A* (geometric mean, 1000 maps)', color=MUTED); a.set_xlim(0, 112)
a.grid(axis='x', color='#e6e5e0', lw=.8); a.set_axisbelow(True)
for g, lab in (('none', 'no waypoints'), ('rand', 'random waypoints (start-goal rectangle)'), ('llm', 'Llama-3 8B waypoints')):
    pass
from matplotlib.patches import Patch
a.legend(handles=[Patch(color=C[g], label=lab) for g, lab in (('none', 'no waypoints'),
         ('rand', 'random waypoints (start-goal rectangle)'), ('llm', 'Llama-3 8B waypoints'), ('gpt', 'GPT-4o mini waypoints'))],
         fontsize=8.5, loc='upper center', bbox_to_anchor=(0.45, -0.11), ncol=2, frameon=False)
a.set_title('(a) operation ratio by method', loc='left', fontsize=11, color=INK)

def waterfall(b, src, name, col):
    steps = [('A*', 100.0), ('weight 2\n(0 waypoints)', src['wA2']['op_gm']), ('+ random\nwaypoints', src['match_standard']['op_gm']),
             (f'{name}\ninstead of random', src['llm_standard']['op_gm'])]
    cols = [MUTED, C['none'], C['rand'], col]
    prev = None
    for i, ((lab, v), c) in enumerate(zip(steps, cols)):
        if prev is None:
            b.bar(i, v, color='#c9c8c2', width=.6)
            b.text(i, v + 1.5, f'{v:.1f}', ha='center', fontsize=10, color=INK)
        else:
            lo, hi = sorted((prev, v))
            b.bar(i, hi - lo, bottom=lo, color=c, width=.6, edgecolor='white', linewidth=2)
            b.text(i, hi + 1.5, f'{v - prev:+.1f}', ha='center', fontsize=10, color=INK, fontweight='bold')
            b.plot([i - 1.3, i - .3], [prev, prev], color=MUTED, lw=.8, ls=':')
        prev = v
    b.set_xticks(range(len(steps))); b.set_xticklabels([s[0] for s in steps], fontsize=9, color=INK)
    b.set_ylabel('operations, % of A*', color=MUTED); b.set_ylim(0, 112)
    b.grid(axis='y', color='#e6e5e0', lw=.8); b.set_axisbelow(True)
    b.set_title(f'{name} standard: LLM-A* = {steps[-1][1]:.1f}', loc='left', fontsize=11, color=INK)


waterfall(b, pm, 'Llama-3 8B', C['llm']); b.set_title('(b) ' + b.get_title(loc='left'), loc='left', fontsize=11, color=INK)
waterfall(b2, gp, 'GPT-4o mini', C['gpt']); b2.set_title('(c) ' + b2.get_title(loc='left'), loc='left', fontsize=11, color=INK)
fig.tight_layout()
out = HERE / 'figs' / 'weight_vs_waypoint.png'; fig.savefig(out, dpi=150); print(out)
