"""Stage 2 of failure_case_data.py (needs only matplotlib): draw the picked failure cases.
Run with any Python that has matplotlib, e.g. the system anaconda: python3 exp/claude/plot_failure_cases.py
Output: exp/claude/figs/failure_cases.png (all cases) and report/fig_failures.png (two cases for the report)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / 'results' / 'failure_case_plots.json').read_text())
TITLES = {'runaway': 'Official LLM-A* runaway (Few-shot)', 'valid_hurts': 'Collision-free LLM path still costs more',
          'detour': 'Endpoint prompt: detour chain', 'false_valid': 'RePE2: "Valid" but cuts a corner',
          'on_barrier': 'RePE2: waypoint on a barrier'}


def draw(ax, c, name):
    q = c['query']
    for y, a, b in q['horizontal_barriers']:
        ax.plot([a, b], [y, y], color='black', lw=2.2)
    for x, a, b in q['vertical_barriers']:
        ax.plot([x, x], [a, b], color='black', lw=2.2)
    pw = c['pops_w2']; pl = c['pops_llm']
    ax.scatter([p[0] for p in pl], [p[1] for p in pl], s=5, color='tab:red', alpha=.45, label=f'LLM-A* expanded ({len(pl)})')
    ax.scatter([p[0] for p in pw], [p[1] for p in pw], s=5, color='tab:blue', alpha=.8, label=f'weighted A* w=2 expanded ({len(pw)})')
    lp = c['llm_path']
    ax.plot([p[0] for p in lp], [p[1] for p in lp], '--', color='tab:orange', lw=1.4, marker='o', ms=3, label='LLM path')
    for (a, b) in c['bad_segments']:
        ax.plot([a[0], b[0]], [a[1], b[1]], color='magenta', lw=2.6, alpha=.8)
    t = c['targets'][1:-1]
    if t:
        ax.scatter([p[0] for p in t], [p[1] for p in t], s=40, marker='x', color='darkgreen', zorder=5, label='waypoints kept by filter')
    s, g = q['start'], q['goal']
    ax.scatter([s[0]], [s[1]], s=60, marker='s', color='green', zorder=6)
    ax.scatter([g[0]], [g[1]], s=80, marker='*', color='gold', edgecolor='k', zorder=6)
    ax.set_xlim(q['range_x'][0], q['range_x'][1] - 1); ax.set_ylim(q['range_y'][0], q['range_y'][1] - 1)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{TITLES[name]}\n{c['sample']}: op {100*c['op']:.0f}% vs {100*c['w2']:.0f}% (of A*)", fontsize=8)


names = list(data)
fig, axes = plt.subplots(2, 3, figsize=(13, 6.2))
for ax, n in zip(axes.flat, names):
    draw(ax, data[n], n)
axes.flat[-1].axis('off')
h, l = axes.flat[0].get_legend_handles_labels()
axes.flat[-1].legend(h, ['LLM-A* expanded states', 'weighted A* (w=2) expanded states', 'LLM path (magenta: colliding segment)',
                         'waypoints kept by the filter'], loc='center', fontsize=9, frameon=False)
fig.tight_layout()
(HERE / 'figs').mkdir(exist_ok=True)
fig.savefig(HERE / 'figs' / 'failure_cases.png', dpi=160)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
for ax, n in zip(axes, ['runaway', 'false_valid']):
    draw(ax, data[n], n)
fig.tight_layout()
out = HERE.parents[1] / 'report' / 'fig_failures.png'
fig.savefig(out, dpi=200)
print('saved', HERE / 'figs' / 'failure_cases.png', out)
