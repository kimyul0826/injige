"""Draw sample maps: walls, optimal A* path, turning cells, string-pulled oracle waypoints.
Output: figs/oracle_waypoints.png"""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o   # noqa: E402

PICKS = [('original', 3), ('original', 11), ('dense', 5), ('rooms', 2), ('utrap', 4), ('maze', 1)]


def oracle(env, grid):
    opt = o.search(env, grid, None, dedupe=True)['path']
    turns = [opt[i] for i in range(1, len(opt) - 1)
             if (opt[i][0] - opt[i-1][0], opt[i][1] - opt[i-1][1]) != (opt[i+1][0] - opt[i][0], opt[i+1][1] - opt[i][1])]
    return opt, turns, o.string_pull(env, [env.start] + turns + [env.goal])


def main():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    for ax, (fam, idx) in zip(axes.flat, PICKS):
        q = json.loads((HERE / 'bench' / f'{fam}.json').read_text())[idx]
        env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
        grid = o.Grid(env)
        opt, turns, wp = oracle(env, grid)
        for a, b in env.walls:
            ax.plot([a[0], b[0]], [a[1], b[1]], color='#333', lw=2.2, solid_capstyle='butt')
        ax.plot(*zip(*opt), color='#7aa6d6', lw=1.2, label='optimal A* path (grid)')
        ax.scatter(*zip(*turns), s=12, color='#7aa6d6', zorder=3, label=f'turning cells ({len(turns)})')
        ax.plot(*zip(*wp), color='#d0542c', lw=1.6, ls='--', label='string-pulled chain')
        if len(wp) > 2:
            ax.scatter(*zip(*wp[1:-1]), s=70, color='#d0542c', edgecolor='k', zorder=4,
                       label=f'oracle waypoints ({len(wp) - 2})')
        ax.scatter(*env.start, s=90, marker='o', color='#2a9d4b', edgecolor='k', zorder=5, label='start')
        ax.scatter(*env.goal, s=120, marker='*', color='#e9b10c', edgecolor='k', zorder=5, label='goal')
        ax.set_xlim(env.rx[0] - 1, env.rx[1]); ax.set_ylim(env.ry[0] - 1, env.ry[1])
        ax.set_aspect('equal'); ax.grid(alpha=.15)
        ax.set_title(f'{fam} #{idx}  ({env.rx[1]}x{env.ry[1]}, {len(env.walls)} walls)  need={len(wp) - 2}')
        ax.legend(fontsize=7, loc='best', framealpha=.85)
    fig.tight_layout()
    out = HERE / 'figs' / 'oracle_waypoints.png'
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == '__main__':
    main()
