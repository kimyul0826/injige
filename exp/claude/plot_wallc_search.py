"""Actual search on the example maps: A* (euclid w=1), wA*2 (euclid w=2), wallC (w=1.5, reopening).
Shows expanded nodes, nodes whose h was evaluated, and the returned path. Output: figs/wallc_search.png"""
import sys
sys.dont_write_bytecode = True
import heapq, json, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o, wall_component_h as wc   # noqa: E402
D = math.dist
PICKS = [('utrap', 4), ('rooms', 2)]


def run(env, grid, h, w):
    """Same loop as wall_component_h.astar_reopen, but also returns path / expanded / h-evaluated sets."""
    seen = set()
    def H(s):
        seen.add(s); return h(s)
    start, goal = env.start, env.goal
    g, parent = {start: 0.0}, {start: start}
    q, closed, expanded, ops = [(w * H(start), start)], set(), [], 0
    while q:
        _, s = heapq.heappop(q)
        if s in closed:
            continue
        ops += 1; closed.add(s); expanded.append(s)
        if s == goal:
            break
        for dx, dy in o.MOTIONS:
            n = (s[0] + dx, s[1] + dy)
            if grid.coll(s, n):
                continue
            ng = g[s] + D(s, n)
            if ng < g.get(n, math.inf):
                g[n], parent[n] = ng, s
                closed.discard(n)
                heapq.heappush(q, (ng + w * H(n), n))
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return dict(path=path[::-1], expanded=expanded, seen=seen, op=ops,
                len=sum(D(a, b) for a, b in zip(path, path[1:])))


fig, axes = plt.subplots(2, 3, figsize=(19, 8.6))
for row, (fam, idx) in enumerate(PICKS):
    q = json.loads((HERE / 'bench' / f'{fam}.json').read_text())[idx]
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely'); grid = o.Grid(env)
    eu = lambda s: D(s, env.goal)
    cfgs = [('A* (euclid, w=1)', eu, 1.0), ('wA*2 (euclid, w=2)', eu, 2.0), ('wallC (w=1.5)', wc.make_wallC(env), 1.5)]
    base = None
    for ax, (name, h, w) in zip(axes[row], cfgs):
        r = run(env, grid, h, w); base = base or r
        for a, b in env.walls:
            ax.plot([a[0], b[0]], [a[1], b[1]], color='#333', lw=2.4)
        sx = [p for p in r['seen'] if p not in set(r['expanded'])]
        if sx: ax.scatter(*zip(*sx), s=9, color='#bcd3ea', marker='s', label=f'h evaluated only ({len(sx)})')
        ax.scatter(*zip(*r['expanded']), s=9, color='#4a7fbf', marker='s', label=f'expanded ({r["op"]})')
        ax.plot(*zip(*r['path']), color='#d0542c', lw=2.2, label=f'path len {r["len"]:.1f}')
        ax.scatter(*env.start, s=90, color='#2a9d4b', edgecolor='k', zorder=5)
        ax.scatter(*env.goal, s=140, marker='*', color='#e9b10c', edgecolor='k', zorder=5)
        ax.set_xlim(env.rx[0] - 1, env.rx[1]); ax.set_ylim(env.ry[0] - 1, env.ry[1]); ax.set_aspect('equal')
        ax.set_title(f'{fam} #{idx} | {name} | op {100 * r["op"] / base["op"]:.0f}% of A*, len {100 * r["len"] / base["len"]:.1f}%')
        ax.legend(fontsize=7.5, loc='lower left', framealpha=.85)
        print(f'{fam:6} {name:20} op {r["op"]:4d} h-evals {len(r["seen"]):4d} len {r["len"]:.2f}')
fig.tight_layout(); out = HERE / 'figs' / 'wallc_search.png'; fig.savefig(out, dpi=100); print(out)
