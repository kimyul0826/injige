"""Illustrate wallC: components, convex hulls, visible hull vertices, best detour for one node, h excess map.
Output: figs/wallc_explain.png"""
import sys
sys.dont_write_bytecode = True
import json, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o, wall_component_h as wc   # noqa: E402
D = math.dist
PICKS = [('utrap', 4), ('rooms', 2)]


def best_detour(env, s):
    """Re-run hull_bound logic for node s and return (component index, path) of the binding detour."""
    goal = env.goal
    x0, x1, y0, y1 = env.rx[0], env.rx[1] - 1, env.ry[0], env.ry[1] - 1
    onb = lambda p: p[0] <= x0 or p[0] >= x1 or p[1] <= y0 or p[1] >= y1
    best = (D(s, goal), None, [s, goal])
    for ci, segs in enumerate(wc.components(env)):
        if not any(wc.intersects(s, goal, a, b) for a, b in segs):
            continue
        hull = MultiLineString([list(t) for t in segs]).convex_hull
        cand = []
        if isinstance(hull, Polygon) and not hull.contains(Point(s)) and not hull.contains(Point(goal)):
            V = list(hull.exterior.coords)[:-1]
            if not hull.exterior.is_ccw: V.reverse()
            V = [tuple(v) for v in V]; n = len(V); B = [onb(v) for v in V]; core = hull.buffer(-1e-6)
            vis = lambda p: [i for i in range(n) if not LineString([p, V[i]]).intersects(core)]
            for a in vis(s):
                for b in vis(goal):
                    for st in (1, -1):
                        i, walk, ok, pts = a, 0.0, not B[a], [V[a]]
                        while ok and i != b:
                            j = (i + st) % n; walk += D(V[i], V[j]); i = j; ok = not B[i]; pts.append(V[i])
                        if ok:
                            cand.append((D(s, V[a]) + walk + D(V[b], goal), [s] + pts + [goal]))
        # wall1 part: max over crossed segments of (min over passable ends); final = max(hull min, wall1)
        w1 = (D(s, goal), [s, goal])
        for a, b in segs:
            ends = [e for e in (a, b) if not onb(e)]
            if ends and wc.intersects(s, goal, a, b):
                w1 = max(w1, min((D(s, e) + D(e, goal), [s, e, goal]) for e in ends))
        cv = max(min(cand), w1) if cand else w1
        if cv[0] > best[0]: best = (cv[0], ci, cv[1])
    return best


fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
for row, (fam, idx) in enumerate(PICKS):
    q = json.loads((HERE / 'bench' / f'{fam}.json').read_text())[idx]
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
    comps = wc.components(env); h = wc.make_wallC(env)
    cols = plt.cm.tab10(np.arange(len(comps)) % 10)
    s = env.start
    v, ci, path = best_detour(env, s)
    for k, ax in enumerate(axes[row]):
        for c, segs in zip(cols, comps):
            for a, b in segs:
                ax.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=2.5)
            if k == 0:
                hull = MultiLineString([list(t) for t in segs]).convex_hull
                if isinstance(hull, Polygon):
                    ax.plot(*hull.exterior.xy, color=c, lw=1, ls=':')
        if k == 0:
            ax.plot([s[0], env.goal[0]], [s[1], env.goal[1]], color='grey', lw=1, ls='--', label=f'straight line d={D(s, env.goal):.1f}')
            ax.plot(*zip(*path), color='#d0542c', lw=2, label=f'binding detour h={v:.1f}')
            ax.set_title(f'{fam} #{idx}: components (colors), hulls (dotted), h(start)')
        else:
            X, Y = np.meshgrid(np.arange(env.rx[0] + 1, env.rx[1] - 1), np.arange(env.ry[0] + 1, env.ry[1] - 1))
            Z = np.array([[h((int(x), int(y))) - D((int(x), int(y)), env.goal) for x in X[0]] for y in Y[:, 0]])
            im = ax.pcolormesh(X, Y, Z, cmap='magma_r', shading='nearest', alpha=.85)
            fig.colorbar(im, ax=ax, fraction=.03, label='h(s) - straight distance')
            ax.set_title('extra distance wallC adds over euclid (0 = line of sight)')
        ax.scatter(*s, s=90, color='#2a9d4b', edgecolor='k', zorder=5, label='start')
        ax.scatter(*env.goal, s=140, marker='*', color='#e9b10c', edgecolor='k', zorder=5, label='goal')
        ax.set_xlim(env.rx[0] - 1, env.rx[1]); ax.set_ylim(env.ry[0] - 1, env.ry[1]); ax.set_aspect('equal')
        if k == 0: ax.legend(fontsize=8, loc='best')
fig.tight_layout(); out = HERE / 'figs' / 'wallc_explain.png'; fig.savefig(out, dpi=105); print(out)
