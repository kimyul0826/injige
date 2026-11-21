"""Are the five released demonstrations representative of the 1000 evaluation problems?
Per problem: barrier count, border-attached barriers, minimal waypoint count k (0 = straight line clear,
1 = some single free cell sees start and goal, 2+ otherwise; exact over all interior cells), turning points of
the optimal path (string-pulled), optimal length / straight distance."""
import sys
sys.dont_write_bytecode = True
import json, math, statistics
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import offline_improve as o           # noqa: E402
import combine_llm_wallc as cb        # noqa: E402

DEMOS = [([5, 5], [20, 20], [[10, 0, 25], [15, 30, 50]], [[25, 10, 22]]),
         ([2, 3], [18, 8], [[7, 0, 10], [5, 30, 40]], [[14, 12, 30], [40, 0, 11]]),
         ([40, 20], [10, 10], [[15, 20, 30], [15, 35, 50]], [[20, 4, 30]]),
         ([4, 4], [16, 16], [[20, 0, 20], [5, 30, 40]], [[40, 0, 15], [20, 10, 21]]),
         ([25, 25], [4, 4], [[7, 0, 20], [15, 10, 15]], [[10, 15, 30], [30, 0, 18], [10, 0, 5]])]


def stats(q):
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
    grid = o.Grid(env)
    S, G = env.start, env.goal
    x0, x1, y0, y1 = env.rx[0], env.rx[1] - 1, env.ry[0], env.ry[1] - 1
    border = lambda p: p[0] <= x0 or p[0] >= x1 or p[1] <= y0 or p[1] >= y1
    nb = len(env.walls)
    attached = sum(border(a) or border(b) for a, b in env.walls)
    if not env.collision(S, G):
        k = 0
    elif any(not env.collision(S, (x, y)) and not env.collision((x, y), G)
             for x in range(x0 + 1, x1) for y in range(y0 + 1, y1)):
        k = 1
    else:
        k = 2
    opt = o.search(env, grid, None, dedupe=True)
    turns = len(cb.oracle_chain(env, grid)) - 2
    return nb, attached, k, turns, opt['len'] / math.dist(S, G)


if __name__ == '__main__':
    qs = [q for _, q in o.load_queries(o.DATA)]
    with Pool(16) as p:
        rows = p.map(stats, qs, chunksize=8)
    demo_rows = [stats(dict(range_x=[0, 51], range_y=[0, 31], start=s, goal=g, horizontal_barriers=h,
                            vertical_barriers=v)) for s, g, h, v in DEMOS]
    print('map sizes in dataset:', Counter((tuple(q['range_x']), tuple(q['range_y'])) for q in qs))
    for name, R in (('dataset (1000)', rows), ('demos (5)', demo_rows)):
        print(f'\n== {name}')
        print('  barriers per map    ', dict(sorted(Counter(r[0] for r in R).items())))
        print('  border-attached     ', dict(sorted(Counter(r[1] for r in R).items())))
        print('  min waypoints k     ', dict(sorted(Counter(r[2] for r in R).items())))
        print('  optimal turn points ', dict(sorted(Counter(r[3] for r in R).items())))
        print('  opt len / straight   mean %.3f median %.3f' % (statistics.mean(r[4] for r in R),
                                                            statistics.median(r[4] for r in R)))
