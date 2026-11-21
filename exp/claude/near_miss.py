"""How near are the near misses? (no API calls) For responses whose path collides (classes 'corner cut',
'multi cross', 'on barrier' in failure_cases.json): can the path be made collision-free by moving ONE
intermediate waypoint to one of its 8 neighbours (radius 1) or within radius 2? Start and goal are fixed.
Run after failure_taxonomy.py."""
import sys
sys.dont_write_bytecode = True
import json
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o           # noqa: E402

rows = [r for r in json.loads((HERE / 'results' / 'failure_cases.json').read_text())
        if r['cls'] in ('corner cut', 'multi cross', 'on barrier') and r['set'].startswith('gpt')]
qs = dict(o.load_queries(o.DATA))


def fixable(r):
    env = o.Environment(json.loads(json.dumps(qs[r['sample']])), collision_backend='shapely')
    P = [tuple(p) for p in r['path']]
    ok = lambda Q: not any(env.collision(a, b) for a, b in zip(Q, Q[1:]))
    for rad in (1, 2):
        for i in range(1, len(P) - 1):
            for dx in range(-rad, rad + 1):
                for dy in range(-rad, rad + 1):
                    if (dx, dy) == (0, 0):
                        continue
                    Q = P[:i] + [(P[i][0] + dx, P[i][1] + dy)] + P[i + 1:]
                    if ok(Q):
                        return r['set'], r['prompt'], r['cls'], rad
    return r['set'], r['prompt'], r['cls'], None


if __name__ == '__main__':
    with Pool(16) as pool:
        res = pool.map(fixable, rows, chunksize=16)
    agg = defaultdict(Counter)
    for s, p, c, rad in res:
        agg[f'{s}|{p}'][(c, rad)] += 1
        agg[f'{s}|{p}'][('all', rad)] += 1
    lines = []
    for k, c in agg.items():
        n = sum(v for (cl, _), v in c.items() if cl == 'all')
        f1 = c[('all', 1)]; f2 = c[('all', 2)]
        cc = sum(v for (cl, _), v in c.items() if cl == 'corner cut')
        lines.append(f'{k:40s} colliding {n:4d}  fixed by moving one waypoint <=1 cell {100*f1/n:5.1f}%  <=2 cells {100*(f1+f2)/n:5.1f}%'
                     f'   (corner cut: {cc}, <=1 cell {100*c[("corner cut", 1)]/max(1,cc):.0f}%)')
    print('\n'.join(lines))
    (HERE / 'results' / 'near_miss.txt').write_text('\n'.join(lines) + '\n')
