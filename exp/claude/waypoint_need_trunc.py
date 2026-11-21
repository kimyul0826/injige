"""Official LLM-A* formula with oracle (string-pulled optimal) waypoints truncated to first k, per map family.
Shows the cost of a cap below the true need. dedupe accounting, ratio to A*."""
import sys
sys.dont_write_bytecode = True
import collections, json, math
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o   # noqa: E402
FAMS = ('original', 'scaled4', 'dense', 'rooms', 'utrap', 'maze')
KS = (0, 1, 2, 3, 'all')

def work(args):
    fam, q = args
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
    grid = o.Grid(env)
    base = o.search(env, grid, None, dedupe=True)
    opt = base['path']
    turns = [opt[i] for i in range(1, len(opt) - 1)
             if (opt[i][0] - opt[i-1][0], opt[i][1] - opt[i-1][1]) != (opt[i+1][0] - opt[i][0], opt[i+1][1] - opt[i][1])]
    T0 = o.string_pull(env, [env.start] + turns + [env.goal])
    out = []
    for k in KS:
        mid = T0[1:-1] if k == 'all' else T0[1:-1][:k]
        r = o.search(env, grid, [T0[0]] + mid + [T0[-1]], dedupe=True)
        out.append((k, r['op'] / base['op'], r['len'] / base['len']))
    return fam, out

def main():
    tasks = [(f, q) for f in FAMS for q in json.loads((HERE / 'bench' / f'{f}.json').read_text())]
    agg = collections.defaultdict(list)
    with Pool(16) as pool:
        for fam, rows in pool.imap_unordered(work, tasks, chunksize=2):
            for k, op, ln in rows:
                agg[(fam, k)].append((op, ln))
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    res = {}
    print('family   ' + ''.join(f'k={k}'.rjust(16) for k in KS))
    for f in FAMS:
        cells = []
        for k in KS:
            v = agg[(f, k)]
            res[f'{f}|{k}'] = dict(op=gm([a for a, _ in v]), len=gm([b for _, b in v]))
            cells.append(f"{res[f'{f}|{k}']['op']:6.1f}/{res[f'{f}|{k}']['len']:6.1f}".rjust(16))
        print(f'{f:8} ' + ''.join(cells))
    (HERE / 'results' / 'waypoint_need_trunc.json').write_text(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
