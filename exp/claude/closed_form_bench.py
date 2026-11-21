"""closed_form_h heuristics on the generated map families (bench/*.json, 200 maps each). No pre-search."""
import sys
sys.dont_write_bytecode = True
import collections, json, math
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o      # noqa: E402
import closed_form_h as cf       # noqa: E402

FAMS = ('original', 'scaled2', 'scaled4', 'dense', 'rooms', 'utrap', 'maze')
CFG = [('euclid', 2.0), ('euclid', 1.5), ('wall1', 1.0), ('wall1', 1.5), ('wall1', 2.0), ('wall_sum', 2.0)]


def work(args):
    fam, q = args
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
    grid = o.Grid(env)
    base = cf.astar(env, grid, cf.make_h(env, 'euclid'), 1.0)
    return fam, [(k, w, *(lambda r: (r['op'] / base['op'], r['len'] / base['len']))(cf.astar(env, grid, cf.make_h(env, k), w)))
                 for k, w in CFG]


def main():
    tasks = [(f, q) for f in FAMS for q in json.loads((HERE / 'bench' / f'{f}.json').read_text())]
    agg = collections.defaultdict(list)
    with Pool(16) as pool:
        for fam, rows in pool.imap_unordered(work, tasks, chunksize=2):
            for k, w, op, ln in rows:
                agg[(fam, k, w)].append((op, ln))
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    res = {}
    print('family    ' + '  '.join(f'{k}{w}'.rjust(15) for k, w in CFG))
    for f in FAMS:
        cells = []
        for k, w in CFG:
            v = agg[(f, k, w)]
            op, ln = gm([x[0] for x in v]), gm([x[1] for x in v])
            res[f'{f}|{k}|{w}'] = dict(op=op, len=ln)
            cells.append(f'{op:6.1f}/{ln:6.1f}'.rjust(15))
        print(f'{f:9s} ' + '  '.join(cells))
    (HERE / 'results' / 'closed_form_bench.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
