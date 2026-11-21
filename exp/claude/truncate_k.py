"""Code-side waypoint cap: keep only the first k waypoints that survive the official filter (k=1,2,3),
on stored responses. Compared with the full response. Upstream accounting, GM % of A*.
Usage: LLM_SRC=<output dir> TAG=<suffix> python truncate_k.py"""
import sys
sys.dont_write_bytecode = True
import collections, json, math, os
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o   # noqa: E402
if os.environ.get('LLM_SRC'):
    o.SRC = Path(os.environ['LLM_SRC'])
TAG = os.environ.get('TAG', '')


def work(args):
    s, q, texts = args
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely'); grid = o.Grid(env)
    base = o.search(env, grid, None)
    out = []
    for p in o.PROMPTS:
        try: nodes = o.parse_path(texts[p], mode='upstream')
        except ValueError: nodes = []
        T = env.targets(nodes, 'upstream')
        for k in (1, 2, 3, None):
            Tk = T if k is None else T[:1] + T[1:-1][:k] + T[-1:]
            r = o.search(env, grid, Tk)
            out.append((p, 'all' if k is None else f'first {k}', r['op'] / base['op'], r['sto'] / base['sto'], r['len'] / base['len']))
    return out


def main():
    texts, _ = o.load_texts()
    tasks = [(s, q, {p: texts[(s, p)] for p in o.PROMPTS}) for s, q in o.load_queries(o.DATA)]
    agg = collections.defaultdict(list)
    with Pool(10) as pool:
        for rows in pool.imap_unordered(work, tasks, chunksize=4):
            for p, k, *m in rows:
                agg[(p, k)].append(m)
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    res = {}
    for (p, k), v in sorted(agg.items()):
        res[f'{p}|{k}'] = dict(op=gm([x[0] for x in v]), sto=gm([x[1] for x in v]), len=gm([x[2] for x in v]),
                               gt2x=sum(x[0] > 2 for x in v))
        r = res[f'{p}|{k}']
        print(f"{TAG or 'llama'} {p:8} {k:8} op {r['op']:6.2f} sto {r['sto']:6.2f} len {r['len']:7.2f} >2x {r['gt2x']:3d}")
    (HERE / 'results' / f'truncate_k{TAG}.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
