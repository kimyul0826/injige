"""Separate precision and strength: f = g + w * h, with h = route heuristic (precision from waypoints)
and w = strength. Compared with weighted A* (strength only). No inference.

h_route(s) = d(s, t_k) + chain(t_k -> goal), progress-based target switching.
'gated' = use h_route only if the waypoint chain is collision-free, else Euclid (weighted A*).
"""
import sys
sys.dont_write_bytecode = True
import collections, json, math
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o      # noqa: E402
import astar_family as af        # noqa: E402

WS = (1.0, 1.1, 1.25, 1.5, 2.0)


def work(args):
    sample, query, texts = args
    env = o.Environment(json.loads(json.dumps(query)), collision_backend='shapely')
    grid = o.Grid(env)
    base, orc = af.oracle_T(env, grid)
    chains = {'oracle': orc}
    for p in o.PROMPTS:
        try:
            nodes = o.parse_path(texts[p], mode='upstream')
        except ValueError:
            nodes = []
        chains[p] = env.targets(nodes, 'upstream')
    out = []
    for w in WS:
        r = o.search(env, grid, None, dedupe=True, wg=w)
        out.append((f'wA*', w, r['op'] / base['op'], r['len'] / base['len']))
        for k, T in chains.items():
            valid = all(not env.collision(a, b) for a, b in zip(T, T[1:]))
            for sw, to in (('progress', None), ('pop', None), ('pop', 60)):
                tag = f'{sw}{to or ""}'
                r = o.search(env, grid, T, dedupe=True, via=w, switch=sw, timeout=to)
                out.append((f'route[{tag}]|{k}', w, r['op'] / base['op'], r['len'] / base['len']))
                r = r if valid else o.search(env, grid, None, dedupe=True, wg=w)
                out.append((f'route_gated[{tag}]|{k}', w, r['op'] / base['op'], r['len'] / base['len']))
    return out


def main():
    texts, _ = o.load_texts()
    tasks = [(s, q, {p: texts[(s, p)] for p in o.PROMPTS}) for s, q in o.load_queries(o.DATA)]
    agg = collections.defaultdict(list)
    with Pool(16) as pool:
        for part in pool.imap_unordered(work, tasks, chunksize=4):
            for name, w, *m in part:
                agg[(name, w)].append(m)
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    res = []
    for (name, w), v in sorted(agg.items()):
        r = dict(method=name, w=w, op=gm([x[0] for x in v]), len=gm([x[1] for x in v]),
                 op_gt_2x=sum(x[0] > 2 for x in v), max_len=max(x[1] for x in v))
        res.append(r)
        print(f"{name:22s} w={w:<5} op {r['op']:6.2f} len {r['len']:7.2f} op>2x {r['op_gt_2x']:3d} max_len {r['max_len']:.3f}")
    (HERE / 'results' / 'route_heuristic.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
