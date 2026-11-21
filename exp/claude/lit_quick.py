"""Quick offline checks of literature ideas on stored V2 responses (no inference).

- MMP-A* adaptive decay: h = h_goal + alpha^k * d(s, t_k)
- iLLM-A* waypoint selection: keep only the two waypoints closest to the start
"""
import sys
sys.dont_write_bytecode = True
import collections, json, math
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offline_improve as o

VARS = [
    ('official', 'official', {}),
    ('mmp_decay0.7', 'official', dict(decay=0.7)),
    ('mmp_decay0.5', 'official', dict(decay=0.5)),
    ('mmp_decay0.7+progress', 'official', dict(decay=0.7, switch='progress')),
    ('illm_first2', 'first2', {}),
    ('illm_first2+progress', 'first2', dict(switch='progress')),
    ('illm_first1', 'first1', {}),
]


def proc(env, nodes, kind):
    T = env.targets(nodes, 'upstream')
    if kind.startswith('first'):
        k = int(kind[-1])
        mid = T[1:-1]
        if len(mid) > 2:
            mid = mid[:k]  # waypoints in LLM order; first ones are the nearest to start along the plan
        else:
            mid = mid[:k] if k < len(mid) else mid
        T = [T[0]] + mid + [T[-1]]
    return T


def work(args):
    sample, query, texts = args
    env = o.Environment(json.loads(json.dumps(query)), collision_backend='shapely')
    grid = o.Grid(env)
    a = o.search(env, grid, None)
    out = []
    for prompt in o.PROMPTS:
        try:
            nodes = o.parse_path(texts[prompt], mode='upstream')
        except ValueError:
            nodes = []
        for name, kind, kw in VARS:
            kw = dict(kw)
            T = proc(env, nodes, kind)
            d = kw.pop('decay', None)
            if d is not None:
                kw['wt'] = [d ** k for k in range(len(T))]
            r = o.search(env, grid, T, **kw)
            out.append((name, prompt, r['op'] / a['op'], r['sto'] / a['sto'], r['len'] / a['len']))
    return out


def main():
    texts, _ = o.load_texts()
    tasks = [(s, q, {p: texts[(s, p)] for p in o.PROMPTS}) for s, q in o.load_queries(o.DATA)]
    agg = collections.defaultdict(list)
    with Pool(16) as pool:
        for part in pool.imap_unordered(work, tasks, chunksize=10):
            for name, p, *m in part:
                agg[(name, p)].append(m)
    res = []
    for (name, p), ms in agg.items():
        g = [100 * math.exp(sum(math.log(x[i]) for x in ms) / len(ms)) for i in range(3)]
        res.append(dict(variant=name, prompt=p, op=g[0], sto=g[1], len=g[2],
                        op_gt_2x=sum(x[0] > 2 for x in ms), max_op=max(x[0] for x in ms)))
    res.sort(key=lambda r: ([v[0] for v in VARS].index(r['variant']), o.PROMPTS.index(r['prompt'])))
    out = Path(__file__).resolve().parent / 'results' / 'lit_quick.json'
    out.write_text(json.dumps(res, indent=2))
    for r in res:
        print(f"{r['variant']:24s} {r['prompt']:8s} op {r['op']:6.2f} sto {r['sto']:6.2f} len {r['len']:7.2f} "
              f"op>2x {r['op_gt_2x']:4d} max {r['max_op']:.2f}")


if __name__ == '__main__':
    main()
