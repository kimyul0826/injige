"""How many waypoints? Offline truncation test (stored V2 Llama, no inference).

Chains truncated to first k intermediate waypoints (k = 0..4, all):
  llm_raw   : Llama targets (official filter)
  llm_topo  : Llama topology + code gate geometry (topology_extract)
  oracle    : string-pulled turning points of the optimal A* path (uncapped)
Evaluated with
  official : LLM-A* f = g + d(s,goal) + d(s,t)   (neighbor switch, dedupe accounting)
  route2   : f = g + 2 * route  (switch on reach, timeout 60), gated: invalid chain -> weighted A* w=2
Also reports the waypoint-count distribution.
"""
import sys
sys.dont_write_bytecode = True
import collections, json, math
from multiprocessing import Pool
from pathlib import Path

from shapely.geometry import LineString, Point

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o           # noqa: E402
import gate_rule as gr                # noqa: E402

D = math.dist
KS = (0, 1, 2, 3, 4, 'all')


def cut(T, k):
    mid = T[1:-1]
    return [T[0]] + (mid if k == 'all' else mid[:k]) + [T[-1]]


def work(args):
    sample, query, texts = args
    env = o.Environment(json.loads(json.dumps(query)), collision_backend='shapely')
    grid = o.Grid(env)
    base = o.search(env, grid, None, dedupe=True)
    opt = base['path']
    turns = [opt[i] for i in range(1, len(opt) - 1)
             if (opt[i][0] - opt[i-1][0], opt[i][1] - opt[i-1][1]) != (opt[i+1][0] - opt[i][0], opt[i+1][1] - opt[i][1])]
    chains = {'oracle': [env.start] + o.string_pull(env, [env.start] + turns + [env.goal])[1:-1] + [env.goal]}
    near = lambda line: (lambda a, b, c: min(c, key=lambda g: line.distance(Point(g))))
    for p in o.PROMPTS:
        try:
            nodes = o.parse_path(texts[p], mode='upstream')
        except ValueError:
            nodes = []
        T = env.targets(nodes, 'upstream')
        chains[f'llm_raw_{p}'] = T
        line = LineString(T) if len(set(T)) > 1 else LineString([env.start, env.goal])
        chains[f'llm_topo_{p}'] = gr.build(env, env.start, env.goal, near(line))
    wa2 = o.search(env, grid, None, dedupe=True, wg=2.0)
    out = [('count', k, len(T) - 2) for k, T in chains.items()]
    for name, T0 in chains.items():
        for k in KS:
            T = cut(T0, k)
            r = o.search(env, grid, T, dedupe=True)
            out.append((name, k, 'official', r['op'] / base['op'], r['len'] / base['len']))
            valid = all(not env.collision(a, b) for a, b in zip(T, T[1:]))
            r = o.search(env, grid, T, dedupe=True, via=2.0, switch='pop', timeout=60) if valid else wa2
            out.append((name, k, 'route2', r['op'] / base['op'], r['len'] / base['len']))
    out.append(('wA*2', '-', '-', wa2['op'] / base['op'], wa2['len'] / base['len']))
    return out


def main():
    texts, _ = o.load_texts()
    tasks = [(s, q, {p: texts[(s, p)] for p in o.PROMPTS}) for s, q in o.load_queries(o.DATA)]
    agg, counts = collections.defaultdict(list), collections.defaultdict(list)
    with Pool(16) as pool:
        for part in pool.imap_unordered(work, tasks, chunksize=4):
            for row in part:
                if row[0] == 'count':
                    counts[row[1]].append(row[2])
                else:
                    agg[row[:3]].append(row[3:])
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    res = {'counts': {}, 'search': []}
    for k, v in counts.items():
        h = collections.Counter(min(x, 5) for x in v)
        res['counts'][k] = dict(mean=sum(v) / len(v), hist={str(i): h[i] for i in range(6)})
        print(f"{k:18s} mean {sum(v)/len(v):.2f}  hist(0..5+) {[h[i] for i in range(6)]}")
    for (name, k, ev), v in sorted(agg.items(), key=lambda x: (x[0][0], x[0][2], str(x[0][1]))):
        r = dict(chain=name, k=k, eval=ev, op=gm([x[0] for x in v]), len=gm([x[1] for x in v]),
                 op_gt_2x=sum(x[0] > 2 for x in v))
        res['search'].append(r)
        print(f"{name:18s} {ev:8s} k={str(k):4s} op {r['op']:6.2f} len {r['len']:7.2f} op>2x {r['op_gt_2x']}")
    (HERE / 'results' / 'waypoint_count.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
