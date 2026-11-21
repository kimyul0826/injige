"""Combine the three prompting styles (standard / CoT / RePE) on stored V2 Llama responses. No inference.

Per prompt: topology chain = Llama route used only as a topology hint, geometry by code (gate_rule.build).
Combinations:
  single_<p>   : that prompt's chain
  vote         : chain shared by >= 2 prompts, else no LLM chain
  shortest     : shortest collision-free chain among the 3 (code verification)
  agree_only   : like vote but requires all 3 identical
Search (dedupe accounting, ratios vs A*):
  route    : f = g + w * route(s)                  (switch on reach / timeout 60)
  max      : f = g + w * max(wall1(s), route(s))   (closed-form wall1 floor)
  no chain -> f = g + w * wall1(s)
"""
import sys
sys.dont_write_bytecode = True
import collections, heapq, json, math
from multiprocessing import Pool
from pathlib import Path

from shapely.geometry import LineString, Point

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o      # noqa: E402
import gate_rule as gr           # noqa: E402
import closed_form_h as cf       # noqa: E402

D = math.dist
INF = math.inf
WS = (1.5, 2.0)


def search(env, grid, T, w, mode, wall1):
    start, goal = env.start, env.goal
    if T is None:
        mode, T = 'wall1', [start, goal]
    chain = [0.0] * len(T)
    for k in range(len(T) - 2, -1, -1):
        chain[k] = chain[k + 1] + D(T[k], T[k + 1])
    st = {'i': 1}

    def H(p):
        i = st['i']
        r = D(p, T[i]) + chain[i]
        if mode == 'route':
            return r
        if mode == 'wall1':
            return wall1(p)
        return max(r, wall1(p))
    g, parent = {start: 0.0}, {start: start}
    q, closed, ops, since = [(w * H(start), start)], set(), 0, 0
    while q:
        _, s = heapq.heappop(q)
        if s in closed:
            continue
        ops += 1
        closed.add(s)
        if s == goal:
            break
        old = st['i']
        if s == T[st['i']] and st['i'] < len(T) - 1:
            st['i'] += 1
        since += 1
        if st['i'] < len(T) - 1 and since > 60:
            st['i'] += 1
        if st['i'] != old:
            since = 0
            q = [(g[e[1]] + w * H(e[1]), e[1]) for e in q]
            heapq.heapify(q)
        for dx, dy in o.MOTIONS:
            n = (s[0] + dx, s[1] + dy)
            if n in closed or grid.coll(s, n):
                continue
            ng = g[s] + D(s, n)
            if ng < g.get(n, INF):
                g[n], parent[n] = ng, s
                heapq.heappush(q, (ng + w * H(n), n))
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return ops, sum(D(a, b) for a, b in zip(path, path[1:]))


def work(args):
    sample, query, texts = args
    env = o.Environment(json.loads(json.dumps(query)), collision_backend='shapely')
    grid = o.Grid(env)
    wall1 = cf.make_h(env, 'wall1')
    bop, blen = search(env, grid, [env.start, env.goal], 1.0, 'route', wall1)  # plain A* (euclid)
    valid = lambda T: all(not env.collision(a, b) for a, b in zip(T, T[1:]))
    clen = lambda T: sum(D(a, b) for a, b in zip(T, T[1:]))
    topo = {}
    for p in o.PROMPTS:
        try:
            nodes = o.parse_path(texts[p], mode='upstream')
        except ValueError:
            nodes = []
        raw = env.targets(nodes, 'upstream')
        line = LineString(raw) if len(set(raw)) > 1 else LineString([env.start, env.goal])
        T = gr.build(env, env.start, env.goal, lambda a, b, c: min(c, key=lambda gg: line.distance(Point(gg))))
        topo[p] = T if valid(T) else None
    combos = {f'single_{p}': topo[p] for p in o.PROMPTS}
    ok = [tuple(T) for T in topo.values() if T]
    cnt = collections.Counter(ok)
    top = cnt.most_common(1)[0] if cnt else (None, 0)
    combos['vote'] = list(top[0]) if top[1] >= 2 else None
    combos['agree_only'] = list(top[0]) if top[1] == 3 else None
    combos['shortest'] = min((list(t) for t in ok), key=clen) if ok else None
    out = [('agree', 'n', top[1])]
    for w in WS:
        r = search(env, grid, None, w, 'wall1', wall1)
        out.append(('wall1', w, r[0] / bop, r[1] / blen))
        for name, T in combos.items():
            for mode in ('route', 'max'):
                if mode == 'route' and T is None:
                    r = search(env, grid, [env.start, env.goal], w, 'route', wall1)   # weighted A*
                else:
                    r = search(env, grid, T, w, mode, wall1)
                out.append((f'{name}|{mode}', w, r[0] / bop, r[1] / blen))
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
    ag = collections.Counter(x[0] for x in agg[('agree', 'n')])
    print('max agreement among 3 prompts (valid topology chains):', dict(sorted(ag.items())))
    res = {'agreement': dict(ag), 'search': []}
    for (name, w), v in sorted(agg.items()):
        if name == 'agree':
            continue
        r = dict(method=name, w=w, op=gm([x[0] for x in v]), len=gm([x[1] for x in v]), max_len=max(x[1] for x in v))
        res['search'].append(r)
        print(f"{name:22s} w={w:<4} op {r['op']:6.2f} len {r['len']:7.2f} max_len {r['max_len']:.3f}")
    (HERE / 'results' / 'prompt_combo.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
