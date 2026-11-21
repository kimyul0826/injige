"""Logic-level LLM-A* variants with bounded suboptimality, on stored V2 responses (no inference).

Each LLM method has a no-LLM control that uses the identical algorithm with the waypoint chain
replaced by [start, goal], so any gain can be attributed to the LLM waypoints.

  focal  : A*_eps / focal search. OPEN ordered by f = g + h (admissible). Among nodes with
           f <= eps * f_min pick the one with the smallest LLM route estimate
           h_F = d(n, t_k) + remaining chain length.               bound: eps * C*
  mha    : Shared Multi-Heuristic A* (Aine et al.). Anchor key g + w1*h, inadmissible key
           g + w1*h_LLM (h_LLM = d(n, t_k) + chain), round robin with factor w2.  bound: w1*w2*C*
  egraph : Experience-graph heuristic (Phillips et al.). Collision-free LLM segments are
           "experience" edges with true cost; leaving them costs eps_E * distance.
           weighted A* with h_E.                                    bound: w * eps_E * C*
All variants: CLOSED duplicates skipped (dedupe accounting). Compared with dedupe A*/weighted A*.
"""
import sys
sys.dont_write_bytecode = True
import collections, gzip, heapq, json, math, time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o  # noqa: E402

D = math.dist
INF = math.inf
MOTIONS = o.MOTIONS


class Route:
    """Waypoint chain with progress-based target switching and remaining-route estimate."""
    def __init__(self, T, goal):
        self.T = list(T)
        self.goal = goal
        self.idx = 1
        self.chain = [0.0] * len(T)
        for k in range(len(T) - 2, -1, -1):
            self.chain[k] = self.chain[k + 1] + D(T[k], T[k + 1])

    def update(self, s):
        old = self.idx
        T = self.T
        while self.idx < len(T) - 1 and (D(s, T[self.idx]) <= 1.5 or D(s, self.goal) < D(T[self.idx], self.goal)):
            self.idx += 1
        return self.idx != old

    def h(self, p):
        return D(p, self.T[self.idx]) + self.chain[self.idx]


def neighbours(grid, s):
    for dx, dy in MOTIONS:
        n = (s[0] + dx, s[1] + dy)
        if not grid.coll(s, n):
            yield n, D(s, n)


def finish(parent, start, goal, ops, g):
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return dict(op=ops, sto=len(g), len=sum(D(a, b) for a, b in zip(path, path[1:])))


# --------------------------------------------------------------------------- focal search
def focal(env, grid, T, eps):
    start, goal = env.start, env.goal
    route = Route(T, goal)
    g, parent = {start: 0.0}, {start: start}
    openf = {start: D(start, goal)}          # node -> f (admissible)
    heap = [(openf[start], start)]
    closed = set()
    ops = 0
    while openf:
        while heap[0][1] not in openf or openf[heap[0][1]] != heap[0][0]:
            heapq.heappop(heap)
        fmin = heap[0][0]
        bound = eps * fmin + 1e-9
        s = min((n for n, f in openf.items() if f <= bound), key=lambda n: (route.h(n), openf[n], n))
        del openf[s]
        ops += 1
        closed.add(s)
        if s == goal:
            return finish(parent, start, goal, ops, g)
        route.update(s)
        for n, c in neighbours(grid, s):
            new = g[s] + c
            if new < g.get(n, INF):          # re-open closed nodes: required for the eps bound
                g[n], parent[n] = new, s
                closed.discard(n)
                openf[n] = new + D(n, goal)
                heapq.heappush(heap, (openf[n], n))
    raise RuntimeError('no path')


# --------------------------------------------------------------------------- shared MHA*
def mha(env, grid, T, w1, w2):
    start, goal = env.start, env.goal
    route = Route(T, goal)
    g, parent = {start: 0.0}, {start: start}
    h0 = lambda p: D(p, goal)
    k0 = lambda p: g[p] + w1 * h0(p)
    k1 = lambda p: g[p] + w1 * route.h(p)
    open0, open1 = [(k0(start), start)], [(k1(start), start)]
    in0, in1 = {start}, {start}
    closed_anchor, closed_inad = set(), set()
    ops = 0

    def top(heap, members, key):
        while heap:
            k, n = heap[0]
            if n not in members:
                heapq.heappop(heap); continue
            cur = key(n)
            if abs(cur - k) > 1e-9:
                heapq.heapreplace(heap, (cur, n)); continue
            return k, n
        return INF, None

    def expand(s):
        nonlocal open1
        in0.discard(s); in1.discard(s)
        if route.update(s):
            open1 = [(k1(n), n) for n in in1]
            heapq.heapify(open1)
        for n, c in neighbours(grid, s):
            new = g[s] + c
            if new < g.get(n, INF):
                g[n], parent[n] = new, s
                if n not in closed_anchor:
                    in0.add(n); heapq.heappush(open0, (k0(n), n))
                    if n not in closed_inad and k1(n) <= w2 * k0(n):
                        in1.add(n); heapq.heappush(open1, (k1(n), n))

    while True:
        m0, s0 = top(open0, in0, k0)
        if s0 is None:
            raise RuntimeError('no path')
        m1, s1 = top(open1, in1, k1)
        if s1 is not None and m1 <= w2 * m0:
            if g.get(goal, INF) <= m1:
                return finish(parent, start, goal, ops, g)
            ops += 1; expand(s1); closed_inad.add(s1)
        else:
            if g.get(goal, INF) <= m0:
                return finish(parent, start, goal, ops, g)
            ops += 1; expand(s0); closed_anchor.add(s0)


# --------------------------------------------------------------------------- E-graph heuristic
def egraph_h(env, T, eps_e):
    """E-graph heuristic. Collision-free LLM segments are densified (~1 cell spacing) so that
    states along an experience segment count as on-graph, as in the original E-graph method."""
    goal = env.goal
    nodes, edges = [], []
    def add(p):
        nodes.append(p)
        return len(nodes) - 1
    gi = add(goal)
    for a, b in zip(T, T[1:]):
        if a == b or env.collision(a, b):
            continue
        n = max(1, math.ceil(D(a, b)))
        ids = [add((a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)) for i in range(n + 1)]
        edges += [(ids[i], ids[i + 1], D(nodes[ids[i]], nodes[ids[i + 1]])) for i in range(n)]
    N = len(nodes)
    adj = [dict() for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            c = eps_e * D(nodes[i], nodes[j])
            adj[i][j] = adj[j][i] = c
    for i, j, c in edges:
        if c < adj[i][j]:
            adj[i][j] = adj[j][i] = c
    hv = [INF] * N
    hv[gi] = 0.0
    pq = [(0.0, gi)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > hv[u]:
            continue
        for v, c in adj[u].items():
            if d + c < hv[v]:
                hv[v] = d + c; heapq.heappush(pq, (d + c, v))
    items = list(zip(nodes, hv))

    def h(p):
        return min(eps_e * D(p, v) + hvv for v, hvv in items)
    return h


def egraph(env, grid, T, eps_e, w):
    start, goal = env.start, env.goal
    h = egraph_h(env, T, eps_e)
    hc = {}
    def H(p):
        if p not in hc:
            hc[p] = h(p)
        return hc[p]
    g, parent = {start: 0.0}, {start: start}
    q = [(w * H(start), start)]
    closed = set()
    ops = 0
    while q:
        _, s = heapq.heappop(q)
        if s in closed:
            continue
        ops += 1
        closed.add(s)
        if s == goal:
            return finish(parent, start, goal, ops, g)
        for n, c in neighbours(grid, s):
            if n in closed:
                continue
            new = g[s] + c
            if new < g.get(n, INF):
                g[n], parent[n] = new, s
                heapq.heappush(q, (new + w * H(n), n))
    raise RuntimeError('no path')


# --------------------------------------------------------------------------- experiment grid
CONFIGS = (
    [(f'focal_eps{e}', 'focal', dict(eps=e)) for e in (1.05, 1.1, 1.2, 1.5, 2.0)] +
    [(f'mha_w1{a}_w2{b}', 'mha', dict(w1=a, w2=b)) for a, b in
     ((1.0, 1.5), (1.0, 2.0), (1.25, 1.25), (1.5, 1.33), (1.5, 1.5), (2.0, 1.25))] +
    [(f'egraph_eE{e}_w{w}', 'egraph', dict(eps_e=e, w=w)) for e, w in
     ((1.5, 1.0), (2.0, 1.0), (3.0, 1.0), (1.5, 1.33), (2.0, 1.25))]
)
CONFIGS += [('llmastar_official', 'llmastar', {}),
            ('llmastar_progress', 'llmastar', dict(switch='progress')),
            ('llmastar_progress_wg2_wt0.5', 'llmastar', dict(switch='progress', wg=2.0, wt=0.5))]
RUN = dict(focal=focal, mha=mha, egraph=egraph,
           llmastar=lambda env, grid, T, **kw: o.search(env, grid, T, dedupe=True, **kw))


def work(args):
    sample, query, texts = args
    env = o.Environment(json.loads(json.dumps(query)), collision_backend='shapely')
    grid = o.Grid(env)
    out = []
    base = o.search(env, grid, None, dedupe=True)
    out.append(dict(sample=sample, prompt='-', method='astar', op=base['op'], sto=base['sto'], len=base['len']))
    for w in (1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0):
        r = o.search(env, grid, None, dedupe=True, wg=w)
        out.append(dict(sample=sample, prompt='-', method=f'wA*{w}', op=r['op'], sto=r['sto'], len=r['len']))
    noll = [env.start, env.goal]
    for name, kind, kw in CONFIGS:
        r = RUN[kind](env, grid, noll, **kw)
        out.append(dict(sample=sample, prompt='-', method=name + '|noLLM', op=r['op'], sto=r['sto'], len=r['len']))
    opt = base['path']
    turns = [opt[i] for i in range(1, len(opt) - 1)
             if (opt[i][0] - opt[i-1][0], opt[i][1] - opt[i-1][1]) != (opt[i+1][0] - opt[i][0], opt[i+1][1] - opt[i][1])]
    oracle = [env.start] + o.string_pull(env, [env.start] + turns + [env.goal])[1:-1][:3] + [env.goal]
    for prompt in o.PROMPTS + ('oracle',):
        if prompt == 'oracle':
            T = oracle
        else:
            try:
                nodes = o.parse_path(texts[prompt], mode='upstream')
            except ValueError:
                nodes = []
            T = env.targets(nodes, 'upstream')
        for name, kind, kw in CONFIGS:
            r = RUN[kind](env, grid, T, **kw)
            out.append(dict(sample=sample, prompt=prompt, method=name, op=r['op'], sto=r['sto'], len=r['len']))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    t0 = time.time()
    texts, _ = o.load_texts()
    tasks = [(s, q, {p: texts[(s, p)] for p in o.PROMPTS}) for s, q in o.load_queries(o.DATA)][:args.limit]
    rows = []
    with Pool(args.workers) as pool:
        for i, part in enumerate(pool.imap_unordered(work, tasks, chunksize=4), 1):
            rows += part
            if i % 100 == 0:
                print(f'{i}/{len(tasks)} {time.time() - t0:.0f}s', flush=True)
    out = HERE / 'results' / ('logic_per_sample.jsonl.gz' if args.limit is None else 'logic_debug.jsonl.gz')
    with gzip.open(out, 'wt') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    print('rows', len(rows), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
