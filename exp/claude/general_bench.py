"""General-map benchmark: where does a non-LLM gate rule fail (= headroom for an LLM)? No inference.

Families (same JSON format as the course dataset: horizontal [y,x1,x2], vertical [x,y1,y2]):
  original : 200 problems from data/environment_50_30.json
  scaled2/4: original maps with all coordinates x2 / x4 (101x61, 201x121)
  dense    : 50x30, 18-26 random short walls
  utrap    : start inside a U whose closed side faces the goal, + random clutter
  rooms    : 3x2 rooms with narrow doors + furniture walls
  maze     : perfect maze on 5x5 cells (50x30)
Generated maps are saved to exp/claude/bench/<family>.json (seeded, reproducible).
"""
import sys
sys.dont_write_bytecode = True
import collections, json, math, random
from multiprocessing import Pool
from pathlib import Path

from shapely.geometry import LineString

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o      # noqa: E402
import logic_variants as lv      # noqa: E402
import astar_family as af        # noqa: E402
import gate_rule as gr           # noqa: E402

BENCH = HERE / 'bench'
N_PER = 200
D = math.dist


def env_ok(q):
    try:
        env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
    except ValueError:
        return None
    return env


def finalize(q, rng, min_dist):
    """Pick a reachable start/goal pair (free cells, far apart)."""
    W, H = q['range_x'][1], q['range_y'][1]
    for _ in range(400):
        s = (rng.randint(2, W - 3), rng.randint(2, H - 3))
        g = (rng.randint(2, W - 3), rng.randint(2, H - 3))
        if D(s, g) < min_dist:
            continue
        qq = dict(q, start=list(s), goal=list(g))
        env = env_ok(qq)
        if env is None:
            continue
        r = o.search(env, o.Grid(env), None, dedupe=True)
        if r['found']:
            return qq
    return None


def rand_walls(rng, W, H, n, lmin, lmax):
    hb, vb = [], []
    for _ in range(n):
        L = rng.randint(lmin, lmax)
        if rng.random() < .5:
            y = rng.randint(2, H - 3); x = rng.randint(1, W - 2 - L); hb.append([y, x, x + L])
        else:
            x = rng.randint(2, W - 3); y = rng.randint(1, H - 2 - L); vb.append([x, y, y + L])
    return hb, vb


def gen_dense(rng):
    hb, vb = rand_walls(rng, 51, 31, rng.randint(18, 26), 2, 10)
    return finalize(dict(range_x=[0, 51], range_y=[0, 31], horizontal_barriers=hb, vertical_barriers=vb), rng, 25)


def gen_utrap(rng):
    for _ in range(200):
        sx, sy = rng.randint(10, 40), rng.randint(8, 22)
        d = rng.choice([1, -1])                       # goal side (+x or -x)
        gx = rng.randint(43, 48) if d == 1 else rng.randint(2, 7)
        gy = rng.randint(3, 27)
        depth, half = rng.randint(5, 8), rng.randint(3, 6)
        wx = sx + d * 2                              # closed side between start and goal
        back = wx - d * depth
        hb = [[sy - half, min(back, wx), max(back, wx)], [sy + half, min(back, wx), max(back, wx)]]
        vb = [[wx, sy - half, sy + half]]
        h2, v2 = rand_walls(rng, 51, 31, 5, 3, 9)
        q = dict(range_x=[0, 51], range_y=[0, 31], horizontal_barriers=hb + h2, vertical_barriers=vb + v2,
                 start=[sx, sy], goal=[gx, gy])
        if not (1 < back < 49 and 1 < sy - half and sy + half < 29):
            continue
        env = env_ok(q)
        if env is None:
            continue
        if o.search(env, o.Grid(env), None, dedupe=True)['found']:
            return q
    return None


def gen_rooms(rng):
    hb, vb = [], []
    for x in (17, 34):                                # vertical room walls with one door per room row
        for y0, y1 in ((0, 15), (15, 30)):
            dpos = rng.randint(y0 + 2, y1 - 4)
            vb += [[x, y0, dpos], [x, dpos + 3, y1]]
    for x0, x1 in ((0, 17), (17, 34), (34, 50)):      # horizontal wall at y=15 with a door per room
        dpos = rng.randint(x0 + 2, x1 - 4)
        hb += [[15, x0, dpos], [15, dpos + 3, x1]]
    h2, v2 = rand_walls(rng, 51, 31, rng.randint(4, 8), 2, 6)
    q = dict(range_x=[0, 51], range_y=[0, 31], horizontal_barriers=hb + h2, vertical_barriers=vb + v2)
    return finalize(q, rng, 25)


def gen_maze(rng):
    C, cw, ch = 5, 10, 6
    seen, stack = {(0, 0)}, [(0, 0)]
    open_e = set()
    while stack:
        c = stack[-1]
        nb = [(c[0] + dx, c[1] + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
              if 0 <= c[0] + dx < cw and 0 <= c[1] + dy < ch and (c[0] + dx, c[1] + dy) not in seen]
        if not nb:
            stack.pop(); continue
        n = rng.choice(nb)
        open_e.add(frozenset((c, n))); seen.add(n); stack.append(n)
    hb, vb = [], []
    for i in range(cw):
        for j in range(ch):
            if i + 1 < cw and frozenset(((i, j), (i + 1, j))) not in open_e:
                vb.append([C * (i + 1), C * j, C * j + C])
            if j + 1 < ch and frozenset(((i, j), (i, j + 1))) not in open_e:
                hb.append([C * (j + 1), C * i, C * i + C])
    q = dict(range_x=[0, 51], range_y=[0, 31], horizontal_barriers=hb, vertical_barriers=vb)
    for _ in range(100):
        a = (rng.randrange(cw), rng.randrange(ch)); b = (rng.randrange(cw), rng.randrange(ch))
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) < 6:
            continue
        qq = dict(q, start=[C * a[0] + 2, C * a[1] + 2], goal=[C * b[0] + 2, C * b[1] + 2])
        env = env_ok(qq)
        if env and o.search(env, o.Grid(env), None, dedupe=True)['found']:
            return qq
    return None


def scaled(q, k):
    s = lambda v: v * k
    return dict(range_x=[0, 50 * k + 1], range_y=[0, 30 * k + 1],
                horizontal_barriers=[[s(a), s(b), s(c)] for a, b, c in q['horizontal_barriers']],
                vertical_barriers=[[s(a), s(b), s(c)] for a, b, c in q['vertical_barriers']],
                start=[s(v) for v in q['start']], goal=[s(v) for v in q['goal']])


def build_family(name):
    path = BENCH / f'{name}.json'
    if path.exists():
        return json.loads(path.read_text())
    rng = random.Random(f'bench-{name}')
    if name in ('original', 'scaled2', 'scaled4'):
        qs = [q for _, q in o.load_queries(o.DATA)]
        pick = random.Random('bench-original').sample(range(len(qs)), N_PER)
        base = [dict(range_x=qs[i]['range_x'], range_y=qs[i]['range_y'],
                     horizontal_barriers=qs[i]['horizontal_barriers'], vertical_barriers=qs[i]['vertical_barriers'],
                     start=qs[i]['start'], goal=qs[i]['goal']) for i in pick]
        out = base if name == 'original' else [scaled(q, int(name[-1])) for q in base]
    else:
        gen = dict(dense=gen_dense, utrap=gen_utrap, rooms=gen_rooms, maze=gen_maze)[name]
        out = []
        while len(out) < N_PER:
            q = gen(rng)
            if q:
                out.append(q)
    BENCH.mkdir(exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def work(args):
    fam, i, q = args
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
    grid = o.Grid(env)
    base = o.search(env, grid, None, dedupe=True)
    opt = LineString(base['path'])
    from shapely.geometry import Point
    rule = gr.build(env, env.start, env.goal, lambda a, b, c: min(c, key=lambda g: D(a, g) + D(g, b)))
    orc_gate = gr.build(env, env.start, env.goal, lambda a, b, c: min(c, key=lambda g: opt.distance(Point(g))))
    _, orc_path = af.oracle_T(env, grid)
    out = []
    for w in (1.5, 2.0, 3.0):
        r = o.search(env, grid, None, dedupe=True, wg=w)
        out.append((fam, f'wA*{w}', r['op'] / base['op'], r['len'] / base['len']))
    for k, T in (('noLLM', [env.start, env.goal]), ('rule', rule), ('oracle_gate', orc_gate), ('oracle_path', orc_path)):
        r = lv.egraph(env, grid, T, 1.5, 1.33)
        out.append((fam, f'egraph|{k}', r['op'] / base['op'], r['len'] / base['len']))
        r = o.search(env, grid, T, dedupe=True, switch='progress')
        out.append((fam, f'llmastar_progress|{k}', r['op'] / base['op'], r['len'] / base['len']))
    r = o.search(env, grid, orc_path, dedupe=True)
    out.append((fam, 'llmastar_official|oracle_path', r['op'] / base['op'], r['len'] / base['len']))
    out.append((fam, '_stats', len(env.walls), base['op'], len(o.string_pull(env, base['path'])) - 2,
                not env.collision(env.start, env.goal)))
    return out


def main():
    fams = sys.argv[1:] or ['original', 'scaled2', 'scaled4', 'dense', 'utrap', 'rooms', 'maze']
    tasks = []
    for f in fams:
        qs = build_family(f)
        tasks += [(f, i, q) for i, q in enumerate(qs)]
        print(f, len(qs), 'maps', flush=True)
    agg = collections.defaultdict(list)
    with Pool(16) as pool:
        for part in pool.imap_unordered(work, tasks, chunksize=2):
            for fam, k, *m in part:
                agg[(fam, k)].append(m)
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    res = []
    for f in fams:
        st = agg[(f, '_stats')]
        print(f'\n== {f}: walls {sum(x[0] for x in st)/len(st):.1f}, A* pops {sum(x[1] for x in st)/len(st):.0f}, '
              f'optimal waypoints {sum(x[2] for x in st)/len(st):.2f}, straight-line clear {sum(x[3] for x in st)}')
        for k in sorted(k for ff, k in agg if ff == f and k != '_stats'):
            v = agg[(f, k)]
            r = dict(family=f, method=k, op=gm([x[0] for x in v]), len=gm([x[1] for x in v]),
                     op_gt_2x=sum(x[0] > 2 for x in v), max_op=max(x[0] for x in v), max_len=max(x[1] for x in v))
            res.append(r)
            print(f"   {k:32s} op {r['op']:6.2f} len {r['len']:7.2f} op>2x {r['op_gt_2x']:3d} max_op {r['max_op']:5.2f} max_len {r['max_len']:.3f}")
    (HERE / 'results' / 'general_bench.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
