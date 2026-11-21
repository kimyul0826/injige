"""Incremental gate route WITHOUT an LLM (rule baseline for the gate-selection idea). No inference.

route(a, b): if segment a-b is collision-free -> done. Otherwise find the first wall hit along a-b,
take its passable end gates (one cell beyond each endpoint, must be free), choose a gate by a rule,
and recurse on (a, gate) and (gate, b). Only walls that actually block get a waypoint.

Rules: 'short' = minimise d(a,g)+d(g,b); 'oracle' = gate lying closest to the optimal A* path.
The resulting chain is used by E-graph A* and by subgoal A* (with fallback).
"""
import sys
sys.dont_write_bytecode = True
import collections, json, math
from multiprocessing import Pool
from pathlib import Path

from shapely.geometry import LineString, Point

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o      # noqa: E402
import logic_variants as lv      # noqa: E402
import astar_family as af        # noqa: E402

D = math.dist


def gates(env, wall):
    (x1, y1), (x2, y2) = wall
    out = []
    if (x1, y1) == (x2, y2):
        cands = [(x1 + dx, y1 + dy) for dx, dy in o.MOTIONS]
    elif y1 == y2:   # horizontal
        lo, hi = sorted((x1, x2))
        cands = [(lo - 1, y1), (hi + 1, y1)]
    else:            # vertical
        lo, hi = sorted((y1, y2))
        cands = [(x1, lo - 1), (x1, hi + 1)]
    for c in cands:
        c = (int(round(c[0])), int(round(c[1])))
        if env.free(c):
            out.append(c)
    return out


def first_block(env, a, b):
    seg = LineString([a, b])
    best = None
    for w in env.walls:
        lw = LineString(w)
        if seg.intersects(lw):
            d = Point(a).distance(seg.intersection(lw))
            if best is None or d < best[0]:
                best = (d, w)
    return best[1] if best else None


def build(env, a, b, pick, depth=0):
    if not env.collision(a, b) or depth > 6:
        return [a, b]
    w = first_block(env, a, b)
    cand = [g for g in gates(env, w) if g not in (a, b)] if w else []
    if not cand:
        return [a, b]
    g = pick(a, b, cand)
    left = build(env, a, g, pick, depth + 1)
    right = build(env, g, b, pick, depth + 1)
    return left[:-1] + right


def work(args):
    sample, query = args
    env = o.Environment(json.loads(json.dumps(query)), collision_backend='shapely')
    grid = o.Grid(env)
    base = o.search(env, grid, None, dedupe=True)
    opt = LineString(base['path'])
    picks = {
        'short': lambda a, b, c: min(c, key=lambda g: D(a, g) + D(g, b)),
        'oracle_gate': lambda a, b, c: min(c, key=lambda g: opt.distance(Point(g))),
    }
    out = []
    chains = {k: build(env, env.start, env.goal, f) for k, f in picks.items()}
    chains['noLLM'] = [env.start, env.goal]
    _, chains['oracle_path'] = af.oracle_T(env, grid)
    for k, T in chains.items():
        valid = all(not env.collision(x, y) for x, y in zip(T, T[1:]))
        for name, fn in (('egraph1.5_1.33', lambda e, g_, t: lv.egraph(e, g_, t, 1.5, 1.33)),
                         ('egraph2_1.25', lambda e, g_, t: lv.egraph(e, g_, t, 2.0, 1.25)),
                         ('subgoal', af.subgoal),
                         ('llmastar_progress', lambda e, g_, t: o.search(e, g_, t, dedupe=True, switch='progress'))):
            r = fn(env, grid, T)
            out.append((name, k, r['op'] / base['op'], r['sto'] / base['sto'], r['len'] / base['len'], valid, len(T) - 2))
    return out


def main():
    tasks = list(o.load_queries(o.DATA))
    agg = collections.defaultdict(list)
    with Pool(16) as pool:
        for part in pool.imap_unordered(work, tasks, chunksize=8):
            for name, k, *m in part:
                agg[(name, k)].append(m)
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    res = []
    for (name, k), v in sorted(agg.items()):
        r = dict(method=name, chain=k, op=gm([x[0] for x in v]), sto=gm([x[1] for x in v]), len=gm([x[2] for x in v]),
                 chain_valid=sum(x[3] for x in v) / 10, mean_waypoints=sum(x[4] for x in v) / len(v),
                 op_gt_2x=sum(x[0] > 2 for x in v), max_len=max(x[2] for x in v))
        res.append(r)
        print(f"{name:18s} {k:12s} op {r['op']:6.2f} sto {r['sto']:6.2f} len {r['len']:7.2f} chain_valid {r['chain_valid']:5.1f}% "
              f"wps {r['mean_waypoints']:.2f} op>2x {r['op_gt_2x']:3d} max_len {r['max_len']:.3f}")
    (HERE / 'results' / 'gate_rule.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
