"""Turn an LLM chain of barrier endpoints into free waypoints, collision checked AFTER the LLM.

Each picked point p (normally a cell just past a barrier end, see barrier_endpoints):
  options = p and its 8 neighbours that are interior and on no barrier (p itself first)
Then a layered shortest-path DP from start to goal picks one option per layer so that every straight
segment is collision-free (strict). With skip=True the DP may also skip layers (order kept), i.e. the
shortest collision-free order-preserving sub-chain. No feasible chain -> [start, goal] (no waypoints).
"""
import math

D = math.dist


def _segments(q):
    s = [((min(a, b), y), (max(a, b), y)) for y, a, b in q.get('horizontal_barriers', [])]
    return s + [((x, min(a, b)), (x, max(a, b))) for x, a, b in q.get('vertical_barriers', [])]


def options(q, p):
    (x0, x1), (y0, y1) = q['range_x'], q['range_y']
    segs = _segments(q)
    on = lambda c: any(a[0] <= c[0] <= b[0] and a[1] <= c[1] <= b[1] for a, b in segs)
    free = lambda c: x0 + 1 <= c[0] <= x1 - 2 and y0 + 1 <= c[1] <= y1 - 2 and not on(c)
    near = [(p[0] + dx, p[1] + dy) for dx in (0, -1, 1) for dy in (0, -1, 1)]
    return [c for c in near if free(c)]


def snap_chain(env, q, nodes, skip=False):
    start, goal = tuple(q['start']), tuple(q['goal'])
    mid = [tuple(n) for n in nodes if tuple(n) not in (start, goal)]
    layers = [[start]] + [o for o in (options(q, p) for p in mid) if o] + [[goal]]
    best = {(0, 0): (0.0, None)}
    for i in range(1, len(layers)):
        for j, c in enumerate(layers[i]):
            cand = []
            for pi in (range(i) if skip else [i - 1]):
                for pj, pc in enumerate(layers[pi]):
                    if (pi, pj) in best and not env.collision(pc, c):
                        cand.append((best[(pi, pj)][0] + D(pc, c), (pi, pj)))
            if cand:
                best[(i, j)] = min(cand)
    end = (len(layers) - 1, 0)
    if end not in best:
        return [start, goal]
    out, k = [], end
    while k is not None:
        out.append(layers[k[0]][k[1]])
        k = best[k][1]
    return out[::-1]
