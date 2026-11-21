"""Render the selected RePE2 demonstrations (exp/claude/results/repe2_demos_selected.json) as prompt text.
Round j (< k) = start + first j waypoints of the accepted chain + goal; it must fail (k is minimal).
Check sentences are generated from the exact segment/barrier intersection test."""
import json, sys
from pathlib import Path
from shapely.geometry import LineString

HERE = Path(__file__).resolve().parent


def crossings(q, a, b):
    seg, out = LineString([a, b]), []
    for y, x1, x2 in q['horizontal_barriers']:
        if seg.intersects(LineString([(x1, y), (x2, y)])):
            out.append(f'the horizontal barrier at y={y} spanning x={x1} to x={x2}')
    for x, y1, y2 in q['vertical_barriers']:
        if seg.intersects(LineString([(x, y1), (x, y2)])):
            out.append(f'the vertical barrier at x={x} spanning y={y1} to y={y2}')
    return out


def render(q, chain):
    S, G, mid = chain[0], chain[-1], chain[1:-1]
    lines = [f"Start Point: {q['start']}", f"Goal Point: {q['goal']}",
             f"Horizontal Barriers: {q['horizontal_barriers']}", f"Vertical Barriers: {q['vertical_barriers']}"]
    for j in range(len(mid) + 1):
        path = [S] + mid[:j] + [G]
        lines.append(f'Round {j}: {path}')
        bad = [(a, b, crossings(q, a, b)) for a, b in zip(path, path[1:])]
        bad = [x for x in bad if x[2]]
        if j < len(mid):
            assert bad, (q, path)
            a, b, c = bad[0]
            lines.append(f"Check: The segment from {a} to {b} crosses {' and '.join(c)}. Not valid.")
        else:
            assert not bad, (q, path)
            lines.append('Check: Every segment avoids all barriers. Valid.')
    lines.append(f'Generated Path: {path}')
    return '\n'.join(lines)


if __name__ == '__main__':
    sel = json.loads((HERE / 'results' / 'repe2_demos_selected.json').read_text())
    print('\n\n'.join(render(r['query'], r['chain']) for r in sel))
