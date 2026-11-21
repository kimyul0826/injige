"""Cost-fair accounting on the 1000 original queries (single process).
wallC: precompute (components + hulls) time, per-node heuristic evaluations, segment-intersection tests,
total wall time = precompute + search (heuristic included). Compared with A*, wA*2 (same timing), and
LLM-A* standard (search time + stored LLM generation time)."""
import sys
sys.dont_write_bytecode = True
import json, math, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o            # noqa: E402
import wall_component_h as wc          # noqa: E402
import llmastar_vs_wallc as lv         # noqa: E402

CNT = {'seg': 0}
_orig = wc.intersects
def counted(*a):
    CNT['seg'] += 1
    return _orig(*a)
wc.intersects = counted

def main():
    qs = list(o.load_queries(o.DATA))
    rows = []
    for s, q in qs:
        env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
        t = time.perf_counter(); grid = o.Grid(env); t_grid = time.perf_counter() - t
        t = time.perf_counter(); base = o.search(env, grid, None); t_astar = time.perf_counter() - t
        t = time.perf_counter(); wa = o.search(env, grid, None, wg=2.0); t_wa = time.perf_counter() - t
        CNT['seg'] = 0
        t = time.perf_counter(); h = wc.make_wallC(env); t_pre = time.perf_counter() - t
        seg_pre = CNT['seg']
        calls = {'n': 0}
        def hc(x, h=h):
            calls['n'] += 1
            return h(x)
        t = time.perf_counter(); r = lv.astar_h(env, grid, hc, 1.5, True); t_wc = time.perf_counter() - t
        rows.append(dict(walls=len(env.walls), t_grid=t_grid, t_astar=t_astar, t_wa=t_wa, t_pre=t_pre, t_wc=t_wc,
                         seg_pre=seg_pre, seg_search=CNT['seg'] - seg_pre, hcalls=calls['n'],
                         op_astar=base['op'], op_wa=wa['op'], op_wc=r['op']))
    n = len(rows)
    m = lambda k: sum(r[k] for r in rows) / n
    res = {k: m(k) for k in rows[0]}
    print(f"queries {n}, walls/map {res['walls']:.1f}")
    print(f"grid collision table (shared by every method): {1000*res['t_grid']:.2f} ms")
    print(f"A*      : {1000*res['t_astar']:.2f} ms, ops {res['op_astar']:.0f}")
    print(f"wA* w=2 : {1000*res['t_wa']:.2f} ms, ops {res['op_wa']:.0f}")
    print(f"wallC   : precompute {1000*res['t_pre']:.3f} ms + search {1000*res['t_wc']:.2f} ms = {1000*(res['t_pre']+res['t_wc']):.2f} ms, "
          f"ops {res['op_wc']:.0f}, h calls {res['hcalls']:.0f}, segment tests {res['seg_search']:.0f} (precompute {res['seg_pre']:.0f})")
    print(f"segment tests per h call: {res['seg_search']/res['hcalls']:.1f}")
    (HERE / 'results' / 'cost_check.json').write_text(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
