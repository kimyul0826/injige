"""Wall time per search (single process), wallC precompute+search vs A*, wA*2, per map family (first 60 maps)."""
import sys
sys.dont_write_bytecode = True
import json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o, wall_component_h as wc, llmastar_vs_wallc as lv   # noqa: E402
res = {}
for fam in ('original', 'scaled2', 'scaled4', 'dense', 'rooms', 'utrap', 'maze'):
    T = {'A*': 0, 'wA*2': 0, 'wallC1.5': 0}; ops = {k: 0 for k in T}
    qs = json.loads((HERE / 'bench' / f'{fam}.json').read_text())[:60]
    for q in qs:
        env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely'); grid = o.Grid(env)
        t = time.perf_counter(); r = o.search(env, grid, None); T['A*'] += time.perf_counter() - t; ops['A*'] += r['op']
        t = time.perf_counter(); r = o.search(env, grid, None, wg=2.0); T['wA*2'] += time.perf_counter() - t; ops['wA*2'] += r['op']
        t = time.perf_counter(); r = lv.astar_h(env, grid, wc.make_wallC(env), 1.5, True); T['wallC1.5'] += time.perf_counter() - t; ops['wallC1.5'] += r['op']
    res[fam] = {k: dict(ms=1000 * T[k] / len(qs), ops=ops[k] / len(qs)) for k in T}
    print(f"{fam:9} size {q['range_x'][1]}x{q['range_y'][1]} walls {len(env.walls):3d} | " +
          ' | '.join(f"{k} {res[fam][k]['ms']:7.2f} ms {res[fam][k]['ops']:7.0f} ops" for k in T))
(HERE / 'results' / 'cost_check_fam.json').write_text(json.dumps(res, indent=2))
