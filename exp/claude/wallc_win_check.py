"""Where does wallC beat wA* in wall time?
mode 'table': precomputed collision table (collision cost excluded, shared).
mode 'online': collision checked per neighbour on demand (as in the released LLM-A* code), cached per edge.
Also per-query win rate and equal-path-quality comparison (wA*1.5 vs wallC1.5). First 60 maps per family."""
import sys
sys.dont_write_bytecode = True
import json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o, wall_component_h as wc, llmastar_vs_wallc as lv   # noqa: E402


class Lazy:
    def __init__(self, env):
        self.env, self.blocked = env, {}
    def coll(self, a, b):
        r = self.blocked.get((a, b))
        if r is None:
            r = self.blocked[(a, b)] = self.env.collision(a, b)
        return r


def run(env, grid, k):
    if k == 'wA*2':
        return o.search(env, grid, None, wg=2.0)
    if k == 'wA*1.5':
        return o.search(env, grid, None, wg=1.5)
    return lv.astar_h(env, grid, wc.make_wallC(env), 1.5, True)   # includes precompute


KS = ('wA*2', 'wA*1.5', 'wallC1.5')
res = {}
for fam in ('original', 'scaled2', 'scaled4', 'dense', 'rooms', 'utrap', 'maze'):
    qs = json.loads((HERE / 'bench' / f'{fam}.json').read_text())[:60]
    out = {}
    for mode in ('table', 'online'):
        T = {k: [] for k in KS}; L = {k: [] for k in KS}; OP = {k: [] for k in KS}
        for q in qs:
            env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
            base = o.Grid(env) if mode == 'table' else None
            for k in KS:
                grid = base if mode == 'table' else Lazy(env)
                t = time.perf_counter(); r = run(env, grid, k); T[k].append(time.perf_counter() - t)
                L[k].append(r['len']); OP[k].append(r['op'])
        win = lambda k: 100 * sum(c < w for c, w in zip(T['wallC1.5'], T[k])) / len(qs)
        out[mode] = {k: dict(ms=1000 * sum(T[k]) / len(qs), ops=sum(OP[k]) / len(qs), len=sum(L[k]) / len(qs)) for k in KS}
        out[mode]['wallC_faster_pct'] = {'vs wA*2': win('wA*2'), 'vs wA*1.5': win('wA*1.5')}
        print(f"{fam:9} {mode:6} " + ' | '.join(f"{k} {out[mode][k]['ms']:7.2f}ms {out[mode][k]['ops']:6.0f}op len {out[mode][k]['len']:6.1f}" for k in KS)
              + f" | wallC faster: {win('wA*2'):4.0f}% vs wA*2, {win('wA*1.5'):4.0f}% vs wA*1.5", flush=True)
    res[fam] = out
(HERE / 'results' / 'wallc_win_check.json').write_text(json.dumps(res, indent=2))
