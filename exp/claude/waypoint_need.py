"""How many waypoints does the optimal path need on bigger / more complex wall maps,
and which search-free map feature predicts it (to set a waypoint cap)?

oracle count = string-pulled turning points of the optimal A* path (same as waypoint_count.py).
features (no search): n_walls, n_cross (walls crossing start-goal segment),
                      n_comp (wall components), n_comp_cross (components crossing start-goal segment).
"""
import sys
sys.dont_write_bytecode = True
import collections, json
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o            # noqa: E402
import wall_component_h as wc          # noqa: E402

FAMS = ('original', 'scaled2', 'scaled4', 'dense', 'rooms', 'utrap', 'maze')


def work(args):
    fam, q = args
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
    grid = o.Grid(env)
    opt = o.search(env, grid, None, dedupe=True)['path']
    turns = [opt[i] for i in range(1, len(opt) - 1)
             if (opt[i][0] - opt[i-1][0], opt[i][1] - opt[i-1][1]) != (opt[i+1][0] - opt[i][0], opt[i+1][1] - opt[i][1])]
    need = len(o.string_pull(env, [env.start] + turns + [env.goal])) - 2
    s, g = env.start, env.goal
    crosses = lambda w: wc.intersects(s, g, w[0], w[1])
    comps = wc.components(env)
    return fam, dict(need=need, n_walls=len(env.walls), n_cross=sum(map(crosses, env.walls)),
                     n_comp=len(comps), n_comp_cross=sum(any(map(crosses, c)) for c in comps))


def main():
    tasks = [(f, q) for f in FAMS for q in json.loads((HERE / 'bench' / f'{f}.json').read_text())]
    rows = collections.defaultdict(list)
    with Pool(16) as pool:
        for fam, r in pool.imap_unordered(work, tasks, chunksize=2):
            rows[fam].append(r)
    res = {}
    rules = {'n_cross': lambda r: r['n_cross'], 'n_comp_cross': lambda r: r['n_comp_cross'],
             '2*n_comp_cross': lambda r: 2 * r['n_comp_cross'], 'n_cross+1': lambda r: r['n_cross'] + 1,
             'n_walls': lambda r: r['n_walls'], 'cap2': lambda r: 2}
    print(f"{'family':9} {'n':>4} {'need':>5} {'p90':>4} {'max':>4} {'walls':>6} {'cross':>6} {'cmpX':>5}  "
          + '  '.join(f'{k}:cov/slack'.rjust(20) for k in rules))
    for f in FAMS:
        R = rows[f]
        need = sorted(r['need'] for r in R)
        mean = lambda k: sum(r[k] for r in R) / len(R)
        cov = {}
        for k, fn in rules.items():
            ok = [r for r in R if r['need'] <= fn(r)]
            slack = sum(fn(r) - r['need'] for r in ok) / max(len(ok), 1)
            cov[k] = dict(coverage=100 * len(ok) / len(R), mean_slack=slack)
        hist = collections.Counter(min(r['need'], 10) for r in R)
        res[f] = dict(n=len(R), need_mean=mean('need'), need_p90=need[int(0.9 * len(need))], need_max=need[-1],
                      n_walls=mean('n_walls'), n_cross=mean('n_cross'), n_comp_cross=mean('n_comp_cross'),
                      hist=dict(sorted(hist.items())), rules=cov)
        print(f"{f:9} {len(R):4d} {mean('need'):5.2f} {res[f]['need_p90']:4d} {need[-1]:4d} {mean('n_walls'):6.1f} "
              f"{mean('n_cross'):6.2f} {mean('n_comp_cross'):5.2f}  "
              + '  '.join(f"{cov[k]['coverage']:6.1f}%/{cov[k]['mean_slack']:5.2f}".rjust(20) for k in rules))
    (HERE / 'results' / 'waypoint_need.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
