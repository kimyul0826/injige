"""Reference for the barrier-endpoint prompt with the OFFICIAL LLM-A* search (no collision handling):
waypoints = best collision-free chain over the endpoint cells (a perfect picker; reference only),
fed unchanged into LLM-A* (official filter, target switching, upstream accounting)."""
import sys
sys.dont_write_bytecode = True
import json, math
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import offline_improve as o           # noqa: E402
import combine_llm_wallc as cb        # noqa: E402
import candidate_points as cp         # noqa: E402
from llm_astar.prompt_variants import barrier_endpoints  # noqa: E402


def work(q):
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
    grid = o.Grid(env)
    base = o.search(env, grid, None)
    ends = [tuple(e) for e in barrier_endpoints(q)]
    T = env.targets(cp.best_chain(env, ends), 'upstream')
    euclid = lambda s: math.dist(s, env.goal)
    r = cb.tsearch(env, grid, T, euclid)
    r0 = cb.tsearch(env, grid, [env.start, env.goal], euclid)
    w2 = o.search(env, grid, None, wg=2.0)
    return (r['op'] / base['op'], r['len'] / base['len'], r0['op'] / base['op'], w2['op'] / base['op'])


if __name__ == '__main__':
    with Pool(16) as p:
        rows = p.map(work, [q for _, q in o.load_queries(o.DATA)], chunksize=4)
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    res = dict(perfect_endpoint_llmastar_op=gm([r[0] for r in rows]), perfect_len=gm([r[1] for r in rows]),
               perfect_maxlen=100 * max(r[1] for r in rows), no_waypoint_llmastar_op=gm([r[2] for r in rows]),
               wastar2_op=gm([r[3] for r in rows]))
    print(json.dumps(res, indent=2))
    (HERE / 'results' / 'endpoint_llmastar_ceiling.json').write_text(json.dumps(res, indent=2))
