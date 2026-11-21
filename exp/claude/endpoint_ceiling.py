"""Offline checks for the barrier-endpoint plan (no API):
  oracle-endpoints : best collision-free chain over the endpoint cells, then snap_chain
                     (what a perfect endpoint-picking LLM would give)
  stored : stored GPT-4o mini original-prompt responses through snap_chain (strict / skip)
Each chain is fed into wallC+visroute w=1.5 (full gating already implied by snap_chain)."""
import sys
sys.dont_write_bytecode = True
import collections, json, math, os
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import offline_improve as o           # noqa: E402
o.SRC = Path(os.environ.get('LLM_SRC', 'outputs/gpt4omini-table1-upstream-batch-v1'))
import combine_llm_wallc as cb        # noqa: E402
import wall_component_h as wc         # noqa: E402
import llmastar_vs_wallc as lv        # noqa: E402
import candidate_points as cp         # noqa: E402
import endpoint_snap as es            # noqa: E402
from llm_astar.prompt_variants import barrier_endpoints  # noqa: E402

TEXTS = None


def work(args):
    s, q = args
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely')
    grid = o.Grid(env)
    base = o.search(env, grid, None)
    hw = wc.make_wallC(env)
    ends = [tuple(e) for e in barrier_endpoints(q)]
    chains = {}
    # perfect LLM: best collision-free chain over the endpoint cells, then the same snap step
    chains['oracle-endpoints'] = es.snap_chain(env, q, cp.best_chain(env, ends)[1:-1])
    chains['oracle-candidates'] = cp.best_chain(env, cp.candidates(env))
    for p in o.PROMPTS:
        try:
            nodes = o.parse_path(TEXTS[(s, p)], mode='upstream')
        except (ValueError, KeyError):
            nodes = []
        chains[f'{p}-strict'] = es.snap_chain(env, q, nodes)
        chains[f'{p}-skip'] = es.snap_chain(env, q, nodes, skip=True)
    out = [('wallC w=1.5', lv.astar_h(env, grid, hw, 1.5, True), 0)]
    for k, T in chains.items():
        out.append((k, lv.astar_h(env, grid, cb.combined(hw, cb.make_route(env, T)), 1.5, True), len(T) - 2))
    return [(k, r['op'] / base['op'], r['len'] / base['len'], n) for k, r, n in out]


if __name__ == '__main__':
    TEXTS, _ = o.load_texts()
    qs = list(o.load_queries(o.DATA))
    agg = collections.defaultdict(list)
    with Pool(16) as pool:
        for rows in pool.imap_unordered(work, qs, chunksize=4):
            for k, op, ln, n in rows:
                agg[k].append((op, ln, n))
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    res = {}
    for k, v in agg.items():
        res[k] = dict(op=gm([x[0] for x in v]), len=gm([x[1] for x in v]), max_len=100 * max(x[1] for x in v),
                      used=sum(x[2] > 0 for x in v), wp=sum(x[2] for x in v) / len(v))
        print(f"{k:22s} op {res[k]['op']:6.2f} len {res[k]['len']:7.2f} max {res[k]['max_len']:6.1f} "
              f"chains_used {res[k]['used']:4d} wp {res[k]['wp']:.2f}")
    (HERE / 'results' / 'endpoint_ceiling.json').write_text(json.dumps(res, indent=2))
