"""Per-map paired comparison on the 1000 original queries (all maps, LLM zero-waypoint maps included).
Methods per map: A*, wA*2, LLM-A* (standard/cot/repe, stored Llama-3, official filter),
random k=1/2/3 (5 draws in the start-goal rectangle, sorted along start->goal; per-map value = mean over draws),
random-matched (same number of surviving waypoints as that map's LLM output; 0 -> no waypoints = wA*2).
Metrics relative to A* on the same map: operations, storage, path length. Upstream accounting.
Outputs: results/per_map_compare.json (summary), results/per_map_rows.json (per-map rows), figs/per_map_compare.png"""
import sys
sys.dont_write_bytecode = True
import collections, json, math, random, statistics
from multiprocessing import Pool
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import offline_improve as o   # noqa: E402
import os
if os.environ.get('LLM_SRC'):          # e.g. outputs/gpt4omini-table1-upstream-batch-v1 (read-only)
    o.SRC = Path(os.environ['LLM_SRC'])
TAG = os.environ.get('TAG', '')        # suffix for output files, e.g. '_gpt4omini'

R = 5


def work(args):
    sample, q, texts = args
    env = o.Environment(json.loads(json.dumps(q)), collision_backend='shapely'); grid = o.Grid(env)
    base = o.search(env, grid, None)
    rel = lambda r: dict(op=r['op'] / base['op'], sto=r['sto'] / base['sto'], len=r['len'] / base['len'])
    row = dict(sample=sample, A_op=base['op'], wA2=rel(o.search(env, grid, None, wg=2.0)))
    S, G = env.start, env.goal
    obs = o.obstacle_set(env)
    cand = [(x, y) for x in range(min(S[0], G[0]), max(S[0], G[0]) + 1)
            for y in range(min(S[1], G[1]), max(S[1], G[1]) + 1)
            if (x, y) not in (S, G) and o.upstream_ok(env, (x, y), obs) and env.free((x, y))]
    proj = lambda p: (p[0] - S[0]) * (G[0] - S[0]) + (p[1] - S[1]) * (G[1] - S[1])

    def rand(k, tag):
        if k == 0:
            return row['wA2']
        rs = []
        for d in range(R):
            pts = sorted(random.Random(f'{sample}-{tag}-{k}-{d}').sample(cand, min(k, len(cand))), key=proj)
            rs.append(rel(o.search(env, grid, env.targets(pts, 'upstream'))))
        return {m: statistics.mean(r[m] for r in rs) for m in ('op', 'sto', 'len')}
    for k in (1, 2, 3):
        row[f'rand{k}'] = rand(k, 'fixed')
    for p in o.PROMPTS:
        try: nodes = o.parse_path(texts[p], mode='upstream')
        except ValueError: nodes = []
        T = env.targets(nodes, 'upstream')
        row[f'llm_{p}'] = dict(rel(o.search(env, grid, T)), k=len(T) - 2)
        row[f'match_{p}'] = rand(len(T) - 2, 'match')
    return row


def main():
    texts, _ = o.load_texts()
    tasks = [(s, q, {p: texts[(s, p)] for p in o.PROMPTS}) for s, q in o.load_queries(o.DATA)]
    with Pool(10) as pool:
        rows = sorted(pool.imap_unordered(work, tasks, chunksize=4), key=lambda r: r['sample'])
    (HERE / 'results' / f'per_map_rows{TAG}.json').write_text(json.dumps(rows))
    gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
    pct = lambda v, q: 100 * sorted(v)[int(q * (len(v) - 1))]
    keys = ['wA2', 'rand1', 'rand2', 'rand3'] + [f'{a}_{p}' for p in o.PROMPTS for a in ('llm', 'match')]
    res = {}
    print(f"{'method':16} {'op GM':>6} {'med':>6} {'p90':>6} {'max':>7} | {'sto GM':>6} | {'len GM':>7} {'p90':>7} {'max':>7} | "
          f"{'>A*':>5} {'win/loss vs wA2':>16}")
    for k in keys:
        op = [r[k]['op'] for r in rows]; ln = [r[k]['len'] for r in rows]; st = [r[k]['sto'] for r in rows]
        w = sum(r[k]['op'] < r['wA2']['op'] for r in rows); l = sum(r[k]['op'] > r['wA2']['op'] for r in rows)
        res[k] = dict(op_gm=gm(op), op_med=pct(op, .5), op_p90=pct(op, .9), op_max=100 * max(op), sto_gm=gm(st),
                      len_gm=gm(ln), len_p90=pct(ln, .9), len_max=100 * max(ln),
                      worse_than_Astar=sum(x > 1 for x in op), win_vs_wA2=w, loss_vs_wA2=l)
        r_ = res[k]
        print(f"{k:16} {r_['op_gm']:6.2f} {r_['op_med']:6.2f} {r_['op_p90']:6.2f} {r_['op_max']:7.1f} | {r_['sto_gm']:6.2f} | "
              f"{r_['len_gm']:7.2f} {r_['len_p90']:7.2f} {r_['len_max']:7.2f} | {r_['worse_than_Astar']:5d} {w:7d}/{l:<7d}")
    print('\npaired LLM vs random-matched / random k=1 (maps where LLM fewer ops / more ops / tie):')
    for p in o.PROMPTS:
        for ref in (f'match_{p}', 'rand1'):
            a = [r[f'llm_{p}']['op'] for r in rows]; b = [r[ref]['op'] for r in rows]
            win = sum(x < y - 1e-12 for x, y in zip(a, b)); loss = sum(x > y + 1e-12 for x, y in zip(a, b))
            la = [r[f'llm_{p}']['len'] for r in rows]; lb = [r[ref]['len'] for r in rows]
            res[f'pair_{p}_vs_{ref}'] = dict(llm_fewer_ops=win, llm_more_ops=loss, tie=len(a) - win - loss,
                                              op_ratio_gm=gm([x / y for x, y in zip(a, b)]),
                                              len_ratio_gm=gm([x / y for x, y in zip(la, lb)]))
            print(f"  {p:8} vs {ref:15} {win:4d} / {loss:4d} / {len(a) - win - loss:4d}   "
                  f"LLM/ref op {res[f'pair_{p}_vs_{ref}']['op_ratio_gm']:6.1f}%  len {res[f'pair_{p}_vs_{ref}']['len_ratio_gm']:6.2f}%")
    (HERE / 'results' / f'per_map_compare{TAG}.json').write_text(json.dumps(res, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    for ax, p in zip(axes, o.PROMPTS):
        x = [100 * r[f'match_{p}']['op'] for r in rows]; y = [100 * r[f'llm_{p}']['op'] for r in rows]
        c = ['#999' if r[f'llm_{p}']['k'] == 0 else '#d0542c' for r in rows]
        ax.scatter(x, y, s=8, c=c, alpha=.55)
        ax.plot([1, 1000], [1, 1000], color='k', lw=.8)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(5, 800); ax.set_ylim(5, 800)
        ax.axhline(100, color='#4a7fbf', lw=.7, ls='--'); ax.axvline(100, color='#4a7fbf', lw=.7, ls='--')
        pr = res[f'pair_{p}_vs_match_{p}']
        ax.set_title(f'{p}: LLM fewer ops on {pr["llm_fewer_ops"]} maps, more on {pr["llm_more_ops"]}, tie {pr["tie"]}')
        ax.set_xlabel('random waypoints, same count as LLM (op % of A*)'); ax.set_ylabel('LLM waypoints (op % of A*)')
        ax.scatter([], [], c='#999', label='LLM 0 waypoints (both = wA*2)'); ax.scatter([], [], c='#d0542c', label='LLM >=1 waypoint')
        ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout(); fig.savefig(HERE / 'figs' / f'per_map_compare{TAG}.png', dpi=105)


if __name__ == '__main__':
    main()
