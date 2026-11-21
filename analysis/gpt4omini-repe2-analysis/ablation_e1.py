"""E1 ablation: paired operation ratios between RePE2, its two ablations and the released prompts.
Run: python analysis/gpt4omini-repe2-analysis/ablation_e1.py"""
import json, math, random
from pathlib import Path

RUNS = {'standard': ('outputs/gpt4omini-table1-upstream-batch-v1', 'standard'),
        'cot': ('outputs/gpt4omini-table1-upstream-batch-v1', 'cot'),
        'repe': ('outputs/gpt4omini-table1-upstream-batch-v1', 'repe'),
        'RePE2': ('outputs/gpt4omini-repe2-batch-v1', 'repe2'),
        'E1-a new demos, no rounds': ('outputs/gpt4omini-repe2_noround-batch-v1', 'repe2_noround'),
        'E1-b released demos + rounds': ('outputs/gpt4omini-repe2_origdemos-batch-v1', 'repe2_origdemos')}
op, base = {}, {}
for name, (d, p) in RUNS.items():
    for r in map(json.loads, open(Path(d) / 'results.jsonl')):
        if r['method'] == 'astar':
            base[r['sample']] = r['operation']
        elif r['method'] == 'llm-astar' and r['prompt'] == p:
            op.setdefault(name, {})[r['sample']] = r['operation']
S = sorted(base)
gm = lambda v: 100 * math.exp(sum(map(math.log, v)) / len(v))
random.seed(0)


def paired(a, b):
    pr = [math.log(op[a][s] / op[b][s]) for s in S]
    bs = sorted(math.exp(sum(random.choice(pr) for _ in pr) / len(pr)) * 100 for _ in range(2000))
    return f'{math.exp(sum(pr) / len(pr)) * 100:.1f} [{bs[50]:.1f}, {bs[1950]:.1f}]'


lines = [f'{n:32s} op {gm([op[n][s] / base[s] for s in S]):6.2f}' for n in RUNS]
for a, b in [('E1-a new demos, no rounds', 'standard'), ('E1-b released demos + rounds', 'standard'),
             ('E1-b released demos + rounds', 'RePE2'), ('RePE2', 'E1-a new demos, no rounds'),
             ('E1-b released demos + rounds', 'E1-a new demos, no rounds'), ('E1-b released demos + rounds', 'repe')]:
    lines.append(f'{a} / {b}: {paired(a, b)}')
print('\n'.join(lines))
Path('outputs/gpt4omini-repe2-analysis/ablation_e1.txt').write_text('\n'.join(lines) + '\n')
