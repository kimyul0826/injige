"""Calibration of the RePE2 self-check (no API calls): every 'Round k: [[...]]' path in a response is paired with
the verdict the model wrote after it and compared with the exact collision test.
Reports a confusion matrix over all rounds, per-round-index accuracy, and how often a 'Not valid' round was
followed by a round that actually fixed the collision.
Usage: python exp/claude/repe2_selfcheck.py [output dir ...]  (default: RePE2 batch and E1-b if present)"""
import sys
sys.dont_write_bytecode = True
import ast, json, re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from llm_astar.__main__ import load_queries   # noqa: E402
from llm_astar.search import Environment      # noqa: E402

dirs = sys.argv[1:] or [d for d in ('outputs/gpt4omini-repe2-batch-v1', 'outputs/gpt4omini-repe2_origdemos-batch-v1')
                        if (ROOT / d / 'summary.json').exists()]
qs = dict(load_queries(ROOT / 'data/environment_50_30.json'))
ROUND = re.compile(r'Round\s*(\d+)\**\s*:?\**\s*(?:\n[^\[\n]*)?(\[\[.*?\]\])(.*?)(?=Round\s*\d+|Generated Path|$)', re.S)
out = []
for d in dirs:
    cm, by_k, fixes, n_resp, parsed = Counter(), Counter(), Counter(), 0, 0
    for r in map(json.loads, open(ROOT / d / 'results.jsonl')):
        if r['method'] != 'llm-astar':
            continue
        n_resp += 1
        env = Environment(qs[r['sample']], collision_backend='shapely')
        text = json.loads((ROOT / d / 'responses' / f"{r['cache_key']}.json").read_text())['text']
        prev_claim = None
        for m in ROUND.finditer(text):
            try:
                path = [tuple(p) for p in ast.literal_eval(m[2])]
            except (ValueError, SyntaxError):
                continue
            v = re.search(r'\b(Not valid|Valid)\b', m[3])
            if not v or len(path) < 2:
                continue
            parsed += 1
            truth = 'valid' if not any(env.collision(a, b) for a, b in zip(path, path[1:])) else 'collides'
            claim = 'Valid' if v[1] == 'Valid' else 'Not valid'
            cm[(claim, truth)] += 1
            k = min(int(m[1]), 3)
            by_k[(k, claim == 'Valid') if True else None] += 0
            by_k[(k, (claim == 'Valid') == (truth == 'valid'))] += 1
            if prev_claim == 'Not valid':
                fixes[truth] += 1
            prev_claim = claim
    acc = sum(v for (c, t), v in cm.items() if (c == 'Valid') == (t == 'valid')) / max(1, sum(cm.values()))
    lines = [f'== {d}: responses {n_resp}, rounds parsed {parsed}, verdict accuracy {100 * acc:.1f}%',
             '   claim Valid    : truly valid %d, actually collides %d' % (cm[('Valid', 'valid')], cm[('Valid', 'collides')]),
             '   claim Not valid: actually collides %d, truly valid %d' % (cm[('Not valid', 'collides')], cm[('Not valid', 'valid')]),
             '   accuracy by round index: ' + '  '.join(
                 f"k={k}{'+' if k == 3 else ''} {100 * by_k[(k, True)] / max(1, by_k[(k, True)] + by_k[(k, False)]):.0f}% (n={by_k[(k, True)] + by_k[(k, False)]})"
                 for k in range(4)),
             '   round after a "Not valid": actually valid %d, still collides %d' % (fixes['valid'], fixes['collides'])]
    out += lines
print('\n'.join(out))
(HERE / 'results' / 'repe2_selfcheck.txt').write_text('\n'.join(out) + '\n')
