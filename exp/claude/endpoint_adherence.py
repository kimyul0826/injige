"""Per-response endpoint adherence + LLM-A* op for barrier-endpoint runs. Usage: python endpoint_adherence.py <out dir>..."""
import json, math, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm_astar.__main__ import load_queries
from llm_astar.models import parse_path
from llm_astar.prompt_variants import barrier_endpoints

for O in map(Path, sys.argv[1:]):
    qs = dict(load_queries(O / 'input.json'))
    rows = [json.loads(l) for l in (O / 'results.jsonl').read_text().splitlines()]
    base = {r['sample']: r for r in rows if r['method'] == 'astar'}
    st = {}
    print('==', O.name)
    for r in rows:
        if r['method'] != 'llm-astar':
            continue
        q = qs[r['sample']]
        E = {tuple(e) for e in barrier_endpoints(q)}
        resp = json.loads((O / 'responses' / f"{r['cache_key']}.json").read_text())
        try:
            P = parse_path(resp['text'], 'upstream')
        except Exception:
            P = []
        mid = [p for p in P if p not in (tuple(q['start']), tuple(q['goal']))]
        s = st.setdefault(r['prompt'], Counter())
        s['n'] += 1; s['wp'] += len(mid); s['in'] += sum(p in E for p in mid)
        s['all_in'] += all(p in E for p in mid); s['zero_wp'] += not mid
        s['out_tokens'] += resp['usage'].get('completion_tokens', 0)
        op = r['operation'] / base[r['sample']]['operation']
        s.setdefault('logop', 0); s['logop'] += math.log(op)
        print(f"{r['sample']:6s} {r['prompt']:8s} op {100*op:6.1f} wp {mid} off={[p for p in mid if p not in E]}")
    for p, s in st.items():
        print(f"{p:8s} op {100*math.exp(s['logop']/s['n']):6.2f} wp {s['wp']} in_list {s['in']} all_in {s['all_in']}/{s['n']} "
              f"zero_wp {s['zero_wp']} mean_out_tokens {s['out_tokens']/s['n']:.0f}")
