from pathlib import Path
import json,sys,math,collections,statistics,gzip,hashlib,zipfile,re,csv,shutil
import numpy as np
P=Path('/home/skkai/Mobility/2026_2/ykim/assign');sys.path.insert(0,str(P))
from llm_astar.__main__ import load_queries,summarize
from llm_astar.models import build_prompt,parse_path
from llm_astar.search import Environment,path_result
SRC=P/'outputs/gpt4omini-cot-vot-batch-v1';OUT=P/'outputs/gpt4omini-cot-vot-analysis';OUT.mkdir(exist_ok=True)
rows=[json.loads(l) for l in (SRC/'results.jsonl').read_text().splitlines()];old=[json.loads(l) for l in (P/'outputs/gpt4omini-table1-upstream-batch-v1/results.jsonl').read_text().splitlines()]
N={(r['sample'],r['method'],r['prompt']):r for r in rows};O={(r['sample'],r['method'],r['prompt']):r for r in old}
Q=dict(load_queries(SRC/'input.json'));samples=sorted(Q,key=lambda s:tuple(map(int,s.split(':'))));R={p.stem:json.loads(p.read_text()) for p in (SRC/'responses').glob('*.json')};summary=summarize(rows)
config=json.loads((P/'configs/gpt4omini_cot_vot.json').read_text());batch=json.loads(next((SRC/'logs/batch').glob('batch_*.json')).read_text());signature=json.loads((SRC/'experiment-code.json').read_text())
with zipfile.ZipFile(str(SRC)+'.zip') as z:assert z.testzip() is None
checks={'rows':len(rows),'unique_rows':len(N),'responses':len(R),'zip_integrity':True,'summary_matches':summary==json.loads((SRC/'summary.json').read_text()),'input_matches_dataset':json.loads((SRC/'input.json').read_text())==json.loads((P/'data/environment_50_30.json').read_text()),'baseline_matches_original':all(all(N[s,'astar','-'][k]==O[s,'astar','-'][k] for k in ['path','valid','operation','storage','length']) for s in samples),'source_hashes_match':all(hashlib.sha256((P/k).read_bytes()).hexdigest()==v for k,v in signature['source_sha256'].items()),'paths_rechecked':0,'prompts_rechecked':0,'batch_response_count':sum('batch_id' in r for r in R.values()),'synchronous_response_count':sum('batch_id' not in r for r in R.values()),'returned_models':dict(collections.Counter(r['returned_model'] for r in R.values()))}
assert len(rows)==len(N)==3000 and len(R)==1000 and checks['summary_matches'] and checks['baseline_matches_original']
for r in rows:
 env=Environment(Q[r['sample']]);checked=path_result(env,[tuple(v) for v in r['path']]);assert checked['valid']==r['valid']
 if r['valid']:assert math.isclose(checked['length'],r['length'],abs_tol=1e-8)
 checks['paths_rechecked']+=1
 if r['method']=='llm':
  response=R[r['cache_key']];assert response['config']==config and response['prompt']==build_prompt(Q[r['sample']],r['prompt'],'gpt','vot')
  try:parsed=[list(v) for v in parse_path(response['text'],mode='upstream')]
  except ValueError:parsed=[]
  assert parsed==r['path'];checks['prompts_rechecked']+=1
print('Integrity checks:',json.dumps(checks),flush=True)
rng=np.random.default_rng(20261121);boot=rng.integers(0,100,size=(20000,100))
def gm(v):return float(np.exp(np.log(v).mean()))
def ci(v,log=False):
 a=np.array(v,dtype=float).reshape(100,10)
 if log:a=np.log(a)
 x=a.mean(axis=1)[boot].mean(axis=1)
 if log:x=np.exp(x)
 return np.quantile(x,[.025,.975]).tolist()
def points(r):return max(0,len(r['path'])-2)
def failures(rs):
 c=collections.Counter()
 for r in rs:
  if r['valid']:continue
  env=Environment(Q[r['sample']]);path=[tuple(v) for v in r['path']]
  if not path or path[0]!=env.start or path[-1]!=env.goal:c['wrong_or_missing_endpoints']+=1
  if any(not env.inside(v) for v in path):c['outside_or_boundary_point']+=1
  if any(env.inside(v) and not env.free(v) for v in path):c['waypoint_on_barrier']+=1
  if any(env._line([a,b]).intersects(env._line([u,v])) for a,b in zip(path,path[1:]) for u,v in env.walls):c['barrier_intersection']+=1
 return dict(c)
weights=[]
for l in gzip.open(P/'exp/claude/results/per_sample.jsonl.gz','rt'):
 r=json.loads(l)
 if r['kind']=='astar' and r['variant']=='wA*2.0' and not r['dedupe'] and r['tie']=='node':weights.append(r)
assert len(weights)==1000;W={r['sample']:r for r in weights};base={s:N[s,'astar','-'] for s in samples}
A={'verification_status':'ANALYZED','checks':checks,'batch':batch,'batch_wall_seconds':batch['completed_at']-batch['created_at'],'gpt_summary':summary,'original_summary':summarize(old),'weighted_astar_2':{metric:100*gm([W[s][key]/base[s][metric] for s in samples]) for metric,key in [('operation','op'),('storage','sto'),('length','len')]},'bootstrap':{'unit':'map cluster:100 maps x10 start-goal pairs','replicates':20000,'seed':20261121,'interval':'95%, exploratory; no multiplicity correction'},'by_prompt':{}}
paired=[];examples=[]
for p in ['cot']:
 raw=[N[s,'llm',p] for s in samples];oldraw=[O[s,'llm',p] for s in samples];hy=[N[s,'llm-astar',p] for s in samples];oh=[O[s,'llm-astar',p] for s in samples]
 tokens=[r['usage']['completion_tokens'] for r in hy];inter=[points(r) for r in raw];responses=[R[r['cache_key']] for r in raw]
 d={'llm_success_delta_pp':100*float(np.mean([int(a['valid'])-int(b['valid']) for a,b in zip(raw,oldraw)])),'llm_success_delta_pp_ci':[100*v for v in ci([int(a['valid'])-int(b['valid']) for a,b in zip(raw,oldraw)])],'llm_transitions_from_original':dict(collections.Counter(('valid' if b['valid'] else 'invalid')+'_to_'+('valid' if a['valid'] else 'invalid') for a,b in zip(raw,oldraw))),'failure_categories_overlapping':failures(raw),'no_bracket_list':sum(not re.findall(r'\[\[.*?\]\]',r['text'],re.S) for r in responses),'waypoints':{'mean':statistics.mean(inter),'median':statistics.median(inter),'p95':float(np.quantile(inter,.95)),'max':max(inter),'ge10':sum(v>=10 for v in inter),'ge20':sum(v>=20 for v in inter)},'completion_tokens':{'total':sum(tokens),'mean':statistics.mean(tokens),'median':statistics.median(tokens),'p95':float(np.quantile(tokens,.95)),'max':max(tokens)},'input_tokens':sum(r['usage']['prompt_tokens'] for r in hy),'truncated':sum(r['finish_reason']=='length' for r in raw),'truncated_raw_valid':sum(r['valid'] and r['finish_reason']=='length' for r in raw),'comparison_to_original':{},'comparison_to_weighted_astar':{},'hybrid_worse_than_astar_ops':sum(r['operation']>base[r['sample']]['operation'] for r in hy),'hybrid_length_over_10pct':sum(r['length']>1.1*base[r['sample']]['length'] for r in hy)}
 for metric,key in [('operation','op'),('storage','sto'),('length','len')]:
  ratios=[a[metric]/b[metric] for a,b in zip(hy,oh)];d['comparison_to_original'][metric]={'vot_over_original_geomean':gm(ratios),'ratio_ci':ci(ratios,True),'gpt_better':sum(a[metric]<b[metric]-1e-8 for a,b in zip(hy,oh)),'gpt_worse':sum(a[metric]>b[metric]+1e-8 for a,b in zip(hy,oh))}
  ratios=[a[metric]/W[a['sample']][key] for a in hy];d['comparison_to_weighted_astar'][metric]={'gpt_over_weighted_geomean':gm(ratios),'ratio_ci':ci(ratios,True),'gpt_better':sum(a[metric]<W[a['sample']][key]-1e-8 for a in hy)}
 common=[(a,b) for a,b in zip(raw,oldraw) if a['valid'] and b['valid']];d['both_raw_valid']=len(common);d['common_raw_length_vot_over_original']=gm([a['length']/b['length'] for a,b in common]) if common else None
 
 sentence='Visualize the state after each reasoning step.\n'
 for r in raw:
  response=R[r['cache_key']];assert response['prompt'].count(sentence)==1 and response['prompt'].replace(sentence,'',1)==build_prompt(Q[r['sample']],p,'gpt','upstream')
 d['original_completion_tokens']={'total':sum(r['usage']['completion_tokens'] for r in oldraw),'median':statistics.median(r['usage']['completion_tokens'] for r in oldraw)}
 d['original_waypoints']={'mean':statistics.mean(points(r) for r in oldraw),'median':statistics.median(points(r) for r in oldraw),'max':max(points(r) for r in oldraw),'ge10':sum(points(r)>=10 for r in oldraw)}
 def grid_rows(text):
  # Conservative screen: rows composed mainly of spatial drawing symbols; excludes headings and coordinate lists.
  good=[]
  for i,line in enumerate(text.splitlines()):
   if re.fullmatch(r'[0-9SGWsgwxyzXYZ .#|+_=─━│┌┐└┘⬜⬛🟩🟥🟦🟨🏠🏢*→←↑↓\-]+',line) and len(re.findall(r'[.#|+_=─━│┌┐└┘⬜⬛🟩🟥🟦🟨🏠🏢*→←↑↓\-]',line))>=5:
    good.append(i)
  return good
 screens=[];fence_types=collections.Counter()
 for r in raw:
  text=R[r['cache_key']]['text'];indices=grid_rows(text)
  before=text.rsplit('Generated Path:',1)[0] if 'Generated Path:' in text else text
  for block in re.findall(r'```[^\n]*\n(.*?)```',text,re.S):
   fence_types['path_list' if 'Generated Path:' in block or block.lstrip().startswith('[[') else 'other']+=1
  screens.append({'sample':r['sample'],'grid_like_rows':indices,'candidate_grid':len(indices)>=3,'fenced':bool('```' in text),'visualize_word':bool(re.search(r'visualiz',before,re.I))})
 d['visualization_screen']={'candidate_grids':sum(r['candidate_grid'] for r in screens),'responses_with_fences':sum(r['fenced'] for r in screens),'fenced_block_types':dict(fence_types),'visualization_word_before_answer':sum(r['visualize_word'] for r in screens),'method':'Heuristic drawing-row screen (>=3 rows, no brackets/commas, >=5 spatial symbols per row), plus fenced-block inspection. Not a semantic visual accuracy audit.'}
 (OUT/'visualization-screen.json').write_text(json.dumps(screens,indent=2))
 
 A['by_prompt'][p]=d
 for r in sorted(raw,key=lambda r:-points(r))[:3]:
  resp=R[r['cache_key']];examples.append({'sample':r['sample'],'prompt':p,'intermediate_count':points(r),'raw_valid':r['valid'],'finish_reason':r['finish_reason'],'response_start':resp['text'][:500],'response_end':resp['text'][-500:]})
 for a,b in zip(hy,oh):paired.append({'sample':a['sample'],'prompt':p,'gpt_operation':a['operation'],'llama_operation':b['operation'],'weighted_operation':W[a['sample']]['op'],'gpt_length':a['length'],'llama_length':b['length'],'weighted_length':W[a['sample']]['len'],'gpt_raw_valid':N[a['sample'],'llm',p]['valid'],'llama_raw_valid':O[a['sample'],'llm',p]['valid']})
(OUT/'analysis.json').write_text(json.dumps(A,indent=2));(OUT/'long-output-examples.json').write_text(json.dumps(examples,indent=2))
with (OUT/'paired-comparison.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
shutil.copy('/tmp/analyze_gptvot.py',OUT/'analyze.py')
print(json.dumps({'weighted':A['weighted_astar_2'],'by_prompt':A['by_prompt']},indent=2),flush=True)
