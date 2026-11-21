from pathlib import Path
import json,zipfile,sys,hashlib,math,collections,statistics,csv,gzip,shutil
import numpy as np
P=Path('/home/skkai/Mobility/2026_2/ykim/assign');sys.path.insert(0,str(P))
from llm_astar.__main__ import summarize,load_queries
from llm_astar.models import build_prompt,parse_path
from llm_astar.search import Environment,path_result
OUT=P/'outputs/llama3-waypoint-limit-v1-analysis';OUT.mkdir(exist_ok=True)
old=[json.loads(l) for l in (P/'outputs/llama3-table1-upstream-v2/results.jsonl').read_text().splitlines()]
with zipfile.ZipFile(P/'outputs/llama3-waypoint-limit-v1.zip') as z:
 assert z.testzip() is None
 root='llama3-waypoint-limit-v1/'
 new=[json.loads(l) for l in z.read(root+'results.jsonl').splitlines()]
 manifest=json.loads(z.read(root+'manifest.json'));stored=json.loads(z.read(root+'summary.json'));status=json.loads(z.read(root+'status.json'))
 responses={Path(n).stem:json.loads(z.read(n)) for n in z.namelist() if '/responses/' in n}
 input_matches=json.loads(z.read(root+'input.json'))==json.loads((P/'data/environment_50_30.json').read_text())
Q=dict(load_queries(P/'data/environment_50_30.json'));samples=sorted(Q,key=lambda s:tuple(map(int,s.split(':'))))
idx=lambda rows:{(r['sample'],r['method'],r['prompt']):r for r in rows}
O,N=idx(old),idx(new);base={s:N[s,'astar','-'] for s in samples};cap=lambda s:len(Q[s]['horizontal_barriers'])+len(Q[s]['vertical_barriers'])
checks={'zip_integrity':True,'rows':len(new),'unique_rows':len(N),'summary_recalculation_matches':summarize(new)==stored,'input_semantically_matches':input_matches,'responses':len(responses),'local_source_hashes_match':{k:hashlib.sha256((P/k).read_bytes()).hexdigest()==v for k,v in manifest['source_sha256'].items()},'baseline_equal_to_original':all(all(base[s][k]==O[s,'astar','-'][k] for k in ['path','valid','operation','storage','length']) for s in samples),'paths_rechecked':0,'prompts_rechecked':0}
assert len(N)==len(new)==7000 and len(responses)==3000
for r in new:
 e=Environment(Q[r['sample']]);rr=path_result(e,[tuple(p) for p in r['path']]);assert rr['valid']==r['valid']
 if r['valid']:assert math.isclose(rr['length'],r['length'],abs_tol=1e-8)
 checks['paths_rechecked']+=1
 if r['method']=='llm':
  response=responses[r['cache_key']];assert response['prompt']==build_prompt(Q[r['sample']],r['prompt'],'llama','waypoint_limit_by_obstacles')
  sentence=f"\nFor this problem, use at most {cap(r['sample'])} intermediate waypoints in the final generated path, excluding the start and goal points."
  assert response['prompt'].count(sentence)==1 and response['prompt'].replace(sentence,'',1)==build_prompt(Q[r['sample']],r['prompt'],'llama','upstream')
  assert [list(p) for p in parse_path(response['text'],mode='upstream')]==r['path'];checks['prompts_rechecked']+=1
rng=np.random.default_rng(20261121);boot=rng.integers(0,100,size=(20000,100))
def gm(v):return math.exp(sum(math.log(x) for x in v)/len(v))
def ci(v,log=False):
 a=np.asarray(v,dtype=float).reshape(100,10)
 if log:a=np.log(a)
 b=a.mean(axis=1)[boot].mean(axis=1)
 if log:b=np.exp(b)
 return np.quantile(b,[.025,.975]).tolist()
def endpoints(r):return bool(r['path']) and r['path'][0]==Q[r['sample']]['start'] and r['path'][-1]==Q[r['sample']]['goal']
def ninter(r):return max(0,len(r['path'])-2)
def count_stats(rows):
 proper=[r for r in rows if endpoints(r)];counts=[ninter(r) for r in rows]
 return {'mean_len_minus_two_all':statistics.mean(counts),'median_len_minus_two_all':statistics.median(counts),'p95_len_minus_two_all':float(np.quantile(counts,.95)),'max_len_minus_two_all':max(counts),'endpoints_correct':len(proper),'exceeds_cap_with_correct_endpoints':sum(ninter(r)>cap(r['sample']) for r in proper),'within_cap_and_correct_endpoints':sum(ninter(r)<=cap(r['sample']) for r in proper),'exceeds_cap_len_minus_two_all':sum(ninter(r)>cap(r['sample']) for r in rows),'exactly_at_cap_with_correct_endpoints':sum(ninter(r)==cap(r['sample']) for r in proper),'zero_intermediate_with_correct_endpoints':sum(ninter(r)==0 for r in proper),'within_cap_rate_valid_endpoints':100*sum(ninter(r)<=cap(r['sample']) for r in proper)/len(proper)}
def failures(rows):
 c=collections.Counter()
 for r in rows:
  if r['valid']:continue
  e=Environment(Q[r['sample']]);path=[tuple(p) for p in r['path']]
  if not endpoints(r):c['wrong_endpoints']+=1
  if any(not e.inside(n) for n in path):c['boundary_or_outside_point']+=1
  if any(e.inside(n) and not e.free(n) for n in path):c['waypoint_on_barrier']+=1
  if any(e._line([u,v]).intersects(e._line([a,b])) for u,v in zip(path,path[1:]) for a,b in e.walls):c['barrier_intersection']+=1
 return dict(c)
A={'verification_status':'ANALYZED','checks':checks,'status':status,'new_packages':manifest['packages'],'bootstrap':{'unit':'100 maps, each with 10 start-goal queries','replicates':20000,'seed':20261121,'ci':'95%, exploratory, no multiple comparison correction'},'old_summary':summarize(old),'new_summary':summarize(new),'by_prompt':{}};paired=[];examples=[]
for p in ['standard','cot','repe']:
 a=[O[s,'llm',p] for s in samples];b=[N[s,'llm',p] for s in samples];ha=[O[s,'llm-astar',p] for s in samples];hb=[N[s,'llm-astar',p] for s in samples]
 common=[(x,y) for x,y in zip(a,b) if x['valid'] and y['valid']]
 d={'old_waypoints':count_stats(a),'new_waypoints':count_stats(b),'llm_transitions':dict(collections.Counter(('valid' if x['valid'] else 'invalid')+'_to_'+('valid' if y['valid'] else 'invalid') for x,y in zip(a,b))),'llm_success_difference_pp_ci':[100*x for x in ci([int(y['valid'])-int(x['valid']) for x,y in zip(a,b)])],'common_success':len(common),'common_success_length_old_pct':100*gm([x['length']/base[x['sample']]['length'] for x,y in common]),'common_success_length_new_pct':100*gm([y['length']/base[y['sample']]['length'] for x,y in common]),'old_failures':failures(a),'new_failures':failures(b),'waypoints_increased':sum(ninter(y)>ninter(x) for x,y in zip(a,b)),'waypoints_decreased':sum(ninter(y)<ninter(x) for x,y in zip(a,b)),'median_generation_seconds':statistics.median(r['generation_seconds'] for r in hb),'median_completion_tokens':statistics.median(r['usage']['completion_tokens'] for r in hb),'hybrid_changes':{}}
 for metric in ['operation','storage','length']:
  ratios=[y[metric]/x[metric] for x,y in zip(ha,hb)];d['hybrid_changes'][metric]={'new_over_old_geomean':gm(ratios),'ratio_ci':ci(ratios,True),'new_better':sum(y[metric]<x[metric]-1e-8 for x,y in zip(ha,hb)),'new_worse':sum(y[metric]>x[metric]+1e-8 for x,y in zip(ha,hb))}
 for label,rows in [('old',ha),('new',hb)]:
  d[label+'_hybrid_tail']={'operation_worse_than_astar':sum(r['operation']>base[r['sample']]['operation'] for r in rows),'operation_over_2x':sum(r['operation']>2*base[r['sample']]['operation'] for r in rows),'length_over_10pct':sum(r['length']>1.1*base[r['sample']]['length'] for r in rows)}
 A['by_prompt'][p]=d
 for x,y in zip(a,b):
  paired.append({'sample':x['sample'],'prompt':p,'cap':cap(x['sample']),'old_llm_valid':x['valid'],'new_llm_valid':y['valid'],'old_intermediate_len_minus_two':ninter(x),'new_intermediate_len_minus_two':ninter(y),'old_endpoints_correct':endpoints(x),'new_endpoints_correct':endpoints(y),'old_hybrid_operation':O[x['sample'],'llm-astar',p]['operation'],'new_hybrid_operation':N[y['sample'],'llm-astar',p]['operation']})
  if x['valid'] and not y['valid'] and len(examples)<10:examples.append({'sample':x['sample'],'prompt':p,'query':Q[x['sample']],'old_path':x['path'],'new_path':y['path'],'new_response':responses[y['cache_key']]['text']})
weights=[]
for line in gzip.open(P/'exp/claude/results/per_sample.jsonl.gz','rt'):
 r=json.loads(line)
 if r['kind']=='astar' and r['variant']=='wA*2.0' and not r['dedupe'] and r['tie']=='node':weights.append(r)
assert len(weights)==1000
A['weighted_astar_2']={metric:100*gm([r[key]/base[r['sample']][metric] for r in weights]) for metric,key in [('operation','op'),('storage','sto'),('length','len')]}
(OUT/'analysis.json').write_text(json.dumps(A,indent=2,ensure_ascii=False));(OUT/'examples.json').write_text(json.dumps(examples,indent=2))
with (OUT/'paired-comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
shutil.copy('/tmp/analyze_waypoint_limit.py',OUT/'analyze.py')
print(json.dumps({'checks':checks,'by_prompt':A['by_prompt'],'weighted_astar_2':A['weighted_astar_2']},indent=2,ensure_ascii=False))
