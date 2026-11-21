from pathlib import Path
import json,zipfile,sys,hashlib,math,collections,statistics,csv,shutil
import numpy as np
P=Path('/home/skkai/Mobility/2026_2/ykim/assign');sys.path.insert(0,str(P))
from llm_astar.__main__ import summarize,load_queries
from llm_astar.search import Environment,path_result
from llm_astar.models import build_prompt,parse_path
OUT=P/'outputs/llama3-8way-only-v1-analysis';OUT.mkdir(exist_ok=True)
def loadzip(fn):
 z=zipfile.ZipFile(P/'outputs'/fn);assert z.testzip() is None
 root=next(n.rsplit('/',1)[0] for n in z.namelist() if n.endswith('/manifest.json'))
 rows=[json.loads(l) for l in z.read(root+'/results.jsonl').splitlines()]
 return z,root,rows,json.loads(z.read(root+'/manifest.json'))
z,root,new,manifest=loadzip('llama3-8way-only-v1.zip')
z3,r3,v3a,m3=loadzip('llama3-diagonal-barriers-v3.zip');zc,rc,v3b,mc=loadzip('llama3-diagonal-barriers-cot-repe-v3.zip')
v3=v3a+[r for r in v3b if r['method']!='astar']
v2=[json.loads(l) for l in (P/'outputs/llama3-table1-upstream-v2/results.jsonl').read_text().splitlines()]
index=lambda rs:{(r['sample'],r['method'],r['prompt']):r for r in rs}
N=index(new);refs={'v2_original':index(v2),'v3_explicit_barriers':index(v3)}
queries=dict(load_queries(P/'data/environment_50_30.json'));samples=sorted(queries,key=lambda s:tuple(map(int,s.split(':'))));base={s:N[s,'astar','-'] for s in samples}
assert len(new)==len(N)==7000
checks={'zip_integrity':True,'rows':7000,'stored_summary_matches':summarize(new)==json.loads(z.read(root+'/summary.json')),'input_semantically_equal':json.loads(z.read(root+'/input.json'))==json.loads((P/'data/environment_50_30.json').read_text()),'same_input_hash_as_v3':manifest['input_sha256']==m3['input_sha256'],'same_packages_as_v3':manifest['packages']==m3['packages'],'same_model_metadata_as_v3':manifest['model_identity']==m3['model_identity'],'same_search_code_hash_as_v3':manifest['source_sha256']['llm_astar/search.py']==m3['source_sha256']['llm_astar/search.py'],'local_source_matches_manifest':{k:hashlib.sha256((P/k).read_bytes()).hexdigest()==v for k,v in manifest['source_sha256'].items()}}
checks['baselines_equal_to_both_references']=all(all(base[s][k]==ref[s,'astar','-'][k] for k in ['path','operation','storage','length','valid']) for ref in refs.values() for s in samples)
responses={Path(n).stem:json.loads(z.read(n)) for n in z.namelist() if '/responses/' in n}
checks['response_count']=len(responses);assert len(responses)==3000
checks['path_verification_count']=0;checks['prompt_exactly_one_added_sentence_count']=0;checks['parse_verification_count']=0
for r in new:
 env=Environment(queries[r['sample']]);evaluated=path_result(env,[tuple(p) for p in r['path']]);assert evaluated['valid']==r['valid']
 if r['valid']:assert math.isclose(evaluated['length'],r['length'],abs_tol=1e-8)
 checks['path_verification_count']+=1
 if r['method']=='llm':
  response=responses[r['cache_key']];original=build_prompt(queries[r['sample']],r['prompt'],'llama','upstream');added='\nYou can move horizontally, vertically, and diagonally.'
  assert response['prompt'].count(added)==1 and response['prompt'].replace(added,'',1)==original
  checks['prompt_exactly_one_added_sentence_count']+=1
  assert [list(p) for p in parse_path(response['text'],mode='upstream')]==r['path'];checks['parse_verification_count']+=1
rng=np.random.default_rng(20261120);boot=rng.integers(0,100,size=(20000,100))
def ci(values,log=False):
 a=np.array(values,dtype=float).reshape(100,10)
 if log:a=np.log(a)
 b=a.mean(axis=1)[boot].mean(axis=1)
 if log:b=np.exp(b)
 return np.quantile(b,[.025,.975]).tolist()
def gm(values):return math.exp(sum(math.log(x) for x in values)/len(values))
result={'verification_status':'ANALYZED','checks':checks,'status':json.loads(z.read(root+'/status.json')),'bootstrap':{'unit':'map','maps':100,'queries_per_map':10,'replicates':20000,'seed':20261120,'confidence':'95% exploratory, no multiple-comparison correction'},'summaries':{'v2_original':summarize(v2),'v3_explicit_barriers':summarize(v3),'eight_way_only':summarize(new)},'by_prompt':{}}
paired=[]
for p in ['standard','cot','repe']:
 rows=[N[s,'llm',p] for s in samples];hybrid=[N[s,'llm-astar',p] for s in samples]
 failures=collections.Counter();truncated=[]
 for r in rows:
  if r['finish_reason']=='length':truncated.append(r['sample'])
  if r['valid']:continue
  e=Environment(queries[r['sample']]);path=[tuple(n) for n in r['path']]
  if not path:failures['empty_path']+=1;continue
  if path[0]!=e.start or path[-1]!=e.goal:failures['wrong_endpoints']+=1
  if any(not e.inside(n) for n in path):failures['boundary_or_outside_point']+=1
  if any(not e.free(n) and e.inside(n) for n in path):failures['waypoint_on_barrier']+=1
  if any(e._line([a,b]).intersects(e._line([u,v])) for a,b in zip(path,path[1:]) for u,v in e.walls):failures['barrier_intersection']+=1
 filtered=[path_result(Environment(queries[r['sample']]),Environment(queries[r['sample']]).targets([tuple(n) for n in r['path']],'upstream')) for r in rows]
 d={'failure_counts_overlapping':dict(failures),'truncated_samples':truncated,'filtered_llm_success_count':sum(r['valid'] for r in filtered),'median_generation_seconds':statistics.median(r['generation_seconds'] for r in hybrid),'median_search_seconds':statistics.median(r['search_seconds'] for r in hybrid),'median_completion_tokens':statistics.median(r['usage']['completion_tokens'] for r in hybrid),'completion_tokens_total':sum(r['usage']['completion_tokens'] for r in hybrid),'hybrid_operation_worse_than_astar':sum(r['operation']>base[r['sample']]['operation'] for r in hybrid),'hybrid_operation_over_2x':sum(r['operation']>2*base[r['sample']]['operation'] for r in hybrid),'hybrid_length_over_10pct':sum(r['length']>1.1*base[r['sample']]['length'] for r in hybrid),'hybrid_total_operations_vs_astar':sum(r['operation'] for r in hybrid)/sum(base[s]['operation'] for s in samples),'comparisons':{}}
 for refname,ref in refs.items():
  old=[ref[s,'llm',p] for s in samples];oh=[ref[s,'llm-astar',p] for s in samples];common=[(a,b) for a,b in zip(old,rows) if a['valid'] and b['valid']]
  trans=collections.Counter(('valid' if a['valid'] else 'invalid')+'_to_'+('valid' if b['valid'] else 'invalid') for a,b in zip(old,rows))
  comp={'llm_transitions':dict(trans),'llm_success_difference_pp':100*(sum(r['valid'] for r in rows)-sum(r['valid'] for r in old))/1000,'llm_success_difference_pp_ci':[100*x for x in ci([int(b['valid'])-int(a['valid']) for a,b in zip(old,rows)])],'common_success_count':len(common),'common_llm_length_old_pct':100*gm([a['length']/base[a['sample']]['length'] for a,b in common]),'common_llm_length_new_pct':100*gm([b['length']/base[b['sample']]['length'] for a,b in common]),'hybrid_metrics':{}}
  for metric in ['operation','storage','length']:
   ratios=[b[metric]/a[metric] for a,b in zip(oh,hybrid)];comp['hybrid_metrics'][metric]={'new_over_reference_geomean':gm(ratios),'ratio_ci':ci(ratios,True)}
  d['comparisons'][refname]=comp
 for s in samples:
  r=N[s,'llm',p];h=N[s,'llm-astar',p]
  paired.append({'sample':s,'prompt':p,'v2_valid':refs['v2_original'][s,'llm',p]['valid'],'v3_valid':refs['v3_explicit_barriers'][s,'llm',p]['valid'],'eightway_valid':r['valid'],'v2_operation':refs['v2_original'][s,'llm-astar',p]['operation'],'v3_operation':refs['v3_explicit_barriers'][s,'llm-astar',p]['operation'],'eightway_operation':h['operation'],'v2_length':refs['v2_original'][s,'llm-astar',p]['length'],'v3_length':refs['v3_explicit_barriers'][s,'llm-astar',p]['length'],'eightway_length':h['length']})
 result['by_prompt'][p]=d
(OUT/'analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
with (OUT/'paired-comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
shutil.copy('/tmp/analyze_8way_only.py',OUT/'analyze.py')
print(json.dumps({'checks':checks,'by_prompt':result['by_prompt']},indent=2,ensure_ascii=False))
