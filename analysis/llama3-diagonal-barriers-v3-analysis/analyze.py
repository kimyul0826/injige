from pathlib import Path
import sys,json,zipfile,hashlib,math,collections,statistics
import numpy as np
P=Path('/home/skkai/Mobility/2026_2/ykim/assign');sys.path.insert(0,str(P))
from llm_astar.search import Environment,path_result
from llm_astar.models import parse_path
from llm_astar.__main__ import summarize,load_queries
OUT=P/'outputs/llama3-diagonal-barriers-v3-analysis';OUT.mkdir(exist_ok=True)
old=[json.loads(l) for l in (P/'outputs/llama3-table1-upstream-v2/results.jsonl').read_text().splitlines()]
new=[];manifests=[];checks={};responses={};statuses=[]
for name in ['llama3-diagonal-barriers-v3','llama3-diagonal-barriers-cot-repe-v3']:
 with zipfile.ZipFile(P/'outputs'/f'{name}.zip') as z:
  assert z.testzip() is None
  def read(fn):return json.loads(z.read(f'{name}/{fn}'))
  manifest=read('manifest.json');manifests.append(manifest);statuses.append(read('status.json'))
  rows=[json.loads(l) for l in z.read(f'{name}/results.jsonl').splitlines()]
  checks[name]={'rows':len(rows),'stored_summary_matches':summarize(rows)==read('summary.json'),'input_hash_matches':hashlib.sha256(z.read(f'{name}/input.json')).hexdigest()==manifest['input_sha256'],'unique_rows':len({(r['sample'],r['method'],r['prompt']) for r in rows})}
  for n in z.namelist():
   if '/responses/' in n: responses[Path(n).stem]=json.loads(z.read(n))
  new.extend(r for r in rows if r['method']!='astar' or name=='llama3-diagonal-barriers-v3')
queries=dict(load_queries(P/'data/environment_50_30.json'))
assert len(new)==7000 and len(old)==7000
index=lambda rows:{(r['sample'],r['method'],r['prompt']):r for r in rows}
N,O=index(new),index(old);samples=sorted(queries,key=lambda s:tuple(map(int,s.split(':'))))
base={r['sample']:r for r in new if r['method']=='astar'}
checks['baseline_equal']=all(all(base[s][k]==O[s,'astar','-'][k] for k in ['path','valid','operation','storage','length']) for s in samples)
checks['same_input_old_new']=json.loads((P/'outputs/llama3-table1-upstream-v2/manifest.json').read_text())['input_sha256']==manifests[0]['input_sha256']
checks['same_new_manifests']=all(manifests[0][k]==manifests[1][k] for k in ['config','source_sha256','model_identity','input_sha256'])
checks['local_source_matches_v3']={k:hashlib.sha256((P/k).read_bytes()).hexdigest()==v for k,v in manifests[0]['source_sha256'].items()}
checks['validity_mismatches']=[];checks['length_mismatches']=[];checks['parse_mismatches']=[]
for r in new:
 e=Environment(queries[r['sample']]); rr=path_result(e,[tuple(n) for n in r['path']])
 if rr['valid']!=r['valid']:checks['validity_mismatches'].append([r['sample'],r['method'],r['prompt']])
 if r['valid'] and not math.isclose(rr['length'],r['length'],abs_tol=1e-8):checks['length_mismatches'].append([r['sample'],r['method'],r['prompt']])
 if r['method']=='llm':
  parsed=parse_path(responses[r['cache_key']]['text'],mode='upstream')
  if [list(n) for n in parsed]!=r['path']:checks['parse_mismatches'].append([r['sample'],r['prompt']])

def gm(a):return math.exp(sum(math.log(x) for x in a)/len(a))
rng=np.random.default_rng(20261120);boot=rng.integers(0,100,size=(20000,100))
def ci_map(values,log=False):
 v=np.asarray(values,dtype=float).reshape(100,10)
 if log:v=np.log(v)
 means=v.mean(axis=1);b=means[boot].mean(axis=1)
 if log:b=np.exp(b)
 return [float(x) for x in np.quantile(b,[.025,.975])]
analysis={'checks':checks,'statuses':statuses,'new_summary':summarize(new),'old_summary':summarize(old),'by_prompt':{},'input_maps':100,'samples_per_prompt':1000,'bootstrap':{'unit':'map (10 start-goal pairs per map)','replicates':20000,'seed':20261120,'ci':'95%, exploratory; no multiplicity correction'}}
cases=[];paired=[]
for prompt in ['standard','cot','repe']:
 a=[O[s,'llm',prompt] for s in samples];b=[N[s,'llm',prompt] for s in samples]
 ha=[O[s,'llm-astar',prompt] for s in samples];hb=[N[s,'llm-astar',prompt] for s in samples]
 trans=collections.Counter(('valid' if x['valid'] else 'invalid')+'_to_'+('valid' if y['valid'] else 'invalid') for x,y in zip(a,b))
 common=[(x,y) for x,y in zip(a,b) if x['valid'] and y['valid']]
 failsets={}
 for label,rows in [('old',a),('new',b)]:
  c=collections.Counter()
  for r in rows:
   if r['valid']:continue
   e=Environment(queries[r['sample']]);path=[tuple(n) for n in r['path']]
   if not path:c['empty_path']+=1;continue
   if path[0]!=e.start or path[-1]!=e.goal:c['wrong_endpoints']+=1
   if any(not e.inside(n) for n in path):c['boundary_or_outside_point']+=1
   wall_hits=[]
   for u,v in zip(path,path[1:]):
    seg=e._line([u,v])
    wall_hits.extend(seg.intersection(e._line([j,k])).geom_type for j,k in e.walls if seg.intersects(e._line([j,k])))
   if wall_hits:c['barrier_intersection']+=1
   if any(not e.free(n) and e.inside(n) for n in path):c['waypoint_on_barrier']+=1
   if wall_hits and not any(not e.free(n) and e.inside(n) for n in path):c['barrier_hit_between_waypoints']+=1
  failsets[label]=dict(c)
 summary={'llm_success_old':sum(x['valid'] for x in a),'llm_success_new':sum(x['valid'] for x in b),'transitions':dict(trans),'llm_valid_difference_pp_ci':[v*100 for v in ci_map([int(y['valid'])-int(x['valid']) for x,y in zip(a,b)])],'common_valid':len(common),'common_valid_length_old_pct':100*gm([x['length']/base[x['sample']]['length'] for x,y in common]),'common_valid_length_new_pct':100*gm([y['length']/base[y['sample']]['length'] for x,y in common]),'failures_overlapping':failsets}
 for label,rows in [('old',ha),('new',hb)]:
  summary[label+'_hybrid']={'operation_worse_than_astar':sum(r['operation']>base[r['sample']]['operation'] for r in rows),'operation_over_2x':sum(r['operation']>2*base[r['sample']]['operation'] for r in rows),'length_over_10pct':sum(r['length']>1.1*base[r['sample']]['length'] for r in rows),'length_worse_than_astar':sum(r['length']>base[r['sample']]['length']+1e-8 for r in rows),'aggregate_operation_ratio':sum(r['operation'] for r in rows)/sum(base[r['sample']]['operation'] for r in rows),'median_generation_seconds':statistics.median(r['generation_seconds'] for r in rows),'median_search_seconds':statistics.median(r['search_seconds'] for r in rows),'median_astar_search_seconds':statistics.median(base[r['sample']]['search_seconds'] for r in rows),'median_completion_tokens':statistics.median(r['usage']['completion_tokens'] for r in rows),'completion_tokens_total':sum(r['usage']['completion_tokens'] for r in rows)}
 summary['hybrid_changes']={}
 for metric in ['operation','storage','length']:
  ratios=[y[metric]/x[metric] for x,y in zip(ha,hb)]
  summary['hybrid_changes'][metric]={'new_over_old_geomean':gm(ratios),'map_bootstrap_ci':ci_map(ratios,True),'new_better':sum(y[metric]<x[metric]-1e-8 for x,y in zip(ha,hb)),'new_worse':sum(y[metric]>x[metric]+1e-8 for x,y in zip(ha,hb))}
 for label,rows in [('old',a),('new',b)]:
  filtered=[path_result(Environment(queries[r['sample']]),Environment(queries[r['sample']]).targets([tuple(n) for n in r['path']],'upstream')) for r in rows]
  summary[label+'_llm_filtered_success']=sum(r['valid'] for r in filtered)
 analysis['by_prompt'][prompt]=summary
 for x,y,hx,hy in zip(a,b,ha,hb):
  paired.append({'sample':x['sample'],'prompt':prompt,'old_llm_valid':x['valid'],'new_llm_valid':y['valid'],'old_operation':hx['operation'],'new_operation':hy['operation'],'old_length':hx['length'],'new_length':hy['length']})
  if x['valid'] and not y['valid']:
   cases.append({'sample':x['sample'],'prompt':prompt,'query':queries[x['sample']],'old_path':x['path'],'new_path':y['path'],'new_response':responses[y['cache_key']]['text']})
(OUT/'analysis.json').write_text(json.dumps(analysis,indent=2,ensure_ascii=False))
(OUT/'regression-cases.json').write_text(json.dumps(cases,indent=2,ensure_ascii=False))
import csv
with (OUT/'paired-comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
print(json.dumps(analysis,ensure_ascii=False,indent=2))
