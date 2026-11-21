"""Summaries, map-cluster uncertainty, all variants preserved."""
import collections,csv,json,math,random,statistics
from experiment import OUT,load,dump,digest

def gm(xs):return 100*math.exp(statistics.mean(math.log(x) for x in xs))
def quant(xs,p):
 xs=sorted(xs);return xs[min(len(xs)-1,int((len(xs)-1)*p))]
def main():
 b,l,a,q=load();groups=collections.defaultdict(list);allrows=[];violations=[]
 for filename in ('results.jsonl','supplement-results.jsonl','budget-results.jsonl','refinement-results.jsonl'):
  with (OUT/filename).open() as f:
   for line in f:
    r=json.loads(line);r.pop('path',None);r.pop('llm_output',None);base=b[r['sample']]
    for k in ('operation','storage','length'):r[k+'_ratio']=r[k]/base[k] if r[k] is not None else None
    if not r['variant'].endswith('anyangle') and r['length_ratio']<1-1e-9:violations.append([r['sample'],r['variant'],'below_optimal',r['length_ratio']])
    if r['variant'].startswith('optimal_') and abs(r['length_ratio']-1)>1e-9:violations.append([r['sample'],r['variant'],'optimality',r['length_ratio']])
    if r['variant'].startswith('bounded_'):
     bound=1+float(r['variant'].split('_')[1])
     if r['length_ratio']>bound+1e-9:violations.append([r['sample'],r['variant'],'bound',r['length_ratio']])
    if r['variant'].startswith('focal_') and r['length_ratio']>float(r['variant'][6:])+1e-9:violations.append([r['sample'],r['variant'],'bound',r['length_ratio']])
    r['expansion_ratio']=r['expansions']/base['unique_expanded']
    allrows.append(r)
    for split in ('all',r['split']):
     groups[r['variant'],r['prompt'],split].append(r)
     if r['prompt']!='ensemble':groups[r['variant'],'pooled',split].append(r)
 def summary(rs):
  return dict(n=len(rs),success=100*sum(r['valid'] for r in rs)/len(rs),operation=gm([r['operation_ratio'] for r in rs]),storage=gm([r['storage_ratio'] for r in rs]),length=gm([r['length_ratio'] for r in rs]),expansion=gm([r['expansion_ratio'] for r in rs]),op_worse=sum(r['operation_ratio']>1+1e-9 for r in rs),length_worse=sum(r['length_ratio']>1+1e-9 for r in rs),length_over10=sum(r['length_ratio']>1.1+1e-9 for r in rs),max_op=max(r['operation_ratio'] for r in rs),max_length=max(r['length_ratio'] for r in rs),p95_length=quant([r['length_ratio'] for r in rs],.95),median_ms=1000*statistics.median(r['seconds'] for r in rs),rebuild_entries=sum(r['rebuild_entries'] for r in rs),selection_scans=sum(r.get('selection_scans',0) for r in rs))
 assert len(allrows)==179000,('incomplete results',len(allrows))
 assert len({(r['variant'],r['sample'],r['prompt']) for r in allrows})==len(allrows),'duplicate results'
 summaries=[dict(variant=v,prompt=p,split=s,**summary(rs)) for (v,p,s),rs in groups.items()]
 with (OUT/'summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
 dump('summary.json',summaries)
 # Selections use ONLY development summaries; exclude cross-prompt and any-angle.
 candidates=[r for r in summaries if r['prompt']=='pooled' and r['split']=='development' and 'anyangle' not in r['variant'] and r['variant'] not in ('reference','astar_control','prune_visible_targets_grid_smooth','prune_octile','prune_octile_grid_smooth')]
 selections={}
 for name,criteria in [('efficiency',lambda r:r['length']<=103 and r['success']==100),('bounded_quality',lambda r:r['max_length']<=1.1+1e-9 and r['success']==100),('optimal_observed',lambda r:r['max_length']<=1+1e-9 and r['success']==100)]:
  eligible=[r for r in candidates if criteria(r)]
  selections[name]=min(eligible,key=lambda r:r['operation'])['variant']
 llm_candidates=[r for r in candidates if not r['variant'].startswith('weighted_astar_') and r['variant'] not in ('optimal_deep_tie','astar_grid_smooth')]
 selections['llm_efficiency']=min([r for r in llm_candidates if r['length']<=103 and r['success']==100],key=lambda r:r['operation'])['variant']
 selections['llm_bounded_quality']=min([r for r in llm_candidates if r['max_length']<=1.1+1e-9 and r['success']==100],key=lambda r:r['operation'])['variant']
 dump('selections.json',{'rule':'development only; minimize operation under stated constraints','efficiency_constraint':'GM length<=103%; success100%','bounded_quality_constraint':'all development path lengths <=110% A*','optimal_observed_constraint':'all development path lengths <=100% A*','variants':selections,'caution':'previous full-dataset analysis means evaluation split is not pristine held-out'})
 # Paired map-cluster bootstrap on evaluation maps, pooled prompts; descriptive intervals, no p-values.
 rng=random.Random(20261117);intervals={};reference={(r['sample'],r['prompt']):r for r in groups['reference','pooled','evaluation']}
 for variant in sorted(set(selections.values())):
  rs=groups[variant,'pooled','evaluation'];values=collections.defaultdict(lambda:collections.defaultdict(list))
  for r in rs:
   ref=reference[r['sample'],r['prompt']]
   for key in ('operation','length'):values[int(r['sample'].split(':')[0])][key].append(math.log(r[key]/ref[key]))
  ids=sorted(values);logs={k:[statistics.mean(values[i][k]) for i in ids] for k in ('operation','length')};out={}
  for k in logs:
   samples=[math.exp(statistics.mean(logs[k][rng.randrange(len(ids))] for _ in ids))*100 for _ in range(5000)]
   out[k]={'ratio_to_llmastar':math.exp(statistics.mean(logs[k]))*100,'map_bootstrap_95pct':[quant(samples,.025),quant(samples,.975)],'maps':len(ids),'resamples':5000}
  intervals[variant]=out
 dump('uncertainty.json',intervals)
 dump('quality-verification.json',dict(rows=len(allrows),variants=len(set(r['variant'] for r in allrows)),errors=violations,all_successful=all(r['valid'] for r in allrows),result_sha256=digest(OUT/'results.jsonl'),supplement_sha256=digest(OUT/'supplement-results.jsonl'),budget_sha256=digest(OUT/'budget-results.jsonl'),refinement_sha256=digest(OUT/'refinement-results.jsonl')))
 top=sorted([r for r in summaries if r['prompt']=='pooled' and r['split']=='all'],key=lambda r:r['operation'])
 print('SELECTIONS',json.dumps(selections));print('TOP',json.dumps(top[:15],indent=2));print('QUALITY_ERRORS',violations[:10])
 assert not violations
if __name__=='__main__':main()
