"""Fresh-cache, rotated-order timing on 100 maps x first pair x 3 prompts x 3 repeats."""
import collections,json,math,random,statistics,time
from experiment import OUT,VARIANTS,BASE,CachedEnvironment,run,smooth,load,dump
from supplement import focal,projected,sequential

def execute(env,raw,name):
 if name.endswith('_grid_smooth'):
  base=name[:-12]
  return smooth(env,run(env,[],{'weight':0}) if base=='astar' else execute(env,raw,base))
 if name.startswith('focal_'):return focal(env,raw,float(name[6:]))
 if name.startswith('project_'):
  t=time.perf_counter();p=projected(env,raw,int(name[-1]));extra=time.perf_counter()-t
  r=run(env,p,BASE);r['seconds']+=extra;return r
 if name=='sequential_repair':return sequential(env,raw)
 c={'prune':True,'octile':True} if name=='prune_octile' else VARIANTS[name]
 return run(env,[env.start,env.goal] if c.get('force_goal') else raw,c)

def main():
 b,l,a,q=load();selected=json.loads((OUT/'selections.json').read_text())['variants'];names=sorted(set(['reference','astar_control','optimal_deep_tie','reference_grid_smooth','prune_visible_targets_grid_smooth','prune_octile_grid_smooth',*selected.values()]))
 expected={}
 for filename in ('results.jsonl','supplement-results.jsonl','budget-results.jsonl','refinement-results.jsonl'):
  with (OUT/filename).open() as f:
   for line in f:
    r=json.loads(line)
    if r['variant'] in names and r['sample'].endswith(':0'):expected[r['variant'],r['sample'],r['prompt']]=(r['operation'],r['length'],tuple(map(tuple,r['path'])))
 errors=[]
 rows=[];rng=random.Random(12345);cases=[(sid,style) for sid in q if sid.endswith(':0') for style in ('standard','cot','repe')];rng.shuffle(cases)
 t0=time.perf_counter()
 for rep in range(3):
  for i,(sid,style) in enumerate(cases):
   order=names[(i+rep)%len(names):]+names[:(i+rep)%len(names)]
   for name in order:
    t=time.perf_counter();env=CachedEnvironment(q[sid]);setup=time.perf_counter()-t
    r=execute(env,list(map(tuple,l[sid,style]['path'])),name)
    measured=time.perf_counter()-t
    actual=(r['operation'],r['length'],tuple(map(tuple,r['path'])))
    if actual!=expected[name,sid,style]:errors.append([name,sid,style,rep])
    rows.append(dict(sample=sid,prompt=style,variant=name,repeat=rep,seconds=measured,operation=r['operation'],length=r['length']))
   if i%100==0:print(json.dumps(dict(repeat=rep,cases=i,elapsed=time.perf_counter()-t0)),flush=True)
 with (OUT/'timing-results.jsonl').open('w') as f:
  for r in rows:f.write(json.dumps(r)+'\n')
 groups=collections.defaultdict(list)
 for r in rows:groups[r['variant']].append(r)
 ref={(r['sample'],r['prompt'],r['repeat']):r for r in groups['reference']}
 summary={}
 for name,rs in groups.items():
  ratio=[r['seconds']/ref[r['sample'],r['prompt'],r['repeat']]['seconds'] for r in rs]
  summary[name]=dict(n=len(rs),median_ms=statistics.median(r['seconds'] for r in rs)*1000,gm_time_to_reference=100*math.exp(statistics.mean(math.log(x) for x in ratio)))
 dump('timing-summary.json',{'protocol':'100 maps, first pair, 3 prompts, 3 repeats; fresh collision cache per call; rotated algorithm order; setup + preprocessing + search + smoothing + path validation; LLM generation and file I/O excluded; shared-server CPU timing is descriptive','results':summary})
 dump('repeat-verification.json',{'replayed':len(rows),'errors':errors,'paths_and_metrics_exact':not errors})
 assert not errors,errors[:5]
 print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
