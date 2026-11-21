"""Post-hoc combinations motivated by first-stage results; exploratory only."""
import json,time
from experiment import OUT,STYLES,CachedEnvironment,run,smooth,load,dump

def main():
 b,l,a,q=load();count=0;t0=time.perf_counter()
 with (OUT/'refinement-results.jsonl').open('w') as f:
  for qi,(sid,query) in enumerate(q.items()):
   env=CachedEnvironment(query)
   for style in STYLES:
    raw=list(map(tuple,l[sid,style]['path']))
    plain=run(env,raw,{'prune':True});oct=run(env,raw,{'prune':True,'octile':True})
    for name,r in [('prune_visible_targets_grid_smooth',smooth(env,plain)),('prune_octile',oct),('prune_octile_grid_smooth',smooth(env,oct))]:
     r.update(variant=name,sample=sid,prompt=style,phase='posthoc_combination',split='evaluation' if int(sid.split(':')[0])%5==0 else 'development')
     assert r['valid'];assert all(max(abs(x[0]-y[0]),abs(x[1]-y[1]))==1 for x,y in zip(r['path'],r['path'][1:]))
     f.write(json.dumps(r,separators=(',',':'))+'\n');count+=1
   if qi%100==0:f.flush();print(qi+1,count,time.perf_counter()-t0,flush=True)
 dump('refinement-verification.json',dict(rows=count,all_valid=True,grid_adjacency_checked=True,inference_calls=0,seconds=time.perf_counter()-t0,phase='posthoc, first-stage full results already observed'))
 print('COMPLETED',count,flush=True)
if __name__=='__main__':main()
