"""Limit LLM-guided work by a geometry-only budget, then restart ordinary A*."""
import inspect,json,math,time
import experiment as e
from reference_search import path_result
source=inspect.getsource(e.run)
assert source.count('while queue:')==1
source=source.replace('while queue:', 'while queue:\n  if pops >= c.get("budget", math.inf):break')
namespace=dict(vars(e));exec(compile(source,str(e.OUT/'budget.py')+':bounded_run','exec'),namespace)
limited=namespace['run']
def main():
 b,l,a,q=e.load();count=0;t0=time.perf_counter()
 with (e.OUT/'budget-results.jsonl').open('w') as f:
  for qi,(sid,query) in enumerate(q.items()):
   env=e.CachedEnvironment(query)
   for style in e.STYLES:
    raw=list(map(tuple,l[sid,style]['path']))
    for multiplier in (2,4,8):
     budget=max(32,math.ceil(multiplier*e.octile(env.start,env.goal)))
     first=limited(env,raw,{'budget':budget});r=dict(first);fallback=not first['valid']
     if fallback:
      r=e.run(env,[],{'weight':0})
      for key in ('operation','seconds','expansions','collision_queries','rebuild_entries'):r[key]+=first[key]
      r['storage']=max(r['storage'],first['storage']);r['unique_expanded']=None
     r.update(variant='budget_restart_'+str(multiplier),sample=sid,prompt=style,budget=budget,fallback=fallback,split='evaluation' if int(sid.split(':')[0])%5==0 else 'development')
     assert r['valid'];assert all(max(abs(x[0]-y[0]),abs(x[1]-y[1]))==1 for x,y in zip(r['path'],r['path'][1:]))
     f.write(json.dumps(r,separators=(',',':'))+'\n');count+=1
   if qi%100==0:f.flush();print(qi+1,count,time.perf_counter()-t0,flush=True)
 e.dump('budget-verification.json',dict(rows=count,all_valid=True,grid_adjacency_checked=True,inference_calls=0,seconds=time.perf_counter()-t0))
 print('COMPLETED',count,flush=True)
if __name__=='__main__':main()
