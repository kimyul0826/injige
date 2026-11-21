"""Additional offline families; uses frozen experiment engine, never model APIs."""
import heapq,json,math,time,sys
from experiment import OUT,BASE,STYLES,CachedEnvironment,prepare,run,smooth,load,dump,octile,digest
from reference_search import MOTIONS,path_result

def projected(env,raw,radius):
 result=[]
 for p in raw:
  p=(round(p[0]),round(p[1]))
  p=(max(env.rx[0]+1,min(env.rx[1]-2,p[0])),max(env.ry[0]+1,min(env.ry[1]-2,p[1])))
  choices=[(p[0]+dx,p[1]+dy) for dx in range(-radius,radius+1) for dy in range(-radius,radius+1)]
  choices=[x for x in choices if env.free(x)]
  if choices:result.append(min(choices,key=lambda x:(math.dist(x,p),math.dist(x,env.goal),x)))
 return result

def sequential(env,raw):
 t0=time.perf_counter();targets=prepare(env,raw,{'filter':'unique'});path=[env.start];ops=store=expansions=checks=rebuild=0;expanded=set()
 original=(env.start,env.goal);failed=False;failed_target=None
 try:
  for a,b in zip(targets,targets[1:]):
   env.start,env.goal=a,b
   r=run(env,[],dict(weight=0,octile=True,dedup=True,reopen=True,finite_storage=True,tie='deep'))
   ops+=r['operation'];store=max(store,r['storage']);expansions+=r['expansions'];checks+=r['collision_queries']
   if not r['valid']:failed=True;failed_target=b;break
   path.extend(r['path'][1:])
 finally:env.start,env.goal=original
 if failed:
  fallback=run(env,[],dict(weight=0,octile=True,dedup=True,reopen=True,finite_storage=True,tie='deep'))
  path=fallback['path'];ops+=fallback['operation'];store=max(store,fallback['storage']);expansions+=fallback['expansions'];checks+=fallback['collision_queries']
 result=path_result(env,path,ops,store)
 result.update(fallback=failed,failed_target=failed_target,expansions=expansions,unique_expanded=None,target_updates=len(targets)-2,rebuild_entries=0,collision_queries=checks,seconds=time.perf_counter()-t0,llm_output=targets)
 return result

def focal(env,raw,w):
 t0=time.perf_counter();checks0=env.checks;targets=prepare(env,raw,{'filter':'unique'});idx=1
 g={env.start:0.0};parent={env.start:env.start};opened={env.start};closed=set();expanded=set();ops=scan=updates=0
 while opened:
  lower=min(g[p]+octile(p,env.goal) for p in opened);scan+=len(opened)
  candidates=[p for p in opened if g[p]+octile(p,env.goal)<=w*lower+1e-10];scan+=len(opened)
  s=min(candidates,key=lambda p:(g[p]+octile(p,env.goal)+math.dist(p,targets[idx]),-g[p],p))
  opened.remove(s);closed.add(s);expanded.add(s);ops+=1
  if s==env.goal:break
  if s==targets[idx]:idx=min(idx+1,len(targets)-1);updates+=1
  for dx,dy in MOTIONS:
   n=(s[0]+dx,s[1]+dy)
   if env.collision(s,n):continue
   ng=g[s]+math.dist(s,n)
   if ng<g.get(n,math.inf):g[n]=ng;parent[n]=s;opened.add(n);closed.discard(n)
 path=[]
 if env.goal in closed:
  path=[env.goal]
  while path[-1]!=env.start:path.append(parent[path[-1]])
  path.reverse()
 result=path_result(env,path,ops,len(g))
 result.update(expansions=ops,unique_expanded=len(expanded),target_updates=updates,rebuild_entries=0,selection_scans=scan,collision_queries=env.checks-checks0,seconds=time.perf_counter()-t0,llm_output=targets)
 return result

def main():
 b,l,a,queries=load();t0=time.perf_counter();count=0
 # Resume only fully written queries, removing partial query rows.
 existing=[];counts={}
 if '--resume' in sys.argv and (OUT/'supplement-results.jsonl').exists():
  for line in (OUT/'supplement-results.jsonl').read_text().splitlines():
   r=json.loads(line);existing.append((r,line));counts[r['sample']]=counts.get(r['sample'],0)+1
 completed={sid for sid,n in counts.items() if n==26}
 with (OUT/'supplement-results.jsonl').open('w') as f:
  for r,line in existing:
   if r['sample'] in completed:f.write(line+'\n');count+=1
  for qi,(sid,q) in enumerate(queries.items()):
   if sid in completed:continue
   env=CachedEnvironment(q);raws={style:list(map(tuple,l[sid,style]['path'])) for style in STYLES}
   singles={style:run(env,raws[style],BASE) for style in STYLES}
   for style in STYLES:
    raw=raws[style];res={}
    for radius in (1,2):
     t=time.perf_counter();p=projected(env,raw,radius);extra=time.perf_counter()-t
     res['project_'+str(radius)]=run(env,p,BASE);res['project_'+str(radius)]['seconds']+=extra
    res['sequential_repair']=sequential(env,raw)
    for w in (1.05,1.1,1.25):res['focal_'+str(w)]=focal(env,raw,w)
    res['bounded_0.1_grid_smooth']=smooth(env,run(env,raw,dict(BASE,cap=.1)))
    res['astar_grid_smooth']=smooth(env,run(env,[],dict(weight=0)))
    for name,r in res.items():
     r.update(sample=sid,prompt=style,variant=name,split='evaluation' if int(sid.split(':')[0])%5==0 else 'development')
     assert r['valid'],(sid,style,name)
     assert all(max(abs(x[0]-y[0]),abs(x[1]-y[1]))==1 for x,y in zip(r['path'],r['path'][1:])),(sid,name)
     f.write(json.dumps(r,separators=(',',':'))+'\n');count+=1
   # Pre-search geometric selection uses all three saved responses, no baseline oracle.
   def score(style):
    p=prepare(env,raws[style],{'filter':'unique'})
    return sum(math.dist(x,y)*(2 if env.collision(x,y) else 1) for x,y in zip(p,p[1:]))
   t=time.perf_counter();chosen=min(STYLES,key=score);extra=time.perf_counter()-t
   r=dict(singles[chosen]);r['seconds']+=extra
   r.update(variant='ensemble_geometry',selected_prompt=chosen)
   portfolio=dict(min(singles.values(),key=lambda r:r['length']))
   portfolio.update(operation=sum(r['operation'] for r in singles.values()),expansions=sum(r['expansions'] for r in singles.values()),unique_expanded=None,storage=max(r['storage'] for r in singles.values()),seconds=sum(r['seconds'] for r in singles.values()),collision_queries=sum(r['collision_queries'] for r in singles.values()),rebuild_entries=sum(r['rebuild_entries'] for r in singles.values()),variant='ensemble_best_path')
   for r in (r,portfolio):
    r.update(sample=sid,prompt='ensemble',split='evaluation' if int(sid.split(':')[0])%5==0 else 'development',generation_calls_required=3,generation_seconds_required=sum(l[sid,s]['generation_seconds'] for s in STYLES))
    f.write(json.dumps(r,separators=(',',':'))+'\n');count+=1
   if qi%25==0:
    f.flush();print(json.dumps({'queries':qi+1,'rows':count,'seconds':time.perf_counter()-t0}),flush=True)
 dump('supplement-verification.json',dict(rows=count,all_valid=True,grid_adjacency_checked=True,inference_calls=0,elapsed_seconds=time.perf_counter()-t0))
 print('COMPLETED',count,flush=True)
if __name__=='__main__':main()
