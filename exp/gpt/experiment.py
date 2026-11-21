"""Offline experiments: no model imports/calls; all writes confined to this directory."""
import csv, hashlib, heapq, json, math, os, statistics, sys, time
from pathlib import Path
from collections import defaultdict
from shapely.geometry import Point
from reference_search import Environment, MOTIONS, path_result
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SOURCE=ROOT/'outputs/llama3-table1-upstream-v2/results.jsonl'
STYLES=('standard','cot','repe')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
def octile(a,b):
 x,y=abs(a[0]-b[0]),abs(a[1]-b[1]);return max(x,y)+(math.sqrt(2)-1)*min(x,y)
class CachedEnvironment(Environment):
 def __init__(self,q):
  super().__init__(q);self.cache={};self.checks=0
 def collision(self,a,b):
  self.checks+=1;k=(a,b) if a<=b else (b,a)
  if k not in self.cache:self.cache[k]=super().collision(a,b)
  return self.cache[k]
def prepare(env,raw,c):
 targets=env.targets(raw,'upstream')
 if c.get('filter'):
  valid=[]
  for p in raw:
   if not env.inside(p) or not all(float(x).is_integer() for x in p):continue
   if any(Point(p).intersects(w) for w in env._segments if w.length):continue
   if p in (env.start,env.goal):continue
   if c['filter']=='unique' and p in valid:continue
   if not valid or valid[-1]!=p:valid.append(p)
  targets=[env.start,*valid,env.goal]
 if c.get('prune'):
  out=[targets[0]];i=0
  while i<len(targets)-1:
   j=next((j for j in range(len(targets)-1,i,-1) if not env.collision(targets[i],targets[j])),i+1)
   out.append(targets[j]);i=j
  targets=out
 return targets

def run(env,raw,c):
 t0=time.perf_counter();checks0=env.checks
 targets=prepare(env,raw,c);index=1;weight=c.get('weight',1.0)
 if c.get('fallback') and len(targets)==2:weight=0.0
 if c.get('confidence'):
  ratio=sum(math.dist(a,b) for a,b in zip(targets,targets[1:]))/max(math.dist(env.start,env.goal),1e-9)
  weight/=max(1,ratio)
 if c.get('reject_detour') and sum(math.dist(a,b) for a,b in zip(targets,targets[1:]))>c['reject_detour']*math.dist(env.start,env.goal):weight=0
 hdist=octile if c.get('octile') else math.dist
 g={env.start:0.0};parent={env.start:env.start};closed=set();expanded=set()
 def guidance(p):
  if c.get('corridor'):
   return min(math.dist(p,a)+math.dist(p,b)-math.dist(a,b) for a,b in zip(targets,targets[1:]))
  return math.dist(p,targets[index])
 def priority(p):
  h=hdist(p,env.goal)
  d=weight*guidance(p)
  if c.get('cap') is not None:d=min(d,c['cap']*h)
  f=g[p]+(h+d)
  tie=-g[p] if c.get('tie')=='deep' else guidance(p) if c.get('tie')=='guide' else 0
  return f,tie,p
 queue=[priority(env.start)];pops=expansions=updates=rebuilds=0;best=math.inf;stale=0;found=False
 def rebuild():
  nonlocal queue,rebuilds
  rebuilds+=len(queue);queue=[priority(p) for _,_,p in queue];heapq.heapify(queue)
 while queue:
  _,_,s=heapq.heappop(queue);pops+=1
  if c.get('dedup') and s in closed:continue
  closed.add(s);expanded.add(s);expansions+=1
  if s==env.goal:found=True;break
  changed=False
  if c.get('advance')=='pop' and s==targets[index] and s!=env.goal:
   index=min(index+1,len(targets)-1);updates+=1;best=math.inf;stale=0;changed=True
  if c.get('goal_fallback') and targets[index]==env.goal and weight:
   weight=0;changed=True
  if c.get('visible_goal') and weight and not env.collision(s,env.goal):weight=0;changed=True
  if c.get('stall') and weight:
   dist=math.dist(s,targets[index])
   if dist<best-1e-9:best=dist;stale=0
   else:stale+=1
   if stale>=c['stall']:weight*=0.5;stale=0;changed=True
   if weight<0.0625:weight=0
  if c.get('decay') and expansions%c['decay']==0 and weight:weight*=0.5;changed=True
  if changed:rebuild()
  for dx,dy in MOTIONS:
   n=(s[0]+dx,s[1]+dy)
   if not c.get('advance') and n==targets[index] and n!=env.goal:
    index=min(index+1,len(targets)-1);updates+=1;rebuild()
   if n in closed and not c.get('reopen'):continue
   blocked=env.collision(s,n)
   if c.get('advance')=='valid_neighbor' and n==targets[index] and n!=env.goal and not blocked:
    index=min(index+1,len(targets)-1);updates+=1;rebuild()
   if not c.get('finite_storage') or not blocked:g.setdefault(n,math.inf)
   new=g[s]+(math.inf if blocked else math.dist(s,n))
   if new<g.get(n,math.inf):
    g[n]=new;parent[n]=s
    if c.get('reopen'):closed.discard(n)
    heapq.heappush(queue,priority(n))
 path=[]
 if found:
  path=[env.goal]
  while path[-1]!=env.start:
   path.append(parent[path[-1]])
   if len(path)>10000:raise RuntimeError('parent cycle')
  path.reverse()
 elapsed=time.perf_counter()-t0
 result=path_result(env,path,pops,len(g))
 result.update(unique_expanded=len(expanded),expansions=expansions,target_updates=updates,rebuild_entries=rebuilds,collision_queries=env.checks-checks0,seconds=elapsed,llm_output=targets)
 return result

VARIANTS={'reference':{},'astar_control':{'weight':0},'filter_exact':{'filter':'exact'},'filter_unique':{'filter':'unique'},'prune_visible_targets':{'prune':True},'advance_valid_neighbor':{'advance':'valid_neighbor'},'advance_on_pop':{'advance':'pop'},'no_targets_astar':{'fallback':True},'goal_target_astar':{'goal_fallback':True},'skip_closed_duplicates':{'dedup':True},'reopen':{'reopen':True},'finite_storage':{'finite_storage':True},'tie_deeper':{'tie':'deep'},'octile':{'octile':True},'confidence_weight':{'confidence':True},'visible_goal_astar':{'visible_goal':True},'corridor':{'corridor':True}}
for w in (.25,.5,.75):VARIANTS['weight_'+str(w)]={'weight':w}
for n in (16,32,64):VARIANTS['stall_'+str(n)]={'stall':n}
for n in (64,128):VARIANTS['decay_'+str(n)]={'decay':n}
for n in (1.5,2,3):VARIANTS['reject_detour_'+str(n)]={'reject_detour':n}
BASE={'filter':'unique','advance':'pop','fallback':True,'goal_fallback':True,'dedup':True,'reopen':True,'finite_storage':True,'octile':True}
VARIANTS['combined']=dict(BASE)
VARIANTS['combined_half']=dict(BASE,weight=.5)
VARIANTS['combined_stall']=dict(BASE,stall=32)
VARIANTS['combined_confidence']=dict(BASE,confidence=True)
VARIANTS['optimal_llm_tie']=dict(BASE,weight=0,tie='guide')
VARIANTS['optimal_deep_tie']=dict(BASE,weight=0,tie='deep')
for eps in (.1,.25,.5):VARIANTS['bounded_'+str(eps)]=dict(BASE,cap=eps)
for w in (.1,.25,.5):VARIANTS['weighted_astar_'+str(1+w)]={'corridor':False,'weight':w,'filter':'unique','advance':'pop','dedup':True,'reopen':True,'finite_storage':True,'octile':True,'force_goal':True}
# Weighted controls use Euclidean goal for extra weight, octile anchor; <=(1+w) bound.

def grid_shortcut(env,a,b):
 dx=b[0]-a[0];dy=b[1]-a[1];n=max(abs(dx),abs(dy))
 if n==0:return [a]
 p=[(a[0]+round(dx*i/n),a[1]+round(dy*i/n)) for i in range(n+1)]
 return p if all(not env.collision(x,y) for x,y in zip(p,p[1:])) else None

def smooth(env,r,anyangle=False):
 t0=time.perf_counter();path=[tuple(p) for p in r['path']];out=[path[0]];i=0;checks0=env.checks
 prefix=[0]
 for a,b in zip(path,path[1:]):prefix.append(prefix[-1]+math.dist(a,b))
 while i<len(path)-1:
  for j in range(len(path)-1,i,-1):
   if anyangle:p=[path[i],path[j]] if not env.collision(path[i],path[j]) else None
   else:
    if octile(path[i],path[j])>prefix[j]-prefix[i]+1e-9:continue
    p=grid_shortcut(env,path[i],path[j])
   if p is not None:break
  out.extend(p[1:]);i=j
 result=dict(r);result.update(path_result(env,out,r['operation'],r['storage']))
 result['seconds']+=time.perf_counter()-t0;result['collision_queries']+=env.checks-checks0
 return result

def load():
 rows=[json.loads(l) for l in SOURCE.read_text().splitlines()]
 b={r['sample']:r for r in rows if r['method']=='astar'}
 l={(r['sample'],r['prompt']):r for r in rows if r['method']=='llm'}
 a={(r['sample'],r['prompt']):r for r in rows if r['method']=='llm-astar'}
 queries={}
 for i,q in enumerate(json.loads((ROOT/'data/environment_50_30.json').read_text())):
  for j,pair in enumerate(q['start_goal']):queries[f"{q.get('id',i)}:{j}"]=dict(q,start=pair[0],goal=pair[1])
 return b,l,a,queries

def main():
 b,l,a,queries=load();variants=dict(VARIANTS)
 manifest={'source':str(SOURCE),'source_sha256':digest(SOURCE),'code_sha256':digest(Path(__file__)),'reference_sha256':digest(OUT/'reference_search.py'),'variants':variants,'inference_calls':0,'map_split':'map_id % 5 == 0: evaluation (20 maps); otherwise development (80 maps). Entire dataset was previously inspected: not a pristine held-out test.','timing':'warm collision memo shared within each query; variant order rotated; includes preprocessing/search, excludes final path validation; not production end-to-end','started':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'pid':os.getpid()}
 dump('manifest.json',manifest)
 errors=[];count=0;t0=time.perf_counter()
 with (OUT/'results.jsonl').open('w') as f:
  for qi,(sid,q) in enumerate(queries.items()):
   env=CachedEnvironment(q)
   # Prime only edges touched by the baseline; every variant shares this same memo.
   control=run(env,[],{'weight':0})
   for key in ('path','operation','storage','length','unique_expanded'):
    if control[key]!=b[sid][key] and not(key=='path' and [list(x) for x in control[key]]==b[sid][key]):errors.append([sid,'astar',key])
   for style in STYLES:
    raw=list(map(tuple,l[sid,style]['path']));local={}
    items=list(variants.items());shift=qi%len(items);items=items[shift:]+items[:shift]
    for name,c in items:
     r=run(env,[env.start,env.goal] if c.get('force_goal') else raw,c)
     if name=='reference':
      for key in ('path','operation','storage','length','unique_expanded','target_updates'):
       if r[key]!=a[sid,style][key] and not(key=='path' and [list(x) for x in r[key]]==a[sid,style][key]):errors.append([sid,style,key])
     local[name]=r
    for name in ('reference','combined','combined_half','optimal_llm_tie'):
     local[name+'_grid_smooth']=smooth(env,local[name])
    local['reference_anyangle']=smooth(env,local['reference'],True)
    for name,r in local.items():
     r=dict(r,sample=sid,prompt=style,variant=name,split='evaluation' if int(sid.split(':')[0])%5==0 else 'development')
     assert r['valid'],(sid,style,name)
     if not name.endswith('anyangle'):assert all(max(abs(x[0]-y[0]),abs(x[1]-y[1]))==1 for x,y in zip(r['path'],r['path'][1:])),(sid,name)
     f.write(json.dumps(r,separators=(',',':'))+'\n');count+=1
   if qi%10==0:
    f.flush();progress={'queries':qi+1,'total':len(queries),'rows':count,'elapsed_seconds':time.perf_counter()-t0,'baseline_errors':len(errors)};dump('progress.json',progress);print(json.dumps(progress),flush=True)
 manifest.update(completed=time.strftime('%Y-%m-%dT%H:%M:%S%z'),elapsed_seconds=time.perf_counter()-t0,rows=count,baseline_errors=errors)
 dump('manifest.json',manifest);dump('verification.json',{'baseline_comparisons':4000,'errors':errors,'all_valid':True,'grid_adjacency_checked':True,'inference_calls':0,'source_unchanged':digest(SOURCE)==manifest['source_sha256']})
 assert not errors,errors[:10]
 print('COMPLETED',count,flush=True)
if __name__=='__main__':main()
