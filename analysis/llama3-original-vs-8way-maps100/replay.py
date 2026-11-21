from pathlib import Path
import json,zipfile,sys,hashlib,shutil
P=Path('/home/skkai/Mobility/2026_2/ykim/assign');sys.path.insert(0,str(P))
OUT=P/'outputs/llama3-original-vs-8way-maps100';OUT.mkdir(exist_ok=True)
previous=P/'outputs/llama3-diagonal-barriers-v3-paths100';cases=json.loads((previous/'traces.json').read_text());source=(P/'llm_astar/search.py').read_text();needle='    return result\n\n\nclass AStar:';assert source.count(needle)==1
ns={};exec(compile(source.replace(needle,"    result['_expanded'] = sorted(closed)\n    return result\n\n\nclass AStar:"),'<expanded-cell recording>','exec'),ns)
with zipfile.ZipFile(P/'outputs/llama3-8way-only-v1.zip') as z:
 assert z.testzip() is None
 name=next(n for n in z.namelist() if n.endswith('/results.jsonl'));rows=[json.loads(l) for l in z.read(name).splitlines()]
N={(r['sample'],r['method'],r['prompt']):r for r in rows};count=0;output=[]
for c in cases:
 s=c['sample'];q=c['query'];panels={'astar:-':c['panels']['astar:-']}
 for prompt in ['standard','cot','repe']:
  panels['original:'+prompt]=c['panels']['v2:'+prompt]
  raw=N[s,'llm',prompt];stored=N[s,'llm-astar',prompt];r=ns['search'](q,[tuple(p) for p in raw['path']],'upstream')
  for k in ['valid','operation','storage','length','target_updates','unique_expanded']:assert r[k]==stored[k],(s,prompt,k)
  assert [list(p) for p in r['path']]==stored['path'];assert len(r['_expanded'])==r['unique_expanded']
  r['raw_llm_valid']=raw['valid'];r['raw_llm_path']=raw['path'];panels['eightway:'+prompt]=r;count+=1
 output.append({'sample':s,'query':q,'panels':panels})
 if len(output)%20==0:print('Verified',len(output),'/100 maps',flush=True)
(OUT/'traces.json').write_text(json.dumps(output))
manifest={'samples':[c['sample'] for c in output],'selection':json.loads((previous/'manifest.json').read_text())['selection'],'seed':20261120,'new_verified_searches':count,'original_and_baseline_verified_searches_reused':400,'all_new_paths_and_metrics_identical':True,'inference_calls':0,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'shading':'Unique expanded cells, not all discovered nodes','original':'llama3-table1-upstream-v2','eightway':'llama3-8way-only-v1.zip'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));shutil.copy('/tmp/replay_8way_maps.py',OUT/'replay.py');print(OUT,flush=True)
