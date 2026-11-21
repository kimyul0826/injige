"""Reusable entry point for tested single-response variants; does not call an LLM."""
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE))
from benchmark import execute
from experiment import CachedEnvironment

def search(query,nodes=None,variant='prune_octile_grid_smooth'):
 """query: original map dict; nodes: existing parsed LLM path. Returns path/metrics."""
 requires_llm=not (variant.startswith('weighted_astar_') or variant in ('optimal_deep_tie','astar_grid_smooth'))
 if requires_llm and nodes is None:raise ValueError('This variant requires a saved parsed LLM path; no inference is performed.')
 env=CachedEnvironment(query)
 result=execute(env,[tuple(p) for p in (nodes or [])],variant)
 result.update(variant=variant,inference_calls=0,uses_llm_response=requires_llm)
 return result

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--query',required=True,help='JSON file containing one map with start/goal')
 parser.add_argument('--nodes',help='JSON file containing an already generated coordinate list')
 parser.add_argument('--variant',default='prune_octile_grid_smooth')
 args=parser.parse_args()
 print(json.dumps(search(json.loads(Path(args.query).read_text()),json.loads(Path(args.nodes).read_text()) if args.nodes else None,args.variant),ensure_ascii=False))
