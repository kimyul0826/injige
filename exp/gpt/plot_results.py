"""Static scientific comparison; cache/output paths stay in exp/gpt."""
import json,os
from pathlib import Path
OUT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(OUT/'.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
S=json.loads((OUT/'summary.json').read_text());D={(r['variant'],r['prompt'],r['split']):r for r in S}
chosen=json.loads((OUT/'selections.json').read_text())['variants']
names=list(dict.fromkeys(['reference','reference_grid_smooth','prune_octile_grid_smooth','combined','combined_half','optimal_llm_tie','optimal_deep_tie','bounded_0.1','focal_1.1',*chosen.values()]))
fig,axes=plt.subplots(1,2,figsize=(14,5.5),layout='constrained')
for ax,split in zip(axes,['all','evaluation']):
 for r in S:
  if r['prompt']=='pooled' and r['split']==split and 'anyangle' not in r['variant']:
   ax.scatter(r['operation'],r['length'],s=18,color='#c7c7c7',alpha=.8)
 for i,name in enumerate(names):
  r=D[name,'pooled',split];ax.scatter(r['operation'],r['length'],s=50,label=name)
 ax.axhline(100,color='black',lw=.6,ls='--');ax.axvline(100,color='black',lw=.6,ls='--')
 ax.set_xlabel('Operation / A* (geometric mean, %)');ax.set_ylabel('Path length / A* (geometric mean, %)')
 ax.set_title('All 100 maps' if split=='all' else 'Evaluation split: 20 previously observed maps')
 ax.set_ylim(99.8,108);ax.grid(alpha=.15)
axes[1].legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=8)
fig.suptitle('Cached Llama 3 8B responses: offline search trade-offs\nLower-left is better; grid movement only. Zoom: length <=108%; full results in CSV.')
fig.savefig(OUT/'tradeoffs.png',dpi=160);fig.savefig(OUT/'tradeoffs.pdf');plt.close(fig)
T=json.loads((OUT/'timing-summary.json').read_text())['results'];names=list(T)
fig,ax=plt.subplots(figsize=(9,4),layout='constrained');ax.barh(names,[T[n]['median_ms'] for n in names],color='#3568a5');ax.set_xlabel('Median offline processing time (ms), fresh cache');ax.set_title('100 maps × 3 prompts × 3 repeats; LLM generation excluded');ax.grid(axis='x',alpha=.2)
fig.savefig(OUT/'timing.png',dpi=160);fig.savefig(OUT/'timing.pdf');plt.close(fig)
print('Figures saved')
