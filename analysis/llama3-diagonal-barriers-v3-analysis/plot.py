from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('/home/skkai/Mobility/2026_2/ykim/assign/outputs/llama3-diagonal-barriers-v3-analysis');a=json.loads((p/'analysis.json').read_text());prompts=['standard','cot','repe'];labels=['Standard','CoT','RePE']
S={(r['method'],r['prompt']):r for r in a['old_summary']};T={(r['method'],r['prompt']):r for r in a['new_summary']}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(14,4.6));colors=['#71869c','#077f86'];x=np.arange(3);w=.34
for ax,(method,metric,title,ylim) in zip(axes,[('llm','valid_path_ratio','LLM-only success (higher is better)',(0,16)),('llm-astar','operation_ratio','LLM+A* operations (lower is better)',(0,85)),('llm-astar','relative_path_length','LLM+A* path length (lower is better)',(95,108))]):
 for j,(data,label) in enumerate([(S,'V2 original'),(T,'V3 explicit barriers')]):
  vals=[data[method,q][metric] for q in prompts];b=ax.bar(x+(j-.5)*w,vals,w,label=label,color=colors[j]);ax.bar_label(b,fmt='%.2f',padding=(3 if metric!='relative_path_length' or j==0 else 16),fontsize=9)
 ax.set_title(title,fontsize=11,pad=18);ax.set_xticks(x,labels);ax.set_ylim(*ylim);ax.set_ylabel('%');ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
 if metric!='valid_path_ratio':ax.axhline(100,color='#a54545',linestyle='--',linewidth=1)
axes[0].legend(frameon=False,fontsize=9,loc='upper right')
fig.suptitle('Same 1,000 queries per prompt | Llama 3 8B Instruct',fontsize=15,y=1.01)
fig.text(.5,.015,'Operations and path length: geometric mean relative to A* = 100%. Hybrid success: 100% in all conditions.\nGeneration hardware/software changed; this comparison does not isolate the prompt effect.',ha='center',fontsize=10,color='#52606d')
fig.subplots_adjust(left=.05,right=.99,bottom=.20,top=.80,wspace=.30);fig.savefig(p/'comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
cases=json.loads((p/'regression-cases.json').read_text());c=next(c for c in cases if c['sample']=='0:8' and c['prompt']=='standard');q=c['query'];fig,ax=plt.subplots(figsize=(9,5.4))
for y,lo,hi in q['horizontal_barriers']:ax.plot([lo,hi],[y,y],color='#343a40',lw=5,solid_capstyle='round')
for x,lo,hi in q['vertical_barriers']:ax.plot([x,x],[lo,hi],color='#343a40',lw=5,solid_capstyle='round')
for key,color,label in [('old_path','#077f86','V2: valid'),('new_path','#d45745','V3: collides with wall')]:
 arr=np.array(c[key]);ax.plot(arr[:,0],arr[:,1],'-o',color=color,label=label,lw=2)
ax.scatter(*q['start'],s=120,marker='s',color='#1a7340',zorder=5);ax.scatter(*q['goal'],s=160,marker='*',color='#9c5b00',zorder=5)
ax.annotate('V3 segment overlaps barrier x=25, y=10..22',xy=(25,17),xytext=(28,5),arrowprops={'arrowstyle':'->','color':'#d45745'},color='#d45745',fontsize=10)
ax.set(xlim=(0,50),ylim=(0,30),xlabel='x',ylabel='y',title='Standard regression example: sample 0:8');ax.set_aspect('equal');ax.legend(loc='upper left');ax.grid(alpha=.15);fig.tight_layout();fig.savefig(p/'regression-example.png',dpi=180);plt.close(fig)
print('Saved comparison.png and regression-example.png')
