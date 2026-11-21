from pathlib import Path
import sys,json,csv,shutil,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
from matplotlib.collections import LineCollection
P=Path('/home/skkai/Mobility/2026_2/ykim/assign');sys.path.insert(0,str(P))
from llm_astar.search import intersects
OUT=P/'outputs/llama3-original-vs-8way-maps100';cases=json.loads((OUT/'traces.json').read_text());analysis=json.loads((P/'outputs/llama3-8way-only-v1-analysis/analysis.json').read_text())
for name in ['hybrid-png','llm-only-png']:(OUT/name).mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
pr=['standard','cot','repe'];versions=['original','eightway'];colors={'original':'#3977b3','eightway':'#098079'};labels={'original':'Original prompt','eightway':'8way-only prompt'};index=[]
legend=[Line2D([0],[0],color=colors['original'],lw=2,label='Original'),Line2D([0],[0],color=colors['eightway'],lw=2,label='8way-only'),Line2D([0],[0],color='#8a939e',lw=1,ls='--',label='A* reference'),Patch(facecolor='#8da7b1',alpha=.25,label='Expanded cells'),Line2D([0],[0],marker='D',color='none',markerfacecolor='#9550a5',label='Filtered target'),Line2D([0],[0],marker='o',color='none',markerfacecolor='#298642',label='Start'),Line2D([0],[0],marker='*',color='none',markerfacecolor='#c33748',markersize=9,label='Goal')]
rawlegend=[legend[0],legend[1],legend[2],Line2D([0],[0],color='#d23f40',lw=2,label='Barrier/boundary contact'),legend[-2],legend[-1]]

def figure(c,raw=False,note=None):
 fig,axs=plt.subplots(3,2,figsize=(12.2,13.1));q=c['query'];base=c['panels']['astar:-'];xmin,xmax=q['range_x'][0],q['range_x'][1]-1;ymin,ymax=q['range_y'][0],q['range_y'][1]-1
 walls=[((lo,y),(hi,y)) for y,lo,hi in q.get('horizontal_barriers',[])]+[((x,lo),(x,hi)) for x,lo,hi in q.get('vertical_barriers',[])]
 boundary=[((xmin,ymin),(xmax,ymin)),((xmin,ymax),(xmax,ymax)),((xmin,ymin),(xmin,ymax)),((xmax,ymin),(xmax,ymax))]
 for row,prompt in enumerate(pr):
  for col,ver in enumerate(versions):
   ax=axs[row,col];d=c['panels'][ver+':'+prompt];color=colors[ver]
   if not raw:
    cells=np.zeros((ymax-ymin+1,xmax-xmin+1,4))
    for x,y in d['_expanded']:cells[y-ymin,x-xmin]=to_rgba(color,.19)
    ax.imshow(cells,origin='lower',extent=(xmin-.5,xmax+.5,ymin-.5,ymax+.5),interpolation='nearest',zorder=0)
   for u,v in walls+boundary:ax.plot([u[0],v[0]],[u[1],v[1]],color='#30363c',lw=3 if (u,v) in walls else 1,zorder=2)
   bp=np.array(base['path']);ax.plot(bp[:,0],bp[:,1],color='#8a939e',lw=1,ls='--',alpha=.8,zorder=3)
   path=d['raw_llm_path'] if raw else d['path'];arr=np.asarray(path)
   if len(arr):
    if raw:
     segs=list(zip(path,path[1:]));bad=[a!=b and any(intersects(a,b,u,v) for u,v in walls+boundary) for a,b in segs]
     ax.add_collection(LineCollection(segs,colors=['#d23f40' if b else color for b in bad],linewidths=2,zorder=5))
     ax.scatter(arr[:,0],arr[:,1],s=20,color=color,edgecolors='white',linewidths=.4,zorder=6)
    else:
     ax.plot(arr[:,0],arr[:,1],color='white',lw=3.7,zorder=4);ax.plot(arr[:,0],arr[:,1],color=color,lw=2,zorder=5)
     t=np.asarray(d.get('llm_output',[])[1:-1])
     if len(t):ax.scatter(t[:,0],t[:,1],s=22,marker='D',color='#9550a5',edgecolor='white',linewidths=.5,zorder=6)
   ax.scatter(*q['start'],marker='o',s=60,color='#298642',edgecolor='white',linewidth=.8,zorder=7);ax.scatter(*q['goal'],marker='*',s=120,color='#c33748',edgecolor='white',linewidth=.7,zorder=7)
   ax.set(xlim=(xmin-.5,xmax+.5),ylim=(ymin-.5,ymax+.5));ax.set_aspect('equal');ax.grid(alpha=.12,linewidth=.5);ax.set_axisbelow(True)
   validity='VALID' if d['raw_llm_valid'] else 'INVALID'
   stats=f"LLM-only: {validity} | output points: {len(path)}" if raw else f"Pops {d['operation']} ({100*d['operation']/base['operation']:.1f}% of A*); expanded {d['unique_expanded']}\nLength {d['length']:.2f} ({100*d['length']/base['length']:.1f}% of A*)"
   ax.set_title(labels[ver]+'\n'+stats,fontsize=10,pad=9)
   if col==0:ax.set_ylabel({'standard':'STANDARD','cot':'CoT','repe':'RePE'}[prompt]+'\ny',fontsize=12,labelpad=10)
   if row==2:ax.set_xlabel('x')
 kind='LLM-only output paths' if raw else 'LLM+A* paths and expanded search areas'
 fig.suptitle(f"Sample {c['sample']} | {kind}\nStart {q['start']} → Goal {q['goal']}",fontsize=16,y=.981)
 subtitle=note or ('Red segments intersect barriers/boundary. Stored validity is shown above each map.' if raw else 'Same query, search rules and A* reference. Light fill = actual unique expanded cells.')
 fig.text(.5,.927,subtitle,ha='center',fontsize=9.5,color='#52606d')
 fig.subplots_adjust(left=.09,right=.97,bottom=.095,top=.875,hspace=.40,wspace=.18)
 fig.legend(handles=rawlegend if raw else legend,loc='lower center',bbox_to_anchor=(.52,.037),ncol=4 if not raw else 3,frameon=False,fontsize=9)
 footer='Raw output has no search area. Out-of-map coordinates are clipped to the common map limits.' if raw else 'Purple diamonds are guidance targets, not the executed route. Pops may count a cell repeatedly.'
 fig.text(.5,.014,footer,ha='center',fontsize=9,color='#52606d');return fig

with PdfPages(OUT/'hybrid-search-maps100.pdf') as hybrid, PdfPages(OUT/'llm-only-maps100.pdf') as rawpdf:
 for i,c in enumerate(cases,1):
  stem=f'{i:03d}-sample-{c["sample"].replace(":","-")}'
  for raw,pdf,folder in [(False,hybrid,'hybrid-png'),(True,rawpdf,'llm-only-png')]:
   fig=figure(c,raw);fig.savefig(OUT/folder/f'{stem}.png',dpi=125,facecolor='white');pdf.savefig(fig);plt.close(fig)
  for p in pr:
   x,y=c['panels']['original:'+p],c['panels']['eightway:'+p]
   index.append({'page':i,'sample':c['sample'],'prompt':p,'original_llm_valid':x['raw_llm_valid'],'eightway_llm_valid':y['raw_llm_valid'],'original_operation':x['operation'],'eightway_operation':y['operation'],'original_expanded':x['unique_expanded'],'eightway_expanded':y['unique_expanded'],'original_length':x['length'],'eightway_length':y['length']})
  if i%10==0:print(f'Rendered {i}/100 map pairs (two PDFs)',flush=True)
S={v:{(r['method'],r['prompt']):r for r in analysis['summaries'][v]} for v in ['v2_original','eight_way_only']}
with PdfPages(OUT/'comparison-summary.pdf') as pdf:
 fig=plt.figure(figsize=(11.7,8.3));ax=fig.add_axes([.06,.08,.88,.84]);ax.axis('off');fig.suptitle('Original prompt vs 8way-only: results and maps',fontsize=19,y=.96)
 ax.text(0,.96,'Full experiment: 1,000 queries per prompt | 100 maps × 10 start/goal pairs',fontsize=12)
 ax.text(0,.90,'Only added sentence: You can move horizontally, vertically, and diagonally.',fontsize=11)
 ax.text(0,.81,'LLM-only valid paths',fontsize=14,weight='bold')
 data=[]
 for p in pr:
  x,y=S['v2_original']['llm',p],S['eight_way_only']['llm',p];data.append([p.title(),f"{x['valid_path_ratio']:.1f}%",f"{y['valid_path_ratio']:.1f}%",f"{y['valid_path_ratio']-x['valid_path_ratio']:+.1f} pp"])
 t=ax.table(cellText=data,colLabels=['Prompt','Original','8way-only','Change'],bbox=[0,.59,1,.18],cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(11)
 ax.text(0,.52,'LLM+A*: 100% valid paths in both experiments',fontsize=14,weight='bold')
 data=[]
 for p in pr:
  x,y=S['v2_original']['llm-astar',p],S['eight_way_only']['llm-astar',p];data.append([p.title(),f"{x['operation_ratio']:.2f}% → {y['operation_ratio']:.2f}%",f"{x['relative_path_length']:.2f}% → {y['relative_path_length']:.2f}%"])
 t=ax.table(cellText=data,colLabels=['Prompt','Operations: original → 8way','Length: original → 8way'],bbox=[0,.30,1,.18],cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(11)
 ax.text(0,.235,'Operations/length are geometric means relative to A*=100%; lower is better.',fontsize=10)
 ax.text(0,.18,'Finding: no broad improvement from the extra movement sentence.',fontsize=12,weight='bold')
 ax.text(0,.125,'Maps: same 100 selected queries as the previous gallery, one per map; selection ignores outcomes.',fontsize=9.5)
 ax.text(0,.075,'Original and 8way-only generation used different PyTorch versions; causal attribution is limited.',fontsize=9.5)
 ax.text(0,.025,'Following pages deliberately show the largest Standard operation improvement and regression among these 100.',fontsize=9)
 pdf.savefig(fig);plt.close(fig)
 ordered=sorted(cases,key=lambda c:c['panels']['eightway:standard']['operation']/c['panels']['original:standard']['operation'])
 for c,label in [(ordered[0],'Selected example: largest Standard operation reduction among these 100'),(ordered[-1],'Selected example: largest Standard operation increase among these 100')]:
  fig=figure(c,False,label);pdf.savefig(fig);plt.close(fig)
with (OUT/'map-index.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(index[0]));w.writeheader();w.writerows(index)
shutil.copy('/tmp/render_8way_maps.py',OUT/'render.py')
shutil.copy(P/'outputs/llama3-8way-only-v1-analysis/report.md',OUT/'analysis-report.md')
(OUT/'README.txt').write_text('''원본 프롬프트 대 8way-only 지도 비교\n\ncomparison-summary.pdf: 전체 1,000문제 성능 요약 + 선택한 100문제 중 Standard 탐색량 변화가 가장 큰 개선·악화 사례. 사례는 의도적으로 극단값을 골랐으므로 전체 경향을 대표하지 않습니다.\nhybrid-search-maps100.pdf: 100문제, 100페이지. 행은 Standard·CoT·RePE, 왼쪽은 원본, 오른쪽은 8way-only입니다. 연한 색은 실제 확장한 칸, 진한 선은 최종 경로, 회색 점선은 A* 기준, 보라색은 필터 후 경유점입니다.\nllm-only-maps100.pdf: 같은 100문제의 LLM 단독 경로. 빨간 선은 장애물·경계 교차 또는 접촉입니다. LLM 직접 출력에는 탐색 영역이 없습니다. 공통 지도 범위 밖 좌표는 그림에서 잘립니다.\nhybrid-png/ 및 llm-only-png/: 각 PDF의 개별 PNG 100개씩.\nmap-index.csv: 문제별 페이지와 수치 비교.\ntraces.json: 경로·확장 칸 원자료.\nmanifest.json: 선택·검증 기록. 앞서 선택한 100문제와 동일하며 지도마다 1문제, 각 시작·도착 인덱스는 10번 등장합니다. seed=20261120.\n\n8way-only의 300개 검색을 재실행하여 원본 경로·지표와 모두 일치함을 확인했습니다. 원본과 A*의 검증된 400개 검색 기록은 기존 자료에서 재사용했습니다. 추가 LLM 추론은 없습니다.\n\n주의: 원본 V2와 8way-only의 PyTorch 환경이 다르므로 순수 프롬프트 효과를 분리한 대조 실험은 아닙니다.\n''')
archive=OUT.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(OUT.rglob('*')):
  if f.is_file():z.write(f,str(Path(OUT.name)/f.relative_to(OUT)))
print('FINISHED',archive,flush=True)
