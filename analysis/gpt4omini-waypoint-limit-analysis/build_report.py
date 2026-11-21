from pathlib import Path
import json,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path('/home/skkai/Mobility/2026_2/ykim/assign');OUT=P/'outputs/gpt4omini-waypoint-limit-analysis';A=json.loads((OUT/'analysis.json').read_text());O={(r['method'],r['prompt']):r for r in A['original_summary']};N={(r['method'],r['prompt']):r for r in A['gpt_summary']};pr=['standard','cot','repe']
fig,axs=plt.subplots(2,2,figsize=(12,9));plt.rcParams['pdf.fonttype']=42
spec=[('Direct valid paths (%)',[O['llm',p]['valid_path_ratio'] for p in pr],[N['llm',p]['valid_path_ratio'] for p in pr]),('LLM-A* operations (A*=100%)',[O['llm-astar',p]['operation_ratio'] for p in pr],[N['llm-astar',p]['operation_ratio'] for p in pr]),('Outputs exceeding obstacle-count cap*',[A['by_prompt'][p]['original_waypoints']['over_cap'] for p in pr],[A['by_prompt'][p]['capped_waypoints']['over_cap'] for p in pr]),('Total output tokens (thousands)',[A['by_prompt'][p]['original_completion_tokens']/1000 for p in pr],[A['by_prompt'][p]['completion_tokens']['total']/1000 for p in pr])]
for ax,(title,a,b) in zip(axs.flat,spec):
 for i,(vals,label,color) in enumerate([(a,'Original','#7186a0'),(b,'Waypoint cap','#008b80')]):
  bars=ax.bar(np.arange(3)+(i-.5)*.34,vals,.32,label=label,color=color)
  for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+max(a+b)*.02,f'{v:.1f}',ha='center',fontsize=9)
 ax.set_title(title);ax.set_xticks(range(3),['Standard','CoT','RePE']);ax.set_ylim(0,max(a+b)*1.28);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(fontsize=9)
fig.suptitle('GPT-4o mini: original vs obstacle-count waypoint cap',fontsize=16);fig.subplots_adjust(top=.9,bottom=.10,hspace=.3,wspace=.22);fig.text(.5,.035,'1,000 queries per style. *Cap counts use outputs with correct start and goal; see report for denominators.',ha='center',fontsize=9)
fig.savefig(OUT/'comparison.png',dpi=160);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
t=['# GPT-4o mini 障害物数 상한 실험 분석'.replace('障害物数','장애물 개수'),'','## Material Passport','','- Origin Mode: validate','- Origin Date: 2026-11-21','- Verification Status: ANALYZED','- Overall Confidence: CAUTION (단일 생성 실행, 탐색적 비교)','','## 검증','','3,000개 응답·7,000개 행 완전성, ZIP 무결성, summary 재계산, 입력 지도, A* 기준, 소스 해시가 일치했습니다. 7,000개 경로의 유효성·길이와 3,000개 프롬프트·파싱을 재검사했습니다. 모든 프롬프트가 원본에 장애물 개수만큼의 중간 경유점 상한 문장 하나만 추가된 형태임을 확인했습니다. 시작·목표점은 상한에서 제외합니다. 추가 모델 호출 없음.','','Batch 3,000개, 순차 재사용 0개. 생성부터 완료까지 385초(6분 25초). 이전 원본은 순차 81개+Batch 2,919개 혼합입니다. 모델은 같은 gpt-4o-mini-2024-07-18, temperature=0, max_tokens=8000입니다.','','## 성능 — 원본 → 상한','','연산·저장량·길이는 동일 A* 대비 문제별 비율의 기하평균(%). 연산은 중복 추출 포함이며 추론 지연·실제 벽시계 시간이 아닙니다. LLM-A*는 두 조건 모두 100% 유효 경로입니다.','','|방식|직접 성공률|연산/A*|저장량/A*|길이/A*|','|---|---:|---:|---:|---:|']
for p in pr:
 a,b=O['llm-astar',p],N['llm-astar',p];t.append(f"|{p}|{O['llm',p]['valid_path_ratio']:.1f}% → {N['llm',p]['valid_path_ratio']:.1f}%|{a['operation_ratio']:.2f} → {b['operation_ratio']:.2f}|{a['storage_ratio']:.2f} → {b['storage_ratio']:.2f}|{a['relative_path_length']:.2f} → {b['relative_path_length']:.2f}|")
t+=['','## 상한 준수와 출력 길이','','초과 개수는 시작·목표점이 정확한 응답에서 집계합니다. 평균·중앙값은 전체 응답에 max(len(parsed)−2,0)을 적용한 기술통계입니다.','','|방식|초과 개수/끝점 정상 응답 (원본 → 상한)|상한 준수율(새 실험)|평균 경유점|중앙값|10개 이상|출력 토큰 변화|','|---|---:|---:|---:|---:|---:|---:|']
for p in pr:
 d=A['by_prompt'][p];a,b=d['original_waypoints'],d['capped_waypoints'];tok=100*(d['completion_tokens']['total']/d['original_completion_tokens']-1)
 t.append(f"|{p}|{a['over_cap']}/{a['correct_endpoints']} → {b['over_cap']}/{b['correct_endpoints']}|{100*b['within_cap']/b['correct_endpoints']:.2f}%|{a['mean']:.2f} → {b['mean']:.2f}|{a['median']:g} → {b['median']:g}|{a['ge10']} → {b['ge10']}|{tok:+.2f}%|")
t+=['','Llama 상한 실험의 끝점 정상 응답 중 준수율은 Standard 99.80%, CoT 99.90%, RePE 100%였습니다. GPT가 이번 템플릿에서는 이 조건을 더 자주 위반했습니다. 모델 계열별 템플릿·환경이 달라 모델 자체의 일반적인 지시 준수 순위로 해석하지 않습니다.','','## 짝지은 변화와 불확실성','','100개 지도(각 10문제)를 단위로 cluster bootstrap 20,000회, seed=20261121, 탐색적 95% 구간. 다중비교 보정 없음; 재생성 변동성은 포함하지 않습니다.','','|방식|연산 상대 변화 [95% 구간]|직접 성공률 변화 pp [95% 구간]|','|---|---:|---:|']
for p in pr:
 d=A['by_prompt'][p];v=d['comparison_to_original']['operation'];lo,hi=v['ratio_ci'];sl,sh=d['llm_success_delta_pp_ci'];t.append(f"|{p}|{100*(v['capped_over_original_geomean']-1):+.2f}% [{100*(lo-1):+.2f}, {100*(hi-1):+.2f}]|{d['llm_success_delta_pp']:+.2f} [{sl:+.2f}, {sh:+.2f}]|")
t+=['','세 방식 모두 연산 변화 구간이 0을 포함합니다. 일관된 탐색 효율 개선을 뒷받침하지 않습니다. Standard 직접 성공률 감소는 탐색적 구간에서 0을 포함하지 않습니다.','','## 실패 원인','','|방식|장애물 교차 실패(새 실험)|출력 한도 도달 원본 → 상한|명시적 파싱 오류(새 실험)|','|---|---:|---:|---:|']
for p in pr:
 d=A['by_prompt'][p];t.append(f"|{p}|{d['failure_categories_overlapping'].get('barrier_intersection',0)}|{O['llm',p]['truncated']} → {N['llm',p]['truncated']}|{N['llm',p]['parse_failures']}|")
t+=['','출력 한도 도달은 37 → 16개로 줄었습니다. 하지만 직접 경로 성공률이 오르지 않았고 장애물 교차는 여전히 주된 실패입니다. 점 수를 줄이는 조건만으로 충돌 회피가 해결되지 않았습니다. 리스트 미발견 시 원본 파서가 대체 좌표를 반환하므로 명시적 parse error만으로 형식 성공을 판단하지 않습니다.','','## Weighted A* 및 Llama 비교','','Weighted A* w=2는 연산 34.32%, 저장량 56.23%, 길이 102.02%로 GPT 상한 세 방식보다 집계 지표가 모두 작습니다. 같은 원본 연산 집계·tie-break(dedupe=False,tie=node)의 기존 결과를 사용했습니다. 개별 문제 모두에서 우세하다는 의미는 아닙니다. Llama 상한 실험에서는 RePE 연산이 원본보다 약 6.73% 줄었지만, 이번 GPT RePE는 약 0.38% 감소에 그쳤습니다.','','## 해석','','긴 좌표 나열·출력량 억제에는 효과가 관찰됐지만 경로 성공률이나 탐색량의 일관된 개선은 없습니다. CoT·RePE는 평균 점 수가 감소하는 동시에 중앙값은 3→4개로 증가했습니다. 긴 꼬리가 줄었다고 전형적인 응답의 점 수도 줄었다고 해석해서는 안 됩니다. API 재호출을 통한 재현 검증은 하지 않았으며 한 번의 결과 비교입니다.','','## Fallacy Scan — 11/11','','|항목|적용|','|---|---|','|Simpson 역설|방식별 보고, 통합 평균으로 방향을 덮지 않음|','|생태학적 오류|집계 개선을 모든 문제의 개선으로 일반화하지 않음|','|Berkson 선택 편향|성능은 전체 1,000문제; 준수율은 끝점 정상 부분집합임을 명시|','|Collider 편향|끝점 정상 응답 준수율을 전체 성공률로 해석하지 않음|','|기저율 무시|초과 개수와 분모 함께 보고|','|평균 회귀|극단값 선택 없이 전체 동일 문제를 비교|','|생존자 편향|실패·절단 포함; 직접 성공 경로 길이는 주 비교에서 제외|','|다중 탐색|여러 비교 구간은 탐색적이고 보정 없음|','|분석 선택 자유도|원본 지표·기하평균·재표집 seed 기록; 사전등록 없음|','|인과 혼동|동일 모델·조건이나 별도 생성 실행 및 원본의 전송 방식 혼합이 있음|','|역인과|점 수와 성능의 인과 기전을 추정하지 않음|']
(OUT/'report.md').write_text('\n'.join(t)+'\n');shutil.copy('/tmp/report_gptcap.py',OUT/'build_report.py')
print('written',OUT)
