from pathlib import Path
import json,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path('/home/skkai/Mobility/2026_2/ykim/assign/outputs/llama3-8way-only-v1-analysis');a=json.loads((P/'analysis.json').read_text());versions=['v2_original','v3_explicit_barriers','eight_way_only'];names=['V2 original','V3 explicit barriers','8way only'];pr=['standard','cot','repe'];S={v:{(r['method'],r['prompt']):r for r in a['summaries'][v]} for v in versions}
lines=['# 대각선 이동 문장만 추가한 실험 결과','', '## Material Passport','', '- Origin Mode: validate','- Origin Date: 2026-11-20 (Asia/Seoul)','- Verification Status: ANALYZED','- Version Label: eight_way_only_v1_analysis','', '## 결론','', '대각선 이동 문장 하나만 추가해도 원본 V2 대비 전반적인 개선은 관찰되지 않았다. LLM 단독 Standard 성공률은 원본 12.7%보다 낮은 10.9%이며, CoT·RePE는 원본과 거의 비슷하다. LLM+A*의 성공률은 모두 100%, 탐색량은 원본 대비 -0.07% / +0.46% / +0.82% 변화로 사실상 비슷하다. 이전 V3에서 나타난 Standard의 악화는 일부 회복됐다.','', '## 완료·검산','', '- 파일: outputs/llama3-8way-only-v1.zip. ZIP 무결성 통과. 3,000/3,000 생성, 7,000/7,000 결과 행 완료; 중복 행 없음.','- KST 2026-11-20 16:22:33 시작, 18:16:46 종료. 총 약 1시간 54분 13초.','- 실제 저장된 3,000개 프롬프트 모두 원본과 비교하여 `You can move horizontally, vertically, and diagonally.` 문장 하나만 추가됐음을 확인했다. 원본 배리어 좌표·설명·예제 응답은 동일하다.','- 저장된 요약표를 재계산하여 일치. 7,000개 행의 성공 판정과 유효 경로 길이, 3,000개 응답 파싱을 재검산하여 불일치 없음.','- 같은 지도 100개 × 시작·도착 10쌍 = 각 프롬프트 1,000문제. A* 기준 경로·탐색량·저장량·길이는 기존 두 실험과 모두 일치한다.','- 추가 LLM 추론 0회. 파싱 오류 0건. 토큰 한도 종료 Standard 2건, CoT 0건, RePE 5건. 한도 종료여도 파싱 및 경로 판정에 따라 원본 방식으로 포함했다.','', '## LLM 단독','', '| 방식 | V2 원본 성공률 | V3 표기 변경 성공률 | 8way-only 성공률 | 원본 대비 |','|---|---:|---:|---:|---:|']
for p in pr:
 vals=[S[v]['llm',p]['valid_path_ratio'] for v in versions];lines.append(f'| {p} | {vals[0]:.1f}% | {vals[1]:.1f}% | {vals[2]:.1f}% | {vals[2]-vals[0]:+.1f}%p |')
lines+=['','| 방식 | 원본 성공→실패 | 원본 실패→성공 | 공통 성공 | 성공률 차이 95% 구간 |','|---|---:|---:|---:|---:|']
for p,d in a['by_prompt'].items():
 c=d['comparisons']['v2_original'];t=c['llm_transitions'];lo,hi=c['llm_success_difference_pp_ci'];lines.append(f"| {p} | {t['valid_to_invalid']} | {t['invalid_to_valid']} | {c['common_success_count']} | [{lo:+.1f}, {hi:+.1f}]%p |")
lines+=['','Standardは 원본 대비 성공 18건이 순감소했다. 이전 V3 대비로는 24건 순증가했으며, 그 성공률 차이의 탐색적 95% 구간은 [+0.3, +4.5]%p다. CoT·RePE의 성공률 차이는 작다.','', '| 방식 | 원본 LLM 길이 | 8way-only LLM 길이 | 공통 성공에서 원본→8way-only 길이 |','|---|---:|---:|---:|']
for p,d in a['by_prompt'].items():
 c=d['comparisons']['v2_original'];lines.append(f"| {p} | {S['v2_original']['llm',p]['relative_path_length']:.2f}% | {S['eight_way_only']['llm',p]['relative_path_length']:.2f}% | {c['common_llm_length_old_pct']:.2f}% → {c['common_llm_length_new_pct']:.2f}% |")
lines+=['','LLM 길이는 성공한 문제만을 대상으로 한 A* 대비 비율 기하평균이다. 성공 집합이 바뀌므로 전체 평균만으로 동일 문제의 경로 개선을 판단하지 않는다. 공통 성공 집합에서도 큰 일관적 개선은 없다.','', '## LLM+A*','', '모든 실험·프롬프트에서 성공률 100%. 아래 탐색량·저장량·길이는 A*=100%에 대한 문제별 비율의 기하평균이다.','', '| 방식 | 탐색량 원본→V3→8way-only | 길이 원본→V3→8way-only | 8way-only 저장량 |','|---|---:|---:|---:|']
for p in pr:
 ops=[S[v]['llm-astar',p]['operation_ratio'] for v in versions];lens=[S[v]['llm-astar',p]['relative_path_length'] for v in versions]
 lines.append(f"| {p} | {' → '.join(f'{x:.2f}%' for x in ops)} | {' → '.join(f'{x:.2f}%' for x in lens)} | {S['eight_way_only']['llm-astar',p]['storage_ratio']:.2f}% |")
lines+=['','| 방식 | 원본 대비 탐색량 변화 | 탐색량 변화 95% 구간 | A*보다 탐색량 큰 문제 | 길이 10% 초과 문제 |','|---|---:|---:|---:|---:|']
for p,d in a['by_prompt'].items():
 m=d['comparisons']['v2_original']['hybrid_metrics']['operation'];lo,hi=m['ratio_ci'];lines.append(f"| {p} | {(m['new_over_reference_geomean']-1)*100:+.2f}% | [{(lo-1)*100:+.2f}, {(hi-1)*100:+.2f}]% | {d['hybrid_operation_worse_than_astar']}/1000 | {d['hybrid_length_over_10pct']}/1000 |")
lines+=['','Standard의 경로 길이는 원본 102.549%→102.390%로 약 0.156% 상대 감소했으나 실질 차이는 작다. 탐색적 95% 구간은 약 [-0.308%, -0.007%]이고 다중 비교 보정 없이 경계에 있으므로 확정적인 개선으로 강조하지 않는다.','', 'Standard의 45.51% 탐색량은 A* 대비 기하평균이다. 실제 탐색 횟수 합계의 비율은 66.18%이므로 전체 합계 감소는 33.82%다. 탐색 횟수 감소가 생성 시간을 포함한 전체 응답 시간 개선을 뜻하지는 않는다.','', '## 오류 유형과 해석','', '아래 오류 건수는 서로 겹칠 수 있다.','', '| 방식 | 벽 교차·접촉 | 벽 위 경유점 | 경계·범위 밖 경유점 | 잘못된 시작·도착 |','|---|---:|---:|---:|---:|']
for p,d in a['by_prompt'].items():
 f=d['failure_counts_overlapping'];lines.append(f"| {p} | {f.get('barrier_intersection',0)} | {f.get('waypoint_on_barrier',0)} | {f.get('boundary_or_outside_point',0)} | {f.get('wrong_endpoints',0)} |")
lines+=['','Standard의 벽 위 경유점은 원본 259건, V3 502건, 이번 257건이다. 원본 좌표 표기로 돌아오면서 이 오류 유형은 원본 수준으로 돌아왔다. 하지만 벽 교차·접촉은 여전히 887건으로 주된 실패 원인이다. RePE의 경계·범위 밖 경유점도 원본 315건과 비슷한 317건으로, 대각선 설명만으로 공간적 제약을 해결한 결과는 아니다.','', '가능한 해석: 대각선 이동을 몰라서 생기는 실패보다, 경유점 사이의 선분이 벽과 교차하는지 판단하는 어려움이 더 큰 병목일 수 있다. 이는 오류 양상에 근거한 가설이며 모델 내부 원인을 직접 증명한 것은 아니다.','', '## 후처리 민감도','', 'V2·V3·이번 결과에 같은 upstream 목표점 필터를 적용한 LLM 단독 성공률이다. 공식 결과와 구분한다.','', '| 방식 | V2 filtered | V3 filtered | 8way-only filtered |','|---|---:|---:|---:|']
for p,d in a['by_prompt'].items():
 prev=json.loads((P.parent/'llama3-diagonal-barriers-v3-analysis/analysis.json').read_text())['by_prompt'][p]
 lines.append(f"| {p} | {prev['old_llm_filtered_success']/10:.1f}% | {prev['new_llm_filtered_success']/10:.1f}% | {d['filtered_llm_success_count']/10:.1f}% |")
lines+=['','## 비용·비교 제한','', '| 방식 | 생성 시간 중앙값 | 검색 시간 중앙값 | 생성 토큰 중앙값 |','|---|---:|---:|---:|']
for p,d in a['by_prompt'].items():lines.append(f"| {p} | {d['median_generation_seconds']:.3f}초 | {d['median_search_seconds']:.4f}초 | {d['median_completion_tokens']:.1f} |")
lines+=['','- V3와 이번 실험은 로그상 RTX 5090, PyTorch 2.14.1+cu132, Transformers 4.57.3, BF16, 모델 revision, 메타데이터가 동일하다. 가중치 전체 바이트 일치와 모든 드라이버·커널 조건까지 검증한 것은 아니다.','- V2는 PyTorch 2.10.0+cu130이므로 원본 대비 변화에는 생성 환경 차이가 섞인다. 대각선 한 문장 자체의 인과 효과를 분리하려면 현재 환경에서 원본 대조군을 실행해야 한다.','- V3와 이번 실험 사이에는 배리어 표기와 `Barriers cannot be crossed.` 문장 유무가 함께 다르다. 따라서 V3 대비 회복을 표기 변경만의 효과로 확정할 수 없다.','- 100개 지도를 단위로 시작·도착 10쌍을 함께 재표집한 짝지은 부트스트랩 20,000회(seed=20261120). 구간은 탐색적 95%이며 다중 비교 보정 없음. 단일 생성 실행으로 반복 실행 변동을 추정할 수 없다.','', '## 해석 점검','', '11/11 통계 오류 유형을 점검했다. 지도별 집계와 전체 표본 성공률을 사용하여 집계 수준과 해석 수준을 맞췄고, 길이의 생존자 편향은 공통 성공 집합으로 보완했다. 여러 지표·분석 선택은 탐색적이다. 인과 주장은 생성 환경·다중 변경의 교란 때문에 제한된다. Simpson 역설을 배제하는 지도별 전수 방향 검증이나 보편적 개선 주장은 하지 않았다. Berkson·collider·기저율 무시·평균 회귀·역인과에 해당하는 추가 설계는 확인되지 않았다.','', '저장된 응답과 경로를 검산한 ANALYZED 상태이며 모델 추론 재현까지 VERIFIED로 판정하지 않는다.','']
(P/'report.md').write_text('\n'.join(lines).replace('Standardは','Standard는'))
fig,axes=plt.subplots(1,3,figsize=(15,4.8));colors=['#71869c','#d99745','#087e83'];x=np.arange(3);w=.25
for ax,(method,metric,title,ylim) in zip(axes,[('llm','valid_path_ratio','LLM-only success (higher is better)',(0,16)),('llm-astar','operation_ratio','LLM+A* operations (lower is better)',(0,85)),('llm-astar','relative_path_length','LLM+A* path length (lower is better)',(99,105))]):
 for j,v in enumerate(versions):
  vals=[S[v][method,p][metric] for p in pr];bars=ax.bar(x+(j-1)*w,vals,w,label=names[j],color=colors[j]);ax.bar_label(bars,fmt='%.2f',padding=(3 if metric=='valid_path_ratio' else 3+j*13),fontsize=8)
 ax.set_title(title,fontsize=11,pad=14);ax.set_xticks(x,['Standard','CoT','RePE']);ax.set_ylim(*ylim);ax.set_ylabel('%');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.14);ax.set_axisbelow(True)
 if metric=='relative_path_length':ax.axhline(100,color='#a34c44',ls='--',lw=1)
axes[0].legend(frameon=False,fontsize=8,loc='upper right');fig.suptitle('Llama 3 8B Instruct | same 1,000 queries per prompt',fontsize=15,y=.98);fig.subplots_adjust(left=.05,right=.985,bottom=.21,top=.79,wspace=.30);fig.text(.5,.04,'Operations/length: geometric mean relative to A*=100%. Hybrid success: 100% in all runs.\nV2 generation software differs; V3 vs 8way-only also changes the barrier-crossing sentence.',ha='center',fontsize=10,color='#52606d');fig.savefig(P/'comparison.png',dpi=170);plt.close(fig)
shutil.copy('/tmp/report_8way_only.py',P/'build_report.py')
print(P/'report.md');print(P/'comparison.png')
