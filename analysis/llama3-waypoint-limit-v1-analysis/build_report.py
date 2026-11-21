from pathlib import Path
import json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/home/skkai/Mobility/2026_2/ykim/assign/outputs/llama3-waypoint-limit-v1-analysis');a=json.loads((P/'analysis.json').read_text());old={(r['method'],r['prompt']):r for r in a['old_summary']};new={(r['method'],r['prompt']):r for r in a['new_summary']};pr=['standard','cot','repe']
lines=['# 原본 프롬프트 대비 경유점 상한 실험','', '## Material Passport','', '- Origin Mode: validate','- Origin Date: 2026-11-21 (Asia/Seoul)','- Verification Status: ANALYZED','- Version Label: waypoint_limit_v1','', '## 결론','', '경유점 수의 상한 준수는 크게 좋아졌지만 전체 성능이 함께 좋아지지는 않았다. LLM 단독 Standard·CoT 성공률은 하락했다. RePE의 LLM+A*는 탐색량과 경로 길이가 개선되는 관측을 보였으나, Weighted A*(w=2)보다 두 지표 모두 나쁘다. 기존 실행과 PyTorch 버전이 달라 이 차이를 프롬프트의 순수 인과 효과로 분리할 수 없다.','', '## 입력·검산','', '- 신규: llama3-waypoint-limit-v1.zip. 원본 기준: llama3-table1-upstream-v2. 같은 100개 지도 × 시작·도착 10쌍, 프롬프트별 1,000문제.','- 완료: 3,000/3,000 생성, 7,000개 결과 행. ZIP 무결성·중복 여부 확인. 파싱 오류·출력 한도 종료 모두 0건.','- KST 2026-11-21 04:27:51~06:49:30, 약 2시간 21분 39초.','- 실제 저장된 프롬프트 3,000개에서 상한 문장 하나 외에는 원본과 완전히 같음을 확인. 예제는 그대로 유지됐다.','- 상한 N은 가로·세로 배리어 항목 수의 합. 시작·도착을 제외한 중간 경유점만 최대 N개. 코드로 경로를 잘라내지 않았다.','- 실제 응답 3,000개를 다시 파싱하고, 7,000개 경로의 유효성·유효 길이와 요약표를 재검산하여 일치 확인. A* 기준 경로·지표도 원본과 일치.','- 추가 모델 추론 없음. 프롬프트·검색 소스 해시 모두 실행 manifest와 일치.','', '## LLM 단독','', '| 방식 | 원본 성공률 | 상한 추가 성공률 | 변화 | 원본 성공→실패 | 원본 실패→성공 |','|---|---:|---:|---:|---:|---:|']
for p in pr:
 x,y=old['llm',p],new['llm',p];d=a['by_prompt'][p];t=d['llm_transitions'];lines.append(f"| {p} | {x['valid_path_ratio']:.1f}% | {y['valid_path_ratio']:.1f}% | {y['valid_path_ratio']-x['valid_path_ratio']:+.1f}%p | {t['valid_to_invalid']} | {t['invalid_to_valid']} |")
lines+=['','| 방식 | 성공률 차이 95% 구간 | 전체 성공 길이 원본→신규 | 공통 성공 수 | 공통 성공 길이 원본→신규 |','|---|---:|---:|---:|---:|']
for p in pr:
 d=a['by_prompt'][p];lo,hi=d['llm_success_difference_pp_ci'];lines.append(f"| {p} | [{lo:+.1f}, {hi:+.1f}]%p | {old['llm',p]['relative_path_length']:.2f}% → {new['llm',p]['relative_path_length']:.2f}% | {d['common_success']} | {d['common_success_length_old_pct']:.2f}% → {d['common_success_length_new_pct']:.2f}% |")
lines+=['','길이는 A* 대비 비율의 기하평균이다. LLM 단독 길이는 성공한 경로만 집계하며 성공 집합이 바뀐다. 특히 Standard 신규 길이 101.25%만 보면 좋아 보이지만 성공률은 7.2%로 하락했다. 공통 성공 집합도 별도로 제시하여 성공 표본 선택의 영향을 구분했다.','', '## LLM+A*','', '원본·상한 추가 모두 세 방식 성공률 100%. 탐색량·저장량·길이는 A*=100% 대비 기하평균이며 낮을수록 좋다.','', '| 방식 | 탐색량 원본→신규 | 저장량 원본→신규 | 길이 원본→신규 |','|---|---:|---:|---:|']
for p in pr:
 x,y=old['llm-astar',p],new['llm-astar',p];lines.append(f"| {p} | {x['operation_ratio']:.2f}% → {y['operation_ratio']:.2f}% | {x['storage_ratio']:.2f}% → {y['storage_ratio']:.2f}% | {x['relative_path_length']:.2f}% → {y['relative_path_length']:.2f}% |")
lines+=['','| 방식 | 원본 대비 탐색량 변화 | 변화 95% 구간 | A* 2배 초과 문제 원본→신규 | 길이 10% 초과 문제 원본→신규 |','|---|---:|---:|---:|---:|']
for p in pr:
 d=a['by_prompt'][p];c=d['hybrid_changes']['operation'];lo,hi=c['ratio_ci'];lines.append(f"| {p} | {(c['new_over_old_geomean']-1)*100:+.2f}% | [{(lo-1)*100:+.2f}, {(hi-1)*100:+.2f}]% | {d['old_hybrid_tail']['operation_over_2x']} → {d['new_hybrid_tail']['operation_over_2x']} | {d['old_hybrid_tail']['length_over_10pct']} → {d['new_hybrid_tail']['length_over_10pct']} |")
w=a['weighted_astar_2'];lines+=['',f"LLM 없는 Weighted A*(w=2)의 동일 집계 기준은 탐색량 {w['operation']:.2f}%, 저장량 {w['storage']:.2f}%, 길이 {w['length']:.2f}%다. 신규 세 방식보다 집계 탐색량·저장량·길이가 모두 낮다. 개별 문제에서 항상 더 좋다는 뜻은 아니다.",'', '## 상한을 실제로 지켰는가','', '아래 초과 수는 전체 1,000개 파싱 경로의 max(좌표 수−2, 0)를 이용한 개수 점검이다. 시작·도착이 잘못된 출력은 별도로 표시했다. 엄밀한 준수율은 시작·도착이 올바른 출력만 분모로 사용한다.','', '| 방식 | 상한 초과 원본→신규 | 신규 시작·도착 오류 | 신규 정상 끝점에서 상한 준수 | 끝점과 상한 모두 충족 / 전체 |','|---|---:|---:|---:|---:|']
for p in pr:
 d=a['by_prompt'][p];x,y=d['old_waypoints'],d['new_waypoints'];lines.append(f"| {p} | {x['exceeds_cap_len_minus_two_all']} → {y['exceeds_cap_len_minus_two_all']} | {1000-y['endpoints_correct']} | {y['within_cap_rate_valid_endpoints']:.2f}% ({y['within_cap_and_correct_endpoints']}/{y['endpoints_correct']}) | {y['within_cap_and_correct_endpoints']}/1000 |")
lines+=['','| 방식 | 경유점 수 중앙값 원본→신규 | 평균 원본→신규 | 개수가 늘어난 문제 | 개수가 줄어든 문제 |','|---|---:|---:|---:|---:|']
for p in pr:
 d=a['by_prompt'][p];x,y=d['old_waypoints'],d['new_waypoints'];lines.append(f"| {p} | {x['median_len_minus_two_all']:.0f} → {y['median_len_minus_two_all']:.0f} | {x['mean_len_minus_two_all']:.3f} → {y['mean_len_minus_two_all']:.3f} | {d['waypoints_increased']} | {d['waypoints_decreased']} |")
lines+=['','상한을 추가했다고 모든 응답의 경유점 수가 줄지는 않았다. Standard·CoT의 중앙값은 2→3으로 늘었고, 각각 579·651문제에서 점 수가 늘었다. Standard 평균은 극단적으로 긴 출력이 줄어 5.703→2.712로 낮아졌다. 원본 최대 중간점 수는 1,016, 신규는 67이다. 평균과 중앙값을 함께 보아야 한다. 이 출력 변화가 어떤 내부 추론 때문에 발생했는지는 확인할 수 없다.','', '원본 RePE는 이미 정상 끝점 출력 991건 중 990건이 상한 이내였다. 따라서 RePE에서 관측한 성능 변화 전체를 단순히 초과 경유점을 제거한 효과라고 볼 수 없다. 새 지시가 생성 경로와 추론 내용을 함께 바꾼 효과 및 실행 환경 차이가 섞일 수 있다. 모델이 응답 끝에서 주장하는 경유점 수는 신뢰하지 않고 실제 파싱 좌표를 세었다.','', '## 충돌 오류','', '원인별 건수는 중복될 수 있다.','', '| 방식 | 벽 교차·접촉 원본→신규 | 벽 위 경유점 원본→신규 | 경계·범위 밖 경유점 원본→신규 |','|---|---:|---:|---:|']
for p in pr:
 d=a['by_prompt'][p];x,y=d['old_failures'],d['new_failures'];lines.append(f"| {p} | {x.get('barrier_intersection',0)} → {y.get('barrier_intersection',0)} | {x.get('waypoint_on_barrier',0)} → {y.get('waypoint_on_barrier',0)} | {x.get('boundary_or_outside_point',0)} → {y.get('boundary_or_outside_point',0)} |")
lines+=['','예: sample 0:0의 신규 Standard는 `[5,5]→[12,12]→[18,18]→[20,20]`을 출력했다. 중간점은 2개로 상한 3개를 지켰으나, 첫 선분이 y=10, x=0~25의 벽을 통과한다. 개수 준수는 충돌 없는 경로를 의미하지 않는다.','', '## 비용·비교 제한','', '- 신규 GPU 로그: RTX 5090, PyTorch 2.11.0+cu128. 원본 V2 환경은 PyTorch 2.10.0+cu130. 최근 8way 실험은 2.14.1+cu132였다. 프롬프트만 변경한 동일 환경 대조군은 없다.','- 모델 ID·revision·BF16·최대 8,000토큰·greedy 설정은 동일하다. 검색·충돌·기존 프롬프트 템플릿 코드도 같다. 가중치 전체 바이트 해시까지 검증하지 않았다.','- 지도 단위(지도당 10문제) 짝지은 부트스트랩 20,000회, seed=20261121. 95% 구간은 탐색적이고 다중 비교 보정은 하지 않았다. 반복 추론 변동을 추정한 구간도 아니다.','- 세 방식의 신규 생성 시간 중앙값: Standard 1.758초, CoT 4.116초, RePE 9.500초. 하드웨어·소프트웨어가 달라 원본 대비 시간 차이를 프롬프트 효과로 해석하지 않는다.','', '## 해석 점검','', '통계·방법론 오류 유형 11/11 점검. 전체 성공률과 공통 성공 집합의 길이를 함께 보고 생존자 편향을 구분했다. 지도 단위 군집 구조를 반영했다. 집계 수치로 모든 지도에서의 개선을 주장하지 않으며 Simpson 역설을 배제한 전수 방향 검증은 하지 않았다. 다중 지표·분석 선택은 탐색적이고, 인과 추론은 실행 환경 차이로 제한된다. 생태학적 오류·Berkson/collider 편향·기저율 무시·평균 회귀·역인과에 해당하는 추가 설계는 확인되지 않았다.','', '현재 결론은 저장 결과의 ANALYZED 상태다. 모델 추론을 재실행하여 완전 재현성을 VERIFIED로 판정한 것은 아니다.','']
(P/'report.md').write_text('\n'.join(lines).replace('原본','원본'))
fig,axes=plt.subplots(2,2,figsize=(12,8));x=np.arange(3);w=.34;colors=['#71869c','#087e83']
for ax,method,metric,title in [(axes[0,0],'llm','valid_path_ratio','LLM-only success (%)'),(axes[0,1],'llm-astar','operation_ratio','LLM+A* operations (A*=100%)'),(axes[1,0],'llm-astar','relative_path_length','LLM+A* length (A*=100%)')]:
 for j,(s,label) in enumerate([(old,'Original'),(new,'Waypoint cap')]):
  v=[s[method,p][metric] for p in pr];bars=ax.bar(x+(j-.5)*w,v,w,label=label,color=colors[j]);ax.bar_label(bars,fmt='%.2f',padding=(3+j*13 if metric=='relative_path_length' else 3),fontsize=9)
 ax.set_title(title);ax.set_xticks(x,['Standard','CoT','RePE']);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
 if metric=='operation_ratio':ax.axhline(a['weighted_astar_2']['operation'],color='#b35b32',ls='--',label='Weighted A* w=2');ax.set_ylim(0,80)
 elif metric=='relative_path_length':ax.axhline(a['weighted_astar_2']['length'],color='#b35b32',ls='--',label='Weighted A* w=2');ax.set_ylim(99,105)
 else:ax.set_ylim(0,17)
 ax.legend(frameon=False,fontsize=8)
ax=axes[1,1]
for j,(key,label) in enumerate([('old_waypoints','Original'),('new_waypoints','Waypoint cap')]):
 vals=[a['by_prompt'][p][key]['exceeds_cap_len_minus_two_all'] for p in pr];bars=ax.bar(x+(j-.5)*w,vals,w,label=label,color=colors[j]);ax.bar_label(bars,padding=3)
ax.set_title('Outputs exceeding cap (per 1,000)');ax.set_xticks(x,['Standard','CoT','RePE']);ax.set_ylim(0,145);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(frameon=False,fontsize=8)
fig.suptitle('Original vs waypoint cap | Llama 3 8B Instruct',fontsize=16);fig.tight_layout(rect=(0,.07,1,.95));fig.text(.5,.018,'Same 1,000 queries per prompt. Operations/length are geometric means.\nGeneration PyTorch versions differ; changes do not isolate the prompt effect.',ha='center',fontsize=10,color='#52606d');fig.savefig(P/'comparison.png',dpi=170);fig.savefig(P/'comparison.pdf');plt.close(fig)
shutil.copy('/tmp/report_waypoint_limit.py',P/'build_report.py');print(P/'report.md')
