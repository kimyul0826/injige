import csv,json,math
from experiment import OUT,VARIANTS,dump
S=json.loads((OUT/'summary.json').read_text());D={(r['variant'],r['prompt'],r['split']):r for r in S}
SELECT=json.loads((OUT/'selections.json').read_text());chosen=SELECT['variants']
TIM=json.loads((OUT/'timing-summary.json').read_text())
U=json.loads((OUT/'uncertainty.json').read_text());V=json.loads((OUT/'quality-verification.json').read_text())

def table(names,prompt='pooled',split='all'):
 lines=['| 조건 | 탐색량 | 저장량 | 길이 | A*보다 탐색 증가 | 길이 >10% 증가 | 최악 길이 |','|---|---:|---:|---:|---:|---:|---:|']
 for name in names:
  r=D[name,prompt,split]
  label='astar_control (목표 가중치 0 진단군)' if name=='astar_control' else name
  lines.append(f"| {label} | {r['operation']:.2f} | {r['storage']:.2f} | {r['length']:.2f} | {r['op_worse']}/{r['n']} | {r['length_over10']}/{r['n']} | {100*r['max_length']:.2f}% |")
 return '\n'.join(lines)

lines=['# 추론 재실행 없는 LLM-A* 개선 실험 결과','', '## Material Passport',
'- 상태: CPU 오프라인 탐색 실험 완료. 모델 호출 0회. 모든 산출물은 `exp/gpt` 내부.',
'- 원자료: `outputs/llama3-table1-upstream-v2/results.jsonl` 및 `data/environment_50_30.json` 읽기 전용.',
'- Llama 3 8B Instruct의 기존 3,000개 응답을 그대로 사용. 새 프롬프트·새 샘플링·새 가중치 없음.',
f"- 조건 수 {V['variants']}개, 기록 {V['rows']:,}개. A*/기존 LLM-A* 재현 검증 4,000건. 기존 재현과 개선 실험을 구분한다.",
'- 100개 지도에 각각 10개 시작/목표. 프롬프트 3종은 같은 문제의 반복 조건이며 3,000개 독립 지도가 아니다.',
'', '## 읽는 방법',
'탐색량·저장량·길이는 해당 문제의 기존 A* 대비 비율의 기하평균 ×100이다. 100보다 작을수록 작다. 기존 Table 1의 operation은 큐 pop 횟수로 중복 꺼내기를 포함한다. Focal은 선택하여 꺼낸 노드 수이며 후보 집합 순회 비용은 별도이다. 모든 조건의 성공률은 아래 검증 결과를 따르고, 증가 횟수·최악값도 함께 본다.',
'각 조건의 탐색 횟수는 알고리즘의 실제 추가 탐색을 포함한다. 단, 경로 후처리는 탐색 횟수에 잡히지 않으므로 별도 시간 실험이 필요하다. 저장량은 g 사전 크기이며 메모리 바이트가 아니다. 구간 탐색·재시작·포트폴리오는 순차 실행의 최대 g 크기이며, 누적 연산은 합산한다. 저장 중인 경로·OPEN·충돌 메모 등 전체 메모리 사용량을 나타내지 않는다.',
'', '## 주요 결과',
'LLM 응답을 유지한 사후 조합 중 목표 생략+octile+격자 후처리는 평균 탐색량과 경로 길이를 모두 줄였다. 하지만 최악 경로 악화가 남아 모든 문제에 대한 개선은 아니다. 비LLM 가중 A* 대조군도 강한 성능을 보였으며, LLM의 기여와 기본 탐색기 변경 효과를 분리해야 한다. 단순히 여러 수정을 합친 combined는 탐색량이 크게 늘었다.',
'', '## 주요 비교',
table(['reference','weighted_astar_1.5','weighted_astar_1.1','reference_grid_smooth','prune_visible_targets','prune_visible_targets_grid_smooth','prune_octile_grid_smooth','combined','combined_half','combined_stall','optimal_llm_tie','optimal_deep_tie','bounded_0.1','focal_1.1','budget_restart_4','astar_control']),
'', '## 개발 구간에서 선택한 후보',
'지도 ID %5==0인 20개 지도(600개 프롬프트별 사례)는 평가 구간, 나머지 80개 지도는 개발 구간으로 정했다. 후보는 개발 구간에서만 선택했다. 다만 앞선 한계점 분석에서 전체 데이터를 이미 보았으므로 평가 구간을 완전히 미관측한 외부 검증이라고 부르지 않는다.',
f"- 탐색량 우선(개발 구간 평균 길이 ≤103%, 성공률100%): `{chosen['efficiency']}`.",
f"- 경로 악화 제한(개발 구간 모든 길이 ≤110%): `{chosen['bounded_quality']}`.",
f"- 관찰된 최단 경로 유지(개발 구간 모든 길이 ≤100%): `{chosen['optimal_observed']}`.",
'', f"- LLM 사용 후보: 탐색량 우선 `{chosen['llm_efficiency']}`, 경로 악화 제한 `{chosen['llm_bounded_quality']}`. 비LLM 후보와 구분한다.",
'평가 구간만의 결과:',table(list(dict.fromkeys(['reference',*chosen.values()])),'pooled','evaluation'),
'', '동일 평가 지도 내 기존 LLM-A* 대비 비율의 지도 단위 bootstrap 95% 구간(5,000회, 20개 지도):']
for name,u in U.items():
 for key,x in u.items():lines.append(f"- {name}, {key}: {x['ratio_to_llmastar']:.2f}% [{x['map_bootstrap_95pct'][0]:.2f}, {x['map_bootstrap_95pct'][1]:.2f}].")
lines += ['구간은 탐색적 기술 통계이다. 여러 후보를 시험한 뒤 선택한 불확실성 전체를 반영하지 않으며 유의성·신규 지도 일반화를 주장하지 않는다.','','## 실제 CPU 처리 시간 비교',TIM['protocol'],
'100개 지도의 첫 시작/목표 쌍 × 프롬프트 3종 × 3회. 각 호출마다 환경과 충돌 메모를 새로 만들고 알고리즘 순서를 회전했다. 기존 저장 로그의 시간과 직접 비교하지 않았다. 공유 서버 부하와 첫 시작/목표만 선택한 표본의 한계가 있다.',
'| 조건 | 중앙 처리 시간(ms) | 기존 LLM-A* 대비 시간 기하평균 |','|---|---:|---:|']
for name,r in TIM['results'].items():lines.append(f"| {name} | {r['median_ms']:.3f} | {r['gm_time_to_reference']:.2f}% |")
lines += ['', '시간 표의 astar_control도 위의 목표 가중치 0 진단군이며 순수 A* 시간으로 해석하지 않는다.',
'이 시간에는 환경 구성·정제·탐색·후처리·경로 검증을 포함하고 파일 입출력과 LLM 생성은 제외한다. LLM을 계속 호출하는 배포에서는 긴 생성 시간이 그대로 남는다. LLM을 쓰지 않는 A* 대조군은 새 문제에서 생성 자체를 생략할 수 있다.',
'', '## 단독 변경과 조합의 해석',
'- `astar_control`이라는 내부 이름은 순수 A*가 아니라 LLM 목표 가중치를 0으로 둔 진단 조건이다. 목표 목록 갱신과 OPEN 재구축이 남아 pop 수가 기준 A*와 다르다. 표의 정규화 분모는 이 조건이 아니라 기존의 실제 A* 1,000개 결과이며, 별도 재생 1,000건도 정확히 일치했다. 비LLM 대조군은 weighted_astar와 optimal_deep_tie로 해석한다.',
'- `filter_exact`: 원본 응답에서 실제 벽/경계·정수 격자 검사와 연속 중복 제거. 원본 필터의 여분 경계 여백도 없앤다. `filter_unique`는 전역 중복까지 제거한다. 시작/목표는 새로 보장한다.',
'- `advance_valid_neighbor`: 충돌 없는 이웃에서 목표를 갱신한다. `advance_on_pop`: 목표 노드가 실제 꺼내져 확장될 때 갱신한다. 목표 도달 의미가 명확해져도 탐색 효율 증가를 보장하지 않는다.',
'- `no_targets_astar`, `goal_target_astar`: 중간 목표가 없거나 현재 목표가 최종 목적지가 된 경우 추가 목표 거리를 제거한다. 원본의 g+2h 효과가 없어지는 대신 탐색량이 늘 수 있다.',
'- `weight_*`, `confidence_weight`, `reject_detour_*`: 목표 거리의 가중치를 줄이거나 목표 연결선 길이/시작-목표 직선 거리에 따라 조절한다. 연결선은 충돌 가능하며 품질의 확실한 척도가 아니다. detour 문턱으로 목표를 끄는 방식은 필요한 우회도 버릴 수 있다.',
'- `stall_*`, `decay_*`: 목표와의 최선 거리 갱신이 N번의 확장 동안 없거나 일정 확장 수에 도달하면 가중치를 반감한다. 가중치 변경 시 OPEN을 재정렬한다. 최선 거리 기준은 전역적이며 실제 이동 진척을 뜻하지 않는다.',
'- `visible_goal_astar`: 확장 노드에서 목표까지 직선 충돌이 없으면 이후 목표 유도를 끈다. 경로를 직선으로 바꾸지 않으며 최단성을 보장하지 않는다.',
'- `prune_visible_targets`: 뒤의 목표까지 직선으로 갈 수 있으면 앞의 중간 목표를 생략. `corridor`: 목표 연결선의 양 끝까지 거리 합에서 선분 길이를 뺀 값을 추가 유도로 사용한다.',
'- `skip_closed_duplicates`, `reopen`, `finite_storage`, `tie_deeper`, `octile`: 큐 처리·더 좋은 g 발견 시 재확장·무한대 g 저장 생략·동점에서 큰 g 우선·8방향 격자 하한 휴리스틱을 각각 변경한다. 저장량 감소 일부는 단순히 무한대 항목을 세지 않는 효과다.',
'- `combined`: 정밀/중복 필터 + 확장 시 목표 갱신 + 중간 목표 부재/최종 목표에서 A* + 중복 확장 생략 + 재확장 + 유한 g 저장 + octile. `_half`, `_stall`, `_confidence`는 각각 추가 가중치 변경이다. 조합의 성능 저하 역시 그대로 보고한다.',
'- `optimal_llm_tie`: 추가 목표 비용은 0, 동점에서 LLM 목표 가까운 노드 우선. `optimal_deep_tie`: LLM 없이 동점에서 큰 g 우선. 일관적인 octile A*와 재확장에 기반한 최단 경로 조건이다. 이 대조군으로 LLM 자체의 추가 기여를 구분한다.',
'- `bounded_*`: 추가 목표 비용을 ε × octile(현재,최종목표) 이하로 제한하고 재확장을 허용한다. 이 구현의 일반적인 양의 비용 격자 탐색 조건에서 (1+ε) 길이 상한을 의도한 설계이며, 모든 실제 결과도 기존 A*와 비교 검사했다.',
'- `weighted_astar_*`: LLM 목표 대신 최종 목표만 사용. g+octile+w×Euclidean 목표 거리 대조군이다. 이름의 수치는 1+w이며 정확히 (1+w)×octile로 구현한 것은 아니다.',
'- `focal_*`: OPEN의 최소 g+octile의 w배 안에 드는 노드 중 LLM 유도 점수가 가장 작은 노드를 선택한다. 재확장을 허용한다. 후보 집합을 매번 순회하는 단순 구현이므로 낮은 선택 횟수와 빠른 실행은 다르다.',
'- `project_1/2`: 범위 밖 좌표를 내부로 자르고 반경1/2 안의 가장 가까운 유효 격자로 보정한 뒤 combined. 동점은 최종 목표 거리와 좌표 순서로 결정한다.',
'- `sequential_repair`: 정제된 중간 목표 사이를 각각 최적 A*로 연결한다. 각 구간은 짧아도 전체 우회가 클 수 있다. 지도89의 5개 프롬프트별 사례에서 중간 목표에 도달할 수 없는 실제 실패를 확인했고, 이런 경우 원래 시작/목표 A*로 복귀하도록 보강했다. 실패 구간과 복귀 연산도 모두 합산한다. sequential-failure.json에 경위를 남겼으며 일회성 초기 코드/오류 로그는 배포 정리 때 제거했다.',
'- `budget_restart_2/4/8`: max(32,ceil(k×시작-목표 octile))회 pop까지만 원본 LLM-A*로 탐색하고, 미완료면 처음부터 일반 A*를 실행한다. 실패한 선행 탐색 비용도 합산한다. 남은 최악 비용이 0인 공짜 복귀가 아니다.',
'- `_grid_smooth`: 기존 경로의 뒤쪽 지점과 디지털 직선을 연결해 충돌 없는 8방향 격자 경로로 대체한다. 원래 부분 경로보다 길어지지 않는 경우만 채택한다. BFS/A* 재탐색은 없지만 충돌 검사 비용이 든다.',
'- `reference_anyangle`: 직선 가시성으로 경로를 단축한다. 8방향 격자 이동 제약을 풀기 때문에 주 비교에서 제외한다. 길이가 A*보다 짧아도 같은 문제의 최단성을 깬 것이 아니다.',
'', '## 관찰 후 추가한 조합',
'첫 단계 결과를 확인한 뒤 목표 생략+격자 후처리, 목표 생략+octile, 목표 생략+octile+후처리의 세 조건을 추가했다. `refinement.py`에 분리했고 후보 선택 풀에서는 제외했다. 전체 결과를 보고 만든 사후 조합이므로 평가 구간 결과도 확증 실험으로 해석하지 않는다.',
'', '## 세 응답을 함께 사용하는 조건',
table(['ensemble_geometry','ensemble_best_path'],'ensemble'),
'`ensemble_geometry`는 정제한 연결선 길이에 충돌 구간 2배 페널티를 주어 응답을 고르고 combined 탐색을 한 번 수행하는 조건이다. 실험 하네스에서 계산을 공유하지만 보고 비용은 선택 점수 계산+선택된 탐색이다. `ensemble_best_path`는 세 combined 탐색을 모두 수행해 가장 짧은 결과를 고른다. 탐색 비용은 세 번 모두 합산하고 g 저장량은 순차 최대값이다. 두 방식 모두 새 문제에서는 LLM 생성 3회가 필요하다. 단일 프롬프트와 동일한 생성 비용으로 비교할 수 없다.',
'', '## 프롬프트별 주요 조건']
for style in ('standard','cot','repe'):
 lines += ['',f'### {style}',table(list(dict.fromkeys(['reference','reference_grid_smooth',*chosen.values(),'optimal_deep_tie'])),style)]
lines += ['', '## 전체 조건 — 3프롬프트 통합',table(sorted({r['variant'] for r in S if r['prompt']=='pooled'})),
'', '## 검증과 제한',
f"- 원본 재현 4,000건 일치. 모든 {V['rows']:,}개 기록에서 유효 경로 확인. any-angle 외에는 모든 인접 이동이 8방향 한 칸인지 검사.",
f"- 최단/허용 길이/격자 최단보다 짧은 결과 검사 오류: {len(V['errors'])}건. 검증 파일에 원본·실험 결과 SHA256을 기록.",
'- 전체 조건과 프롬프트별/개발/평가 수치는 summary.csv 및 summary.json에 보존. 각 경로와 계산량은 4개 results JSONL에 보존.',
'- 최종 후보 10종 × 300사례 × 3회 = 9,000건의 추가 재생에서 경로/탐색량/길이가 모두 정확히 일치했다(repeat-verification.json).',
'- 원본 추론 응답과 기본 실행기는 수정하지 않았다. 원본 코드 복사본 reference_search.py를 사용해 Claude 쪽 변경과 분리했다.',
'- 현재 결과는 같은 공개 지도·같은 Llama 응답에서의 탐색적 비교다. GPT-4o mini나 숨겨진 지도에서의 우수성, 알고리즘의 학술적 신규성을 입증하지 않는다.',
'- 모든 가능한 개선을 완전 탐색한 것은 아니다. 새 추론, 프롬프트 변경, 모델 학습/양자화, 외부 추가 응답은 제외했고 확인된 실패 원인을 겨냥한 61개 조건을 실험했다.',
'', '## 통계적 오류 점검 — 11/11',
'1. Simpson: 프롬프트별 결과와 통합 결과를 함께 제공한다. 통합 순위가 모든 프롬프트의 순위를 의미하지 않는다.\n2. 생태학적 오류: 평균으로 모든 사례의 개선을 주장하지 않고 악화 횟수·최악값을 제공한다.\n3. Berkson: 성공 사례만 선택하지 않고 전체 문제를 포함한다. 공개 데이터셋 선택 편향은 남는다.\n4. Collider: 결과로 표본을 걸러 비교하지 않는다. 후보 선택은 개발 구간 제약을 공개한다.\n5. 기저율 무시: 모든 비율의 분모를 제공한다.\n6. 평균 회귀: 기존 최악 사례만 다시 평가하지 않고 전체 문제를 평가한다.\n7. 생존 편향: 전체 유효성/건수를 검사한다.\n8. 다중 탐색: 실패 조건 포함 전 조건 공개, p값 및 확증적 유의성 주장 없음.\n9. 분석 선택 자유도: 코드/설정/분할/선택 기준 보존, 탐색적 실험임을 명시한다.\n10. 상관과 인과: 고정 입력에서 코드 변경의 효과는 비교하지만 새 데이터나 LLM 품질 전반의 인과로 확장하지 않는다.\n11. 역인과: A* 정답 길이는 집계·검증에만 사용하며 실제 알고리즘의 목표 선택·가중치 계산에는 넣지 않는다.',
'', '## 재실행',
'작업 폴더에서 다음 명령을 순서대로 실행한다. 기존 exp/gpt 결과를 덮어쓰므로 보존이 필요하면 이 폴더를 먼저 복사한다. 모든 실행은 모델 호출 없이 CPU에서 수행한다.',
'```bash\nPYTHONDONTWRITEBYTECODE=1 .conda/envs/llm-astar/bin/python -B exp/gpt/experiment.py > exp/gpt/run.log 2>&1\nPYTHONDONTWRITEBYTECODE=1 .conda/envs/llm-astar/bin/python -B exp/gpt/supplement.py > exp/gpt/supplement.log 2>&1\nPYTHONDONTWRITEBYTECODE=1 .conda/envs/llm-astar/bin/python -B exp/gpt/budget.py > exp/gpt/budget.log 2>&1\nPYTHONDONTWRITEBYTECODE=1 .conda/envs/llm-astar/bin/python -B exp/gpt/refinement.py > exp/gpt/refinement.log 2>&1\nPYTHONDONTWRITEBYTECODE=1 .conda/envs/llm-astar/bin/python -B exp/gpt/summarize.py > exp/gpt/summary.log 2>&1\nPYTHONDONTWRITEBYTECODE=1 .conda/envs/llm-astar/bin/python -B exp/gpt/benchmark.py > exp/gpt/timing.log 2>&1\nPYTHONDONTWRITEBYTECODE=1 .conda/envs/llm-astar/bin/python -B exp/gpt/write_report.py\n```']
formatted=[]
for item in lines:
 if item.startswith(('|','- ')) and formatted and formatted[-1] and not formatted[-1].startswith(('|','- ')):formatted.append('')
 formatted.append(item)
 if item.startswith('#'):formatted.append('')
(OUT/'REPORT.md').write_text('\n'.join(formatted)+'\n')
print('REPORT.md written')
