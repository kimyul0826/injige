from pathlib import Path
import json,zipfile,hashlib,statistics,shutil,csv
P=Path('/home/skkai/Mobility/2026_2/ykim/assign');out=P/'outputs/llama3-diagonal-barriers-v3-analysis';a=json.loads((out/'analysis.json').read_text())
for z in (P/'outputs').glob('*.zip'):
 with zipfile.ZipFile(z) as f:
  n=next(n for n in f.namelist() if n.endswith('/input.json'))
  a['checks'][z.stem]['input_semantically_matches_source']=json.loads(f.read(n))==json.loads((P/'data/environment_50_30.json').read_text())
  a['checks'][z.stem]['input_hash_note']='Manifest hashes original input bytes; bundled input is JSON-reserialized. Semantic contents equal.'
old=[json.loads(l) for l in (P/'outputs/llama3-table1-upstream-v2/results.jsonl').read_text().splitlines()];bs=[r for r in old if r['method']=='astar']
for p in a['by_prompt']:a['by_prompt'][p]['old_hybrid']['median_astar_search_seconds']=statistics.median(r['search_seconds'] for r in bs)
a['checks']['old_new_search_hash_equal']=json.loads((P/'outputs/llama3-table1-upstream-v2/experiment-code.json').read_text())['source_sha256']['llm_astar/search.py']==hashlib.sha256((P/'llm_astar/search.py').read_bytes()).hexdigest()
(out/'analysis.json').write_text(json.dumps(a,ensure_ascii=False,indent=2))
S={(r['method'],r['prompt']):r for r in a['old_summary']};T={(r['method'],r['prompt']):r for r in a['new_summary']}
lines=['# V3 장애물·8방향 설명 변경 결과 비교','', '## Material Passport','', '- Origin Mode: validate','- Origin Date: 2026-11-20 (Asia/Seoul)','- Verification Status: ANALYZED','- Version Label: llama3_v3_vs_v2_v1','', '## 결론','', '새 프롬프트가 전반적인 성능 개선을 보였다고 보기 어렵다. LLM 단독 Standard 성공률은 12.7%에서 8.5%로 하락했다. CoT·RePE 성공률은 각각 0.5%p, 0.1%p 하락했다. LLM+A*는 모든 조건에서 100% 성공했으며 탐색량과 경로 길이는 기존과 대체로 비슷하다. 현재 V3의 탐색 효율과 생성 비용을 함께 보면 Standard가 가장 유리하다.','', '## 비교 자료·검산','', '- 신규: llama3-diagonal-barriers-v3.zip, llama3-diagonal-barriers-cot-repe-v3.zip. 각 1,000건·2,000건 생성 완료. 원본 ZIP 유지.','- 기준: llama3-table1-upstream-v2. 초기 V1은 부록에서 비교하고, 후처리 filtered 결과는 동일 후처리를 적용한 별도 민감도 분석으로 비교했다.','- 지도 100개 × 시작·도착 조합 10개 = 방식당 1,000문제. 같은 입력 내용·sample ID로 대조했다.','- ZIP 무결성 통과. 중복 A* 기준값을 제거하여 7,000개 결과 행으로 통합했다. 두 ZIP의 A* 기준값은 기존 V2와 경로·성공 여부·탐색량·저장량·길이가 같다.','- 저장된 요약표를 재계산하여 일치 확인. V3 7,000개 행의 성공 여부·유효 경로 길이와 3,000개 응답의 파싱 결과를 재검산했으며 불일치 0건.','- 신규 생성 3,000건 모두 파싱 오류·토큰 한도 종료 0건. 이는 경로 충돌이 없다는 뜻은 아니다.','- ZIP input.json의 바이트 해시는 manifest의 원본 해시와 다르다. 원본을 JSON으로 다시 저장한 데 따른 차이이며, JSON 내용은 원본과 완전히 일치한다.','', '## LLM 단독','', '| 방식 | V2 성공 | V3 성공 | 차이 | V2 경로 길이 | V3 경로 길이 |','|---|---:|---:|---:|---:|---:|']
for p in a['by_prompt']:
 x,y=S['llm',p],T['llm',p];lines.append(f"| {p} | {x['valid_path_ratio']:.1f}% | {y['valid_path_ratio']:.1f}% | {y['valid_path_ratio']-x['valid_path_ratio']:+.1f}%p | {x['relative_path_length']:.2f}% | {y['relative_path_length']:.2f}% |")
lines+=['','길이는 성공한 문제에서 A* 대비 비율의 기하평균이다. 성공 집합이 달라지므로 전체 평균 길이 감소만으로 동일 문제의 경로 개선을 주장할 수 없다.','', '| 방식 | 기존 성공→실패 | 기존 실패→성공 | 공통 성공 수 | 공통 성공 길이 V2→V3 | 성공률 차이 95% 구간 |','|---|---:|---:|---:|---:|---:|']
for p,d in a['by_prompt'].items():
 lo,hi=d['llm_valid_difference_pp_ci'];lines.append(f"| {p} | {d['transitions']['valid_to_invalid']} | {d['transitions']['invalid_to_valid']} | {d['common_valid']} | {d['common_valid_length_old_pct']:.2f}% → {d['common_valid_length_new_pct']:.2f}% | [{lo:+.1f}, {hi:+.1f}]%p |")
lines+=['','위 구간은 지도를 단위로 20,000회 짝지은 부트스트랩을 수행한 탐색적 95% 구간이다. 같은 지도의 시작·도착 10개를 함께 재표집했다. 다중 비교 보정은 적용하지 않았으며, 인과 효과나 여러 번 실행했을 때의 변동을 나타내지 않는다.','', '## LLM+A*','', '성공률은 V2·V3의 모든 방식에서 100%. 아래 수치는 모두 A*를 100%로 한 문제별 비율의 기하평균이다. 탐색량은 heap pop 횟수, 저장량은 g 사전의 항목 수로, 실제 지연 시간이나 GPU 메모리 사용량과 다르다.','', '| 방식 | 탐색량 V2→V3 | 저장량 V2→V3 | 길이 V2→V3 |','|---|---:|---:|---:|']
for p in a['by_prompt']:
 x,y=S['llm-astar',p],T['llm-astar',p];lines.append(f"| {p} | {x['operation_ratio']:.2f}% → {y['operation_ratio']:.2f}% | {x['storage_ratio']:.2f}% → {y['storage_ratio']:.2f}% | {x['relative_path_length']:.2f}% → {y['relative_path_length']:.2f}% |")
lines+=['','| 방식 | 탐색량 변화 V3/V2 | 지도 부트스트랩 95% 구간 | A*보다 탐색량 큰 문제 V2→V3 | 길이 10% 초과 문제 V2→V3 |','|---|---:|---:|---:|---:|']
for p,d in a['by_prompt'].items():
 c=d['hybrid_changes']['operation'];lo,hi=c['map_bootstrap_ci'];lines.append(f"| {p} | {(c['new_over_old_geomean']-1)*100:+.2f}% | [{(lo-1)*100:+.2f}, {(hi-1)*100:+.2f}]% | {d['old_hybrid']['operation_worse_than_astar']} → {d['new_hybrid']['operation_worse_than_astar']} | {d['old_hybrid']['length_over_10pct']} → {d['new_hybrid']['length_over_10pct']} |")
lines+=['','탐색량 변화 구간은 세 방식 모두 0을 포함한다. V3 Standard의 탐색량 45.15%는 기하평균으로 약 54.85% 감소한 수치다. 문제 전체의 탐색 횟수 합계로 계산하면 64.37%이므로 총량 감소율은 약 35.63%다. 서로 다른 집계 방식을 혼동하면 안 된다.','', '## 실패 원인','', '아래 항목은 중복될 수 있다. 동일한 Shapely 충돌 검사로 경유점과 경유점 사이 선분까지 확인했다.','', '| 방식 | 장애물 교차·접촉 V2→V3 | 장애물 위 경유점 V2→V3 | 경계·범위 밖 경유점 V2→V3 |','|---|---:|---:|---:|']
for p,d in a['by_prompt'].items():
 x,y=d['failures_overlapping']['old'],d['failures_overlapping']['new'];lines.append(f"| {p} | {x.get('barrier_intersection',0)} → {y.get('barrier_intersection',0)} | {x.get('waypoint_on_barrier',0)} → {y.get('waypoint_on_barrier',0)} | {x.get('boundary_or_outside_point',0)} → {y.get('boundary_or_outside_point',0)} |")
lines+=['','Standard는 벽 좌표·끝점을 경유점으로 삼는 사례가 늘었다. 예를 들어 sample 0:8에서 기존 경로는 `[20,6]→[26,9]→[25,23]→[40,25]`였고, 신규는 `[20,6]→[25,6]→[25,25]→[40,25]`이다. 신규 경로의 x=25 선분이 y=10~22의 수직 벽과 겹친다. 좌표를 명시하는 표현이 벽 좌표를 경유점으로 선택하게 했을 가능성은 있지만, 원인은 추가 대조 실험 없이 확정할 수 없다.','', 'RePE의 범위·경계 위 경유점은 315→34건으로 크게 줄었다. 일부 오류 유형은 줄었지만 벽 교차·접촉이 늘어 전체 성공률 개선으로 이어지지 않았다.','', '## 동일한 경유점 필터를 적용한 민감도 분석','', '공식 원본 결과를 바꾸지 않고, V2와 V3 모두 기존 upstream 목표점 필터를 적용하여 LLM 단독 성공률을 재평가했다. 벽 위 일부 점·범위 밖 점을 제거하고 시작·도착을 보정하는 후처리다. 벽 사이 충돌을 모두 해결하는 필터는 아니다.','', '| 방식 | V2 filtered | V3 filtered |','|---|---:|---:|']
for p,d in a['by_prompt'].items():lines.append(f"| {p} | {d['old_llm_filtered_success']/10:.1f}% | {d['new_llm_filtered_success']/10:.1f}% |")
lines+=['','## 생성 비용과 비교 제한','', '- V3 실행 로그: RTX 5090 두 장, PyTorch 2.14.1+cu132. V2 환경 기록: PyTorch 2.10.0+cu130. 모델 revision·BF16·temperature=0·최대 8,000토큰·Transformers 4.57.3는 같다. 가중치 전체 바이트 해시는 검증하지 않았다.','- 검색·충돌 코드 해시는 V2와 같고, V3의 로컬 분석 코드 해시도 manifest와 일치한다. 프롬프트와 생성 환경이 함께 달라져 순수 프롬프트 효과를 분리할 수 없다.','- V3 작업시간: Standard 약 11분 21초, CoT+RePE 약 100분 24초. 병렬 실행과 초기화가 포함되어 개별 질의 지연 시간과 다르다.','', '| 방식 | V3 생성 시간 중앙값 | V3 검색 시간 중앙값 | 생성 토큰 중앙값 |','|---|---:|---:|---:|']
for p,d in a['by_prompt'].items():
 h=d['new_hybrid'];lines.append(f"| {p} | {h['median_generation_seconds']:.3f}초 | {h['median_search_seconds']:.4f}초 | {h['median_completion_tokens']:.0f} |")
lines+=['','V3의 순수 A* 검색 중앙값은 약 0.0101초다. LLM+A*는 탐색 횟수를 줄여도 LLM 생성 비용이 추가되어, 이 작은 지도에서 전체 응답이 더 빠르다는 근거는 없다. 이전 환경과 생성 시간만 직접 비교하여 프롬프트의 속도 개선으로 주장하면 안 된다.','', '## 초기 V1 참고','', '| 방식 | V1 LLM 성공률 | V2 LLM 성공률 | V3 LLM 성공률 |','|---|---:|---:|---:|']
v1=json.loads((P/'outputs/llama3-table1/summary.json').read_text());v1={r['prompt']:r for r in v1 if r['method']=='llm'}
for p in a['by_prompt']:lines.append(f"| {p} | {v1[p]['valid_path_ratio']:.1f}% | {S['llm',p]['valid_path_ratio']:.1f}% | {T['llm',p]['valid_path_ratio']:.1f}% |")
lines+=['','## 해석 점검','', '통계·방법론 오류 유형 11/11 점검. 지도별 짝지음과 전체 표본의 성공률을 유지했다. 생존자 편향은 성공 경로 길이에서 주의가 필요하여 공통 성공 집합을 별도로 제시했다. 여러 지표 탐색과 분석 선택은 탐색적이며 다중 비교 보정을 적용하지 않았다. 인과 주장에는 생성 환경 변경이라는 교란이 있다. Simpson 역설은 지도별 차이를 고려하여 전체 결과를 일반화하지 않았으며 모든 지도에서 같은 방향이라고 주장하지 않는다. 생태학적 오류·collider/Berkson 편향·기저율 무시·평균 회귀·역인과를 뒷받침하는 추가 설계는 확인되지 않았다.','', '새 모델 추론은 수행하지 않았다. 저장된 응답·경로·집계값을 분석하고 재검산한 결과이며, 모델 추론 재현성까지 VERIFIED로 판정하지 않는다.','', '현재 결과에서는 V2를 기존 기준 실험으로 유지하고 V3는 입력 설명 변경 실험으로 구분하는 해석이 적절하다. 추가 인과 확인에는 동일한 GPU·소프트웨어에서 기존 프롬프트 대조군을 실행하고, 8방향 문장과 벽 표기 변경을 각각 분리해야 한다.','']
(out/'report.md').write_text('\n'.join(lines))
shutil.copy('/tmp/analyze_llama3_v3.py',out/'analyze.py')
shutil.copy('/tmp/report_llama3_v3.py',out/'build_report.py')
print(out/'report.md')
