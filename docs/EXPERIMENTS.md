# 실험 전체 정리 (2026-11-22 기준)

이 문서는 과제(LLM-A* Table 1 재현 + 개선 알고리즘 제안)에서 수행한 **모든 추론 실험과 추론 이후 알고리즘 실험**을
하나의 논리로 묶는다. 각 실험마다 (1) 원 논문 대비 바뀐 점, (2) 설정 기준, (3) 결과, (4) 의도와 구현이 어긋난 점,
(5) 재현 방법을 적는다. 수치는 모두 저장된 원자료에서 다시 계산한 값이다.

- 지표: 문제별 A* 대비 비율의 기하평균 ×100 (낮을수록 좋음). op = heap pop 수(공식 코드처럼 중복 pop 포함, "upstream 집계"),
  sto = g 사전 크기, len = 경로 길이. LLM-A*의 성공률은 모든 실험에서 100%.
- 데이터: `data/environment_50_30.json` = 51×31 격자 지도 100개 × 시작·도착 10쌍 = 1,000문제 (원 논문 공개 데이터).
- "paired"는 같은 문제끼리의 비율의 기하평균, 괄호는 문제 단위 bootstrap 95% CI.
- 통계 해석 주의: 같은 지도의 10개 문제는 독립이 아니다. 문제 단위 bootstrap은 CI를 좁게 잡을 수 있다(지도 단위 bootstrap은 Llama 오프라인 분석에서만 했음).

---

## 0. 한 문단 요약 (논리의 뼈대)

1. **재현:** Llama-3-8B로 Table 1을 재현했다(Few-Shot op 45.55 대 논문 44.59). GPT-4o mini로 같은 실험을 확장했다(49.96 / 48.41 / 53.24).
2. **진단:** 공식 LLM-A*는 경유점이 없으면 f = g + 2·h가 되어 weighted A*(w=2, op 34.32)와 같다. 즉 **LLM 경유점이 붙으면 오히려 나빠진다**
   (GPT-4o mini, LLM/no-LLM paired 141–155%). 원인은 (a) 충돌·우회 경유점, (b) 지나친 target을 계속 쫓는 target 전환 규칙.
   무작위 경유점도 LLM 경유점과 비슷한 성능을 낸다(§B3).
3. **추론 쪽 개선(프롬프트):** 장애물 표기·8방향·상한·VoT·후보점·끝점 프롬프트는 효과가 없거나 작았다. **RePE2**(경유점 수를 0개부터
   하나씩 늘리며 스스로 점검하는 1회 호출 프롬프트)가 GPT-4o mini LLM-A*를 **op 40.12**로 낮췄다(원본 대비 paired 75–83%, 유의).
   그러나 여전히 weighted A*보다 나쁘다(116.9%).
4. **추론 이후 쪽 개선(알고리즘):** target 전환 수정(progress)은 폭주를 없애고, 기하 휴리스틱 **wallC**(벽 연결 성분의 볼록껍질을 도는 우회 거리,
   허용적)는 LLM 없이 op **15.44**를 낸다. 공식 탐색 변형(progress, 가중치)과 wallC에서는 GPT-4o mini 경유점의 순기여가 0 근처다
   (같은 알고리즘의 no-LLM 대조군 대비 99–104%). **예외: 충돌 없는 구간만 쓰는 E-Graph에서 끝점 standard 응답은 no-LLM 대비 −12%**
   (87.6% [85.6, 89.5], 지도 홀짝 양쪽 동일; 다른 지도 유형 500문제에서도 93.3% [90.7, 95.8]). RePE2의 개선도 다른 유형으로 일반화(standard 대비 71.3%, 미로 제외). E1 절제로 RePE2의 개선은 예시가 아니라 Round 절차에서 온다는 것이 확인됐다.
5. **상한:** 정답 경유점을 주면 wallC+route가 op 12.14까지 내려간다(−21%). 병목은 LLM의 충돌·가시성 판단이다
   (RePE2 최종 경로의 실제 무충돌 2.8%, 자기 판정 "Valid" 98.1%).

---

## A. 추론(LLM 호출) 실험

공통 원 논문 설정: LLM-A* (Meng et al., 2024). 공개 프롬프트 standard(5-shot), CoT(3-shot), RePE(3-shot).
공식 코드의 GPT 호출은 `gpt-3.5-turbo-instruct` completion API, `temperature=0`, `max_tokens=100`, `stop=["\n"]`.

### A1. Table 1 재현 — Llama-3-8B-Instruct
| 항목 | 원 논문 | 우리 |
|---|---|---|
| 모델 | Llama-3-8B | Meta-Llama-3-8B-Instruct, revision 고정, bf16, 로컬 Transformers |
| 디코딩 | 명시 없음 | greedy(temperature 0), max_new_tokens 8000 (잘림 방지 기준) |
| 프롬프트 | 공개 3종 | 동일 (vendor 복사본이 upstream과 바이트 동일함을 확인) |
| 탐색/집계 | 공식 코드 | 공식 코드 재구현, 4,000개 경로·지표가 공식 구현과 일치 (`scripts/verify_table1.py`) |

| | 논문 op / sto / len | 우리 op / sto / len | LLM 단독 성공 (논문 / 우리) |
|---|---|---|---|
| standard | 44.59 / 64.02 / 102.47 | 45.55 / 64.78 / 102.55 | 12.6 / 12.7% |
| CoT | 47.60 / 66.27 / 102.46 | 48.04 / 66.73 / 102.52 | 12.0 / 11.4% |
| RePE | 64.08 / 80.19 / 102.54 | 63.95 / 80.14 / 102.60 | 16.4 / 10.5% (공식 필터 적용 시 16.6%) |

- 재현: `configs/llama3_table1_upstream.json`, 결과 `outputs/llama3-table1-upstream-v2`, 검증 `python scripts/verify_table1.py`.

### A2. Table 1 확장 — GPT-4o mini (과제 평가 모델)
- 변경: 모델 `gpt-4o-mini-2024-07-18`(스냅샷 고정), Chat Completions, `temperature 0`, `max_tokens 8000`, OpenAI Batch API.
- 설정 기준: TA 평가 모델이 GPT-4o mini라서. max_tokens는 CoT/RePE가 잘리지 않도록 크게 둠. 스냅샷 고정은 재현성.
- 결과(1000문제): op 49.96 / 48.41 / 53.24, sto 68.10 / 67.08 / 70.51, len 102.95 / 102.63 / 102.54, LLM 단독 6.5 / 7.4 / 8.8%, 잘림 23 / 12 / 2.
- **원 논문 GPT-3.5 행의 재현이 아님**: 모델·API·stop/max_tokens가 다름. 논문 GPT-3.5 Few-Shot op 57.39는 참고로만 비교. (`configs/gpt35_upstream.json`은 준비했지만 실행하지 않음.)
- 재현: `configs/gpt4omini_table1_upstream.json`, `scripts/run_batch_table1.py --zip`, 결과 `outputs/gpt4omini-table1-upstream-batch-v1`, 분석 `analysis/gpt4omini-table1-upstream-analysis/`.

### A3. Llama 프롬프트 표기 실험 (1000문제 × 3)
| 실험 | 원 프롬프트 대비 변경 | 설정 기준 | op std / cot / repe | LLM 단독 |
|---|---|---|---|---|
| diagonal-barriers v3 | 장애물을 "B1: horizontal barrier at y=…, from (…) to (…)" 목록으로, 8방향 이동 문장 | x·y 축 혼동 줄이기 | 45.15 / 49.44 / 66.62 | 8.5 / 10.9 / 10.4% |
| 8way-only v1 | 대각선 이동 문장 1줄만 | 위 효과 분리 | 45.51 / 48.26 / 64.48 | 10.9 / 11.2 / 10.2% |
| waypoint-limit v1 | "경유점 최대 N개" 문장, N = 장애물 개수 | 과다 경유점 억제. N은 맵 정보만으로 정해지는 값 | 44.69 / 48.00 / 59.65 | 7.2 / 7.7 / 11.6% |
- 결론: 원본과 거의 같다. weighted A*(34.3)보다 나쁘다.
- 어긋난 점: waypoint-limit 실행은 원본과 PyTorch 버전이 달라 순수 프롬프트 효과로 분리 불가(분석 보고서에 명시).
- 재현: `configs/llama3_8way_walls.json`, `llama3_8way_only.json`, `llama3_waypoint_limit.json`; 분석 `analysis/llama3-*`.

### A4. GPT-4o mini 경유점 상한 (1000문제 × 3)
- 변경: A3의 상한 문장을 GPT 프롬프트에 동일하게 추가(`prompt_variant: waypoint_limit_by_obstacles`).
- 결과: op 48.50 / 49.57 / 53.04 (원본 49.96 / 48.41 / 53.24). LLM 단독 4.2 / 7.2 / 8.6%.
- 결론: 효과 없음. 재현: `configs/gpt4omini_waypoint_limit.json`, 결과 `outputs/gpt4omini-waypoint-limit-batch-v1`.

### A5. VoT(Visualization-of-Thought, Wu et al. 2024) 계열
| 실행 | 문제 수 | 변경 | op | 관찰 |
|---|---|---|---|---|
| cot-vot (batch) | 1000 | CoT 프롬프트 + "Visualize the state after each reasoning step." | 51.34 (CoT 48.41) | 효과 없음 |
| vot-zeroshot (smoke) | 10, 지도 0번 | CoT 지시 + 마지막 문제만(예시 제거) + VoT 문장 | 74.60 | – |
| vot-waypoint-limit (batch) | 1000 | zero-shot VoT + 상한 | 52.51 | 잘림 98, 한 칸씩 걷기 294, 상한 준수 498/901 |
| vot-map (smoke, 중단) | 9/10, 지도 0번 | + ASCII 지도 | – | 격자 복사, BFS 흉내 |
| vot-map-waypoint-limit (smoke) | 10, 지도 0번 | + ASCII 지도 + 상한 | 74.09 | x·y 뒤바뀜(0:8에서 816%) |
| vot-coarse-waypoint-limit (smoke) | diverse10 | + "칸 단위가 아니라 장벽·꺾임점 단위로 추론" | 50.61 | 대부분 경유점이 필터에 걸려 weighted A*화 |
- **어긋난 점:**
  - 지도를 넣지 않은 VoT는 시각화가 아니라 "트리거 문장"뿐이다. 보고서에는 "VoT trigger (zero-shot)"로 표기해야 한다. 지도를 넣은 진짜 VoT는 모델이 지도를 베끼거나 축을 뒤바꿔 실패했다.
  - zero-shot VoT는 예시 제거와 VoT 문장 추가가 동시에 바뀌어 효과가 섞인다.
  - 지도 0번 하나로 한 스모크(45.3)가 1000문제(52.5)와 반대로 나와, 이후 스모크는 `data/smoke_diverse10.json`(서로 다른 지도 10개)으로 바꿨다.
- 재현: `configs/gpt4omini_cot_vot.json`, `gpt4omini_vot_*.json`; 프롬프트 생성 `llm_astar/prompt_variants.py`(`vot`, `ascii_map`, `coarse_reasoning`, `+` 조합).

### A6. 후보점 제시 프롬프트 (diverse10 스모크)
- 변경: 마지막 문제에 "Candidate Waypoints"(벽 끝점의 대각선 빈 칸, 맵당 약 30개)와 "후보 중에서 고르라" 문장 추가.
- 설정 기준: 좌표 오타·벽 위 좌표를 구조적으로 없애기. 후보 상한은 오프라인 계산으로 op 12.34(wallC+route)라 여지 손실이 없음을 먼저 확인.
- 결과: op 54.43 / 54.01 / 64.83, 후보 준수 standard 28/28, 충돌 없는 체인 1/30.
- 어긋난 점: CoT·RePE는 추론 중 새 좌표를 만들어 목록을 벗어남(20/26, 18/26만 준수).
- 재현: `configs/gpt4omini_candidate_waypoints.json`, 분석 `exp/claude/analyze_candidate_smoke.py`.

### A7. 장애물 끝점 프롬프트 (끝점 = 벽 양 끝 바로 옆 칸)
- 정의: 수평 [y, x1, x2] → [x1−1, y], [x2+1, y]; 수직 [x, y1, y2] → [x, y1−1], [x, y2+1]. 공식 필터가 지우는 칸(테두리 띠, 벽 위)은 제외(맵당 평균 7.0개).
- v1 스모크(`barrier_endpoints`): 마지막 문제에만 목록 + "후보에서 고르되 다 쓸 필요는 없음". op 52.97 / 56.30 / 58.82.
- **batch(`endpoint_demos+barrier_endpoints`)**: 예시 5개에도 끝점 목록, 예시 답 좌표 2개를 끝점으로 한 칸 이동([26,9]→[26,10], [11,6]→[11,7]; 둘 다 충돌 없음 확인).
  | | op 원본 → 끝점 | paired [CI] | LLM 단독 | 목록 준수 |
  |---|---|---|---|---|
  | standard | 49.96 → 52.50 | 105.1 [99.0, 111.7] | 6.5 → 34.7% | 94.2% |
  | CoT | 48.41 → 47.93 | 99.0 [93.7, 104.4] | 7.4 → 24.1% | 63.9% |
  | RePE | 53.24 → 46.89 | 88.1 [83.4, 92.7] | 8.8 → 14.8% | 30.4% |
- 실패 분석: 손해의 주원인은 "필요 없는 끝점을 순서대로 이어 붙인 우회 체인"(standard에서 최적의 1.6배 이상 체인 18 → 148건). 테두리 틈이 필요한 80문제의 악화는 부차적.
- 어긋난 점:
  - v1과 batch 사이에 "예시에 목록 추가"와 "예시 답 2개 수정"이 함께 바뀌어 효과를 분리할 수 없다. batch는 수정판만 돌렸다.
  - "다 쓸 필요 없음" 문장이 있었지만 경유점 0개 응답은 0건이었다.
  - 공식 필터와 맞추느라 테두리 쪽 틈 칸은 후보로 줄 수 없다(64:4에서 299%).
  - RePE의 개선은 목록 준수와 무관했다(지킨 응답 47.1, 어긴 응답 46.8). 원인은 미검증.
- 재현: `configs/gpt4omini_barrier_endpoints.json`, `gpt4omini_barrier_endpoints_demos.json`; 분석 `analysis/gpt4omini-barrier-endpoints-demos-analysis/`.

### A8. RePE2 (제안 프롬프트) — 경유점 수 단계적 증가 + 자기 점검, 1회 호출
- 원 논문 대비 변경:
  1. 새 스타일(standard/CoT/RePE와 별개). 머리말 = 원본 지시문 + "Round k는 경유점 정확히 k개, Round 0(직선)부터, 매 Round 후 모든 구간 점검, 처음 유효한 Round에서 멈춤" 1문장.
  2. 예시 5개를 **평가셋 밖의 새 무작위 지도**에서 생성(최소 경유점 0, 2, 1, 3, 1). 마지막 문제는 `Generated Path:` 없이 끝남(모델이 Round 0부터 쓰도록).
  3. 출력 형식·파서·탐색은 원본 그대로. 좌표는 자유(끝점 목록 없음).
- 설정 기준:
  - 원본 예시 5개는 평가 분포와 다름(직선 문제가 예시 40% 대 평가 0.6%; 예시 3개는 필요보다 많은 경유점 사용) → 새 예시.
  - 새 지도는 데이터셋의 장애물 개수·길이 분포에서 표본추출, 12,000문제 생성. 경유점 최소 개수 k를 공식 필터가 남기는 모든 칸에서 정확히 계산.
  - 각 k에서 "장애물 5–7개, 길이 3 미만 장애물 없음, 시작–목표 25 이상, 실제 우회(최적/직선 ≥ 1.1)" 조건을 통과한 문제 중 **정답 체인을 공식 LLM-A*에 넣었을 때 op가 가장 낮은 것**을 고름.
  - Check 문장은 코드의 실제 교차 판정으로 생성, 모든 Round 판정이 충돌 코드와 일치함을 테스트로 검증.
- 결과(1000문제, batch):
  | 비교 대상 | op | RePE2/대상 paired [CI] |
  |---|---|---|
  | **RePE2** | **40.12** (sto 60.98, len 102.37, 잘림 0) | – |
  | 원본 standard / CoT / RePE | 49.96 / 48.41 / 53.24 | 80.3 / 82.9 / 75.3 (모두 CI < 100) |
  | 끝점 RePE | 46.89 | 85.6 [81.8, 89.5] |
  | weighted A* w=2 | 34.32 | 116.9 [114.0, 120.3] |
  - A* 2배 초과 탐색 14건(원본 49–67건). 경유점 수 분포 {0:86, 1:609, 2:201, 3:96, 4+:8}가 최적 경로 꺾임 수 분포 {0:6, 1:774, 2:176, 3:40, 4:4}와 비슷.
  - 자기 판정 Valid 98.1%, 실제 무충돌 2.8%(= LLM 단독 성공률). 실패의 72%는 충돌 구간이 하나뿐이지만, **경유점 하나를 한 칸 옮겨 고쳐지는 경우는 7%뿐**(두 칸 이내 14%)이라 "근소한 실패"는 아니다. 경로가 장벽의 잘못된 쪽을 지나는 구조적 오류다 (`near_miss.py`). 성공 경로 길이 97.8.
  - Round별 자기 판정 정확도(`repe2_selfcheck.py`, 2,517개 Round): Round 0(직선) 91%, Round 1 37%, Round 2 37%. "Not valid" 판정은 99% 맞지만 "Valid" 판정은 97% 틀림(996건 중 968건). "Not valid" 다음 Round가 실제로 충돌을 고친 경우 2.4%. → 모델은 **막힌 것은 알아보지만, 자기가 고친 경로가 맞는지는 검증하지 못한다.**
- **E1 절제 실험 (1000문제 × 2, 이번에 실행):**
  | 프롬프트 | op | paired |
  |---|---|---|
  | E1-a 새 예시, Round 없음 (standard 형식) | 51.15 | 원본 standard 대비 102.4 [97.7, 107.7] |
  | E1-b 원본 예시를 Round 형식으로 + Round 지시문 | **40.27** | RePE2 대비 100.4 [97.7, 103.2]; 원본 standard 대비 80.6 [76.9, 84.4] |
  - **개선은 Round 절차에서 나온다. 새 예시(효율 기준 선택)는 효과가 없다.** 예시 선택 기준에 대한 우려도 함께 해소됨.
  - E1-b는 잘림 17건(RePE2 0건), Round를 길게 반복하는 응답이 있음.
- 해석: 경유점이 정확해서가 아니라 **적고 직선에 가까워져** 공식 LLM-A*의 손해가 줄었다.
- **어긋난 점:**
  - (해소됨) 예시 교체와 Round 절차가 동시에 바뀐 문제는 E1으로 분리했다. Round 절차 = "Round 형식 예시 + 지시문 한 문장"이며, 둘은 분리하지 않았다.
  - 이름은 RePE2지만 RePE 템플릿에서 파생되지 않았다(standard 머리말 기반). 보고서에서 명칭 설명 필요.
  - "자기 평가"를 의도했지만 판정은 거의 항상 Valid였다(실질적 평가가 일어나지 않음).
  - 성공 Round의 Check 문장은 "Every segment avoids all barriers."로 이유가 없다.
  - 예시 선택에 op(평가 지표)를 썼다. 평가 지도와 겹치지 않음은 테스트로 확인했지만, 선택 기준은 보고서에 명시해야 한다.
- 재현: `configs/gpt4omini_repe2.json`, `--prompts repe2`; 예시 생성 `exp/claude/build_repe2_demos.py` → `render_repe2_demos.py`; 분석 `analysis/gpt4omini-repe2-analysis/`; 테스트 `tests/test_repe2_prompt.py`.

### A9. E3 — 다른 지도 유형에서의 일반화 (GPT-4o mini, 500문제)
- 입력: `data/bench_e3.json` = `exp/claude/bench`(시드 고정 생성)의 dense, rooms, utrap(U자 함정), maze(미로), scaled2(101×61)에서 앞 100문제씩. 원본과 같은 JSON 형식, 프롬프트·예시 변경 없음.
- 프롬프트: 원본 standard(기준), RePE2, 끝점 standard. 공식 LLM-A*는 upstream 집계, E-Graph(εE=1.5, w=1.33)는 중복 제거 집계.

| 유형 | wA*2 | wallC | LLM-A* standard | **LLM-A* RePE2** | RePE2/standard | E-Graph 경유점 없음 | **E-Graph 끝점** | E-Graph 끝점/없음 |
|---|---|---|---|---|---|---|---|---|
| dense | 21.30 | 18.59 | 37.14 | 23.26 | 62.6 [51.7, 75.5] | 23.49 | 23.62 | 100.6 [95.3, 106.9] |
| rooms | 23.96 | 16.26 | 42.68 | 27.71 | 64.9 [54.4, 76.8] | 25.86 | 23.92 | **92.5 [87.3, 97.7]** |
| utrap | 52.75 | 47.86 | 76.79 | 54.33 | 70.7 [62.8, 78.9] | 49.87 | 48.11 | **96.5 [94.1, 98.8]** |
| maze | 78.64 | 64.56 | 92.66 | 88.98 | 96.0 [88.6, 103.8] | 68.66 | 67.62 | 98.5 [96.5, 99.9] |
| scaled2 | 22.01 | 6.90 | 40.38 | 27.01 | 66.9 [55.2, 81.1] | 22.71 | 18.17 | **80.0 [72.1, 87.8]** |
| 전체 | 34.17 | 23.00 | 53.91 | **38.46** | **71.3 [66.6, 76.1]** | 34.26 | **31.97** | **93.3 [90.7, 95.8]** |

- **RePE2의 개선은 다른 지도 유형으로 일반화된다**(미로 제외 4개 유형에서 standard 대비 29–37% 감소). 그러나 모든 유형에서 weighted A*보다 나쁘다(전체 112.6% [108.8, 116.8]).
- **끝점 + E-Graph의 LLM 기여도 일반화된다**(전체 −6.7%, rooms·utrap·scaled2에서 유의; dense는 효과 없음, maze는 경계).
- wallC가 모든 유형에서 가장 낮다(scaled2 6.9).
- 실패 지점 (`failure_by_family.py`): 미로는 최적 경로 꺾임이 평균 6.57개 필요한데 RePE2는 2.21개(필터 후 1.14개, 28%는 0개)만 내서 "가장 적은 개수부터" 규칙이 과소 공급으로 실패한다. standard는 dense/rooms/utrap에서 경유점 13–16개(한 칸씩 걷기)를 내며 잘림 5–10건.
- LLM 단독 성공률: standard 1%, RePE2 3%, 끝점 10% (scaled2 끝점 31%).
- 재현: `--input data/bench_e3.json`, 출력 `outputs/gpt4omini-e3-bench-batch-v1`, `outputs/gpt4omini-e3-bench-endpoint-batch-v1`; 분석 `analysis/e3-bench/analyze.py`, `failure_by_family.py`.

---

## B. 추론 이후(알고리즘) 실험 — 새 LLM 호출 없음, 저장 응답 재사용

모든 오프라인 하네스는 공식 LLM-A* 결과를 저장값과 assert로 대조한다(Llama 3,000건 + A* 1,000건, GPT 각 세트 1,000건 일치).

### B1. 기준선 진단
- 공식 LLM-A*에서 경유점이 모두 필터되면 f = g + 2h → weighted A*(w=2)와 동일(op 34.32 / len 102.02). 공식 LLM-A*(Llama 45.6–64.0, GPT 46.9–53.2)보다 둘 다 좋음.
- 경유점이 하나라도 남은 문제만 보면 LLM-A*가 같은 문제의 weighted A*보다 대체로 나쁨(`waypoint_subset.py`).
- 논문의 dynamic weighted A* 기준선도 재구현: 해석 두 가지에 따라 op 33.66 / 31.26 (`dwa_check.py`).

### B2. 공식 LLM-A* 탐색 변형 (Llama, `offline_improve.py`, 35개 변형)
- target 전환 `progress`(현재 노드가 target보다 목표에 가까우면 그 target 폐기): op 45.55/48.04/63.95 → 36.16/36.03/35.24, A* 2배 초과 40/50/124 → 3/3/3.
- `wg=2, wt=0.5`: 34.10/34.18/35.37. snap/clamp(무효 경유점 살리기)는 악화. 3응답 앙상블, 유효성 게이트(+약 1%) 등.
- 결론: 개선의 대부분은 weighted A* 쪽으로 가까워지는 효과. 변형 선택이 같은 데이터에서 이루어진 사후 탐색임(지도 홀짝 분할로 방향만 확인).

### B3. 경유점 정보량 진단 (Llama)
- 무작위 경유점(시작–목표 사각형 안, 진행 방향 정렬, 공식 탐색): 1개 39.84, 2개 45.32, 3개 50.99. Llama 원본 프롬프트 LLM-A*(45.55–63.95)보다 같거나 좋음 (`random_waypoints.py`, `per_map_compare.py`).
- 토폴로지(어느 쪽으로 도는지) 정답률: 규칙 86.4%, 직선 기준 81.3%, Llama standard 77.2%, 무작위 56.0% (`topology_vs_trivial.py`).
- 휴리스틱 품질 분해: LLM 경유점 route 휴리스틱의 순위상관 ≈ 0 (oracle 0.75) (`heuristic_quality.py`).

### B4. 경계 보장 탐색 (Llama, `logic_variants.py`)
- E-Graph(Phillips et al., 2012), MHA*(Aine et al., 2016), Focal(A*ε). 각 방법에 no-LLM 대조군.
- E-Graph εE=1.5, w=1.33: oracle 경유점이면 op 21.8 / len 100.25, Llama RePE는 no-LLM 대비 96.4% [94.6, 98.2], standard/CoT는 이득 없음.
- 결론: 로직은 좋은 경유점의 이득을 키우지만 Llama 경유점의 순기여는 거의 0.

### B5. 기하 휴리스틱 wallC (LLM 없음)
- 정의: 맞닿은 벽을 연결 성분으로 묶고, s→goal 선분이 가로지르는 성분마다 볼록껍질을 도는 최단 우회 거리; 테두리에 붙은 껍질 꼭짓점 통과 금지. h = max(euclid, 성분별 우회). 노드마다 닫힌 형태로 계산(사전 탐색 없음), 허용적.
- 1000문제(upstream 집계): w=1.5 op **15.44**, len 100.15, 무효 경로 0 (`fairness_check.py`). 중복 제거 집계에서는 18.48.
- 비용: 확장 수는 줄지만 휴리스틱 계산 때문에 1회 탐색 시간은 A* 0.80 ms, weighted A* 0.43 ms, wallC 1.67 ms (`cost_check.py`). **op 지표의 이득이 실행 시간 이득은 아님.**
- 다른 지도 유형(`exp/claude/bench/`, 중복 제거 집계)에서도 weighted A*(w=2)보다 op가 낮음: dense 23.1→20.6, rooms 26.1→19.2, U-trap 48.0→42.6, maze 67.0→60.5, 원본 34.5→18.5 (`wall_component_h.py`).
- 선행 연구: 가시성 그래프(Lozano-Pérez & Wesley 1979), Subgoal Graphs(Uras & Koenig 2013), 허용 휴리스틱의 max(Pearl 1984)와 가깝다. 정확히 같은 정의는 찾지 못했지만 **새로움을 주장하지 않는다**.

### B6. LLM 경유점 + wallC 결합 (Llama, GPT)
- wallC+route: f = g + 1.5·max(wallC(s), route(s)), route = s에서 보이는 체인 꼭짓점까지 + 체인 따라 목표까지. `full` 규칙(모든 구간 무충돌일 때만 체인 사용).
- GPT 원본·상한 응답: 15.40–15.46 = wallC 단독 15.44. 정답 경유점이면 12.14. `raw`(검증 없음)는 28–32로 악화, 최악 경로 374%.

### B7. 추론 후 변형을 GPT-4o mini 전 응답에 적용 (이번 정리에서 추가, `post_inference_gpt.py`)
같은 알고리즘에서 경유점만 [start, goal]로 바꾼 no-LLM 대조군과 paired 비교.
| 응답 세트 | official | progress | progress + wg2·wt0.5 | wallC+route (full) |
|---|---|---|---|---|
| no-LLM 대조군 | 34.32 (wA*2) | 34.32 | 31.60 (= wA* 2.5) | 15.44 (wallC) |
| 원본 std / cot / repe | 49.96 / 48.41 / 53.24 | 37.90 / 36.87 / 38.25 | 32.56 / 32.22 / 32.76 | 15.42 / 15.46 / 15.45 |
| 끝점 std / cot / repe | 52.50 / 47.93 / 46.89 | 34.49 / 34.70 / 35.58 | 31.37 / 31.57 / 32.03 | **15.05** / 15.41 / 15.40 |
| RePE2 | 40.12 | 37.15 | 32.57 | 15.44 |
| LLM/no-LLM paired | 137–155% | 100.5–113% | 99.3–104.4% | 97.4–100.1% |
- 유의하게 100 미만인 조합은 **끝점 standard + wallC+route (97.4% [95.9, 98.8])** 하나뿐.
- 결론: 탐색을 고치면 LLM 경유점의 손해는 사라지지만 이득도 거의 없다.

E1 세트 추가: E1-a 51.15 → progress+w 32.71 / wallC 15.38, E1-b 40.27 → 32.43 / 15.49 (no-LLM 대비 99.6–103.5%). 결론 동일.

### B9. E-Graph를 GPT 응답에 적용 (이번 정리에서 추가, `egraph_gpt.py`, 중복 제거 집계)
| 응답 | εE=1.5, w=1.33 op / len | LLM/no-LLM [CI] (지도 짝/홀) | εE=2, w=1.25 op / len | LLM/no-LLM |
|---|---|---|---|---|
| no-LLM | 34.52 / 101.99 | – | 32.19 / 102.98 | – |
| oracle 경유점 | 21.76 / 100.25 | 63.0 | 18.47 / 100.22 | 57.4 |
| 원본 std / cot / repe | 35.1–35.8 | 101.7–103.8 | 34.6–36.2 | 107.4–112.4 |
| **끝점 standard** | **30.24 / 102.85** | **87.6 [85.6, 89.5] (88.1 / 87.1)** | **28.24 / 104.58** | **87.7 [85.6, 90.3]** |
| 끝점 cot / repe | 32.16 / 32.75 | 93.2 / 94.9 (CI < 100) | 30.50 / 31.40 | 94.7 / 97.6 |
| RePE2 / E1-a / E1-b | 37.37 / 36.26 / 36.45 | 108.3 / 105.0 / 105.6 | 35.90 / 36.97 / 35.04 | 108.9–114.9 |
| weighted A* w=2 (참고) | 34.48 / 102.02 | | | |
- **이번 작업에서 처음으로 LLM 경유점의 유의한 순기여**: 끝점 standard 응답 + E-Graph에서 같은 탐색의 no-LLM 대비 −12%, 지도 짝/홀 양쪽에서 같은 방향. weighted A* 곡선(w=2.5: 32.2 / 102.98, w=3: 30.8 / 103.68)보다도 아래(30.24 / 102.85).
- 이유: E-Graph는 **충돌 없는 구간만** 경험 경로로 쓴다. 끝점 standard는 LLM 단독 무충돌 34.7%로 가장 높다. RePE2(2.8%)는 E-Graph에서 오히려 손해.
- 즉 "어떤 프롬프트가 좋은가"는 탐색 방식에 따라 뒤집힌다: 공식 LLM-A*에서는 RePE2(적고 보수적인 경유점)가, E-Graph에서는 끝점 standard(충돌 없는 경유점)가 좋다.
- 주의: εE, w는 Llama 실험에서 고른 값. wallC 단독(15.4, upstream 집계 / 18.5 중복 제거)보다는 여전히 나쁨.

### B10. 실패 유형 분류 (`failure_taxonomy.py`, 전 응답 세트)
각 응답을 한 유형으로 분류하고, 같은 문제의 weighted A*와 비교 (op 기하평균, LLM-A* / wA*2).
| 세트 | 무충돌 | 벽 위 경유점 | 구간 1개 충돌 | 2개 이상 충돌 | 전부 필터됨 | 잘림 | 우회(≥1.3배) |
|---|---|---|---|---|---|---|---|
| GPT 원본 standard | 64 (29/21) | 597 (54/36) | 212 (47/33) | 62 (77/38) | 39 | 23 | 54 |
| GPT 끝점 standard | 348 (39/30) | 53 | 421 (56/35) | 174 (79/40) | 4 | 0 | **290** |
| RePE2 | 27 (23/25) | 125 (36/32) | 611 (40/33) | 88 (44/34) | 63 | 0 | 4 |
- **무충돌 경로도 공식 LLM-A*에서는 손해**(GPT 원본 standard: 29 대 21). 축을 따라 도는 긴 경로를 target으로 쫓기 때문. 완벽한 끝점 체인으로도 공식 LLM-A* 상한은 31.3(−9%)뿐 → 공식 탐색 구조 자체의 한계.
- 최소 경유점 수 k별: k=1(847문제)에서 RePE2 36.9 대 wA* 31.3, k=2·3에서 모든 프롬프트가 크게 나쁨(63–89 대 58–59).
- 실패 사례 그림: `exp/claude/figs/failure_cases.png` (규칙으로 고른 5개: 공식 LLM-A* 폭주 10:8 622% 대 17%; 무충돌인데 손해 19:6; 끝점 우회 체인 26:8; RePE2 "Valid"인데 모서리 관통 23:4 423% 대 26%; RePE2 벽 위 경유점 10:4). 생성: `failure_case_data.py` → `plot_failure_cases.py`(matplotlib 있는 파이썬).

### B11. 근소 실패 여부 (`near_miss.py`)
충돌하는 응답 중 경유점 하나를 한 칸 옮겨 무충돌이 되는 비율: 원본 9–14%, 끝점 16–21%, RePE2 7% (두 칸 이내 14%). 충돌 구간이 하나여도 대부분 국소 수정으로 안 고쳐진다.

### B12. wallC 실행 시간 (`wallc_win_check.py`, 지도 유형별 60개 지도)
충돌을 그때그때 검사하는 모드(공식 코드 방식): 원본 지도 wA*2 2.76 ms 대 wallC 3.05 ms (op 195 대 83), 큰 지도 scaled4 33.5 대 24.8 ms(wallC 빠름), dense·rooms·maze에서는 wallC가 2–3배 느림. 충돌표 사전 계산 모드에서는 원본 지도 0.54 대 1.95 ms.

### B8. 기타 사용자 아이디어 검증 (Llama, 오프라인)
- 축소 지도에서 A* 후 경유점 추출(`coarse_then_fine.py`): 세밀 탐색만 보면 op 28–35지만 거친 탐색 비용 포함 시 60–67, 유효율 72–97%.
- 각도 페널티(`angle_penalty.py`), 평가함수 선택만 LLM에 맡기기(`eval_choice_headroom.py`: oracle 선택 시 23.3 대 고정 규칙 28.6), 문헌 아이디어(MMP-A* 감쇠, iLLM-A* 2점 선택; `lit_quick.py`), 경유점 수 절단(`truncate_k.py`, `waypoint_count.py`).

---

## C. 의도와 구현이 어긋난 점 (요약)
1. GPT-4o mini 실험은 원 논문 GPT-3.5 행의 재현이 아니다(모델, chat API, max_tokens 8000 대 100+줄바꿈 stop).
2. "VoT" 배치는 지도 없이 트리거 문장만 넣은 것이라 진짜 시각화가 아니다. zero-shot VoT는 예시 제거와 섞여 있다.
3. Llama 상한 실험은 PyTorch 버전이 달라 프롬프트 효과만 분리할 수 없다.
4. 후보점·끝점 실험에서 "목록에서 고르기"를 의도했지만 CoT/RePE는 추론 중 새 좌표를 만든다. 끝점 batch는 예시 답 수정과 목록 추가가 동시에 바뀌었다.
5. 끝점은 공식 필터와 맞추느라 테두리 쪽 틈을 후보로 줄 수 없다.
6. RePE2는 이름과 달리 RePE에서 파생되지 않았고, 자기 평가가 실질적으로 작동하지 않았다(Valid 판정의 97%가 틀림). 예시 교체와 Round 절차의 혼동은 E1으로 해소(개선 = Round 절차).
6-1. "충돌 구간 1개 = 근소한 실패"라는 초기 해석은 틀렸다(한 칸 수정으로 고쳐지는 비율 7%).
7. 오프라인 알고리즘 변형(β, timeout, 가중치, εE 등)은 같은 데이터에서 골랐다(사후 탐색). 지도 홀짝 분할로 방향만 확인했다.
8. op는 heap pop 수라 휴리스틱 계산 비용을 반영하지 않는다. wallC는 op가 1/2이지만 실행 시간은 weighted A*의 약 4배다.
9. 스모크 일부(지도 0번 10문제)는 1000문제 결과와 방향이 달랐다. 스모크 수치는 보고서 본문에 쓰지 않는다.

## D. 근거가 부족한 부분과 필요한 실험
완료: E1(RePE2 절제, A8), E3(다른 지도 유형, A9), E5(E-Graph on GPT, B9), E7(실행 시간, B12), 실패 유형·근소 실패·자기 판정 정확도(B10, B11, A8).

| 우선 | 실험 | 목적 | 비용 |
|---|---|---|---|
| E2 | 제안 방법을 `llm_astar` 메인 경로에 넣고 CLI로 1000문제 재실행 (후보: 끝점 standard + E-Graph) | 보고서 수치가 제출 코드에서 그대로 나오는지 (TA 평가 경로) | 저장 응답 재사용, API 없음 |
| E8 | **충돌 피드백 1회**: RePE2 Round 사이에 코드가 "어느 구간이 어느 벽과 충돌"을 알려주는 2회 호출 변형 | 실패가 "제안 능력" 문제인지 "검증 능력" 문제인지 분리 (B 결과상 검증이 병목) | API 2×1000회, 실시간 필요 |
| E6 | RePE2 반복 실행 | temperature 0 단일 실행 변동성 | API 1000회 |
| E9 | Round 지시문만 / Round 예시만 | Round 절차의 두 구성요소 분리 | API 1000회 |
| E4 | 끝점 v1(예시 미수정) 1000문제 | 예시 수정 효과 분리 | API 3000회, 우선순위 낮음 |

## E. 재현 방법 (공통)
```bash
cd /home/skkai/Mobility/2026_2/ykim/assign
read -rsp 'OpenAI API key: ' OPENAI_API_KEY; echo; export OPENAI_API_KEY    # API 실험만
.conda/envs/llm-astar/bin/python scripts/run_batch_table1.py --config <config> --output <out> --prompts <p...> --zip
PYTHONPATH=. .conda/envs/llm-astar/bin/python -B analysis/<run>-analysis/analyze.py
PYTHONDONTWRITEBYTECODE=1 .conda/envs/llm-astar/bin/python exp/claude/<script>.py      # 오프라인 실험
.conda/envs/llm-astar/bin/python -B -m unittest tests/test_*.py
```
실험별 config ↔ 출력 폴더 ↔ 분석 스크립트 대응은 `docs/RUNS.md`에 있다.
