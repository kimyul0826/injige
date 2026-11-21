# GPT 오프라인 실험 전용 폴더

- 결과 설명: [REPORT.md](REPORT.md)
- 전체 수치: [summary.csv](summary.csv), [summary.json](summary.json)
- 주요 비교 그림: [tradeoffs.png](tradeoffs.png), [timing.png](timing.png)
- 주 실험: `experiment.py`, `results.jsonl`, `manifest.json`, `run.log`
- 추가 실험: `supplement.py`, `supplement-results.jsonl`, `supplement.log`
- 예산 복귀: `budget.py`, `budget-results.jsonl`, `budget.log`
- 사후 조합: `refinement.py`, `refinement-results.jsonl`, `refinement.log`
- 집계: `summarize.py`, `selections.json`, `uncertainty.json`
- 공정한 캐시 조건의 CPU 시간: `benchmark.py`, `timing-results.jsonl`, `timing-summary.json`
- 검증: `verification.json`, `supplement-verification.json`, `budget-verification.json`, `quality-verification.json`, `repeat-verification.json`
- 원본 탐색 코드의 복사본: `reference_search.py`

원본 데이터/응답은 읽기 전용으로 참조한다. 모델 실행·API 호출·패키지 설치는 없다. 코드와 결과 모두 이 폴더에만 생성한다. 기존 기본 추론 실행기에는 통합하지 않았으므로 원래 추론 명령은 바뀌지 않는다.

보고서의 순차 재실행 명령을 사용할 수 있다. matplotlib 그림 생성은 현재 system Python의 `.deps`를 사용한다:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.deps MPLCONFIGDIR="$PWD/exp/gpt/.mplconfig" python -B exp/gpt/plot_results.py
```

선정 후보를 기존 코드에서 호출하려면 `planner.py`의 `search(query, nodes, variant)`를 사용한다. `nodes`는 기존 응답에서 파싱한 좌표 목록이다. 기본 후보는 `prune_octile_grid_smooth`; 경로 악화 제한 후보는 `bounded_0.1`이다. 비LLM 대조군은 `weighted_astar_1.5`, `weighted_astar_1.1`, `optimal_deep_tie`이며 이때 nodes를 생략할 수 있다. 이 진입점도 모델 호출이나 파일 생성을 하지 않는다.

`sequential_repair`는 도달 불가능한 중간 목표를 발견하면 일반 A*로 복귀하며, 실패에 사용한 계산도 합산한다. 실패 사례 기록은 `sequential-failure.json`, 재개 로그는 `supplement-resume.log`이다. 일회성 초기 코드/오류 로그는 배포 정리 때 제거했다. 전체 재실행은 기본적으로 새 결과를 만들고, 보조 실험만 동일 코드로 중단된 경우에는 `supplement.py --resume`을 사용할 수 있다. 최종 집계는 수정 후 전체 179,000건을 검증했다.
