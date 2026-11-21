# 실행 기록 ↔ 설정 ↔ 분석 대응표

`outputs/`는 용량 때문에 Git에서 제외된다. 각 실행 폴더에는 `manifest.json`(입력 해시, 설정, 코드 해시),
`responses/`(모델 원문 응답), `results.jsonl`(문제별 결과), `summary.json`, `table1.csv`가 있다.
분석 스크립트는 `analysis/`에 Git으로 관리되는 복사본이 있다.

## LLM 추론 실행
| 실행 폴더 (`outputs/`) | 모델 | config | `--prompts` | 문제 | 분석 |
|---|---|---|---|---|---|
| `llama3-table1-upstream-v2` | Llama-3-8B | `llama3_table1_upstream.json` | standard cot repe | 1000 | `scripts/verify_table1.py` |
| `llama3-table1` | Llama-3-8B | `llama3_table1.json` (초기 버전) | standard cot repe | 1000 | – |
| `llama3-diagonal-barriers-v3*` | Llama-3-8B | `llama3_8way_walls.json` | standard / cot repe | 1000 | `analysis/llama3-diagonal-barriers-v3-analysis/` |
| `llama3-8way-only-v1` (zip) | Llama-3-8B | `llama3_8way_only.json` | standard cot repe | 1000 | `analysis/llama3-8way-only-v1-analysis/` |
| `llama3-waypoint-limit-v1` (zip) | Llama-3-8B | `llama3_waypoint_limit.json` | standard cot repe | 1000 | `analysis/llama3-waypoint-limit-v1-analysis/` |
| `gpt4omini-table1-upstream-batch-v1` | GPT-4o mini | `gpt4omini_table1_upstream.json` | standard cot repe | 1000 | `analysis/gpt4omini-table1-upstream-analysis/` |
| `gpt4omini-waypoint-limit-batch-v1` | GPT-4o mini | `gpt4omini_waypoint_limit.json` | standard cot repe | 1000 | `analysis/gpt4omini-waypoint-limit-analysis/` |
| `gpt4omini-cot-vot-batch-v1` | GPT-4o mini | `gpt4omini_cot_vot.json` | cot | 1000 | `analysis/gpt4omini-cot-vot-analysis/` |
| `gpt4omini-vot-waypoint-limit-batch-v1` | GPT-4o mini | `gpt4omini_vot_waypoint_limit.json` | vot | 1000 | `analysis/gpt4omini-vot-waypoint-limit-analysis/` |
| `gpt4omini-barrier-endpoints-demos-batch-v1` | GPT-4o mini | `gpt4omini_barrier_endpoints_demos.json` | standard cot repe | 1000 | `analysis/gpt4omini-barrier-endpoints-demos-analysis/` |
| `gpt4omini-repe2-batch-v1` | GPT-4o mini | `gpt4omini_repe2.json` | repe2 | 1000 | `analysis/gpt4omini-repe2-analysis/` |
| `gpt4omini-repe2_noround-batch-v1` (E1-a) | GPT-4o mini | `gpt4omini_repe2.json` | repe2_noround | 1000 | `analysis/gpt4omini-repe2-analysis/ablation_e1.py` |
| `gpt4omini-repe2_origdemos-batch-v1` (E1-b) | GPT-4o mini | `gpt4omini_repe2.json` | repe2_origdemos | 1000 | 같음 |
| `gpt4omini-e3-bench-batch-v1` (E3) | GPT-4o mini | `gpt4omini_table1_upstream.json`, `--input data/bench_e3.json` | standard repe2 | 500 (5개 유형) | `analysis/e3-bench/` |
| `gpt4omini-e3-bench-endpoint-batch-v1` (E3) | GPT-4o mini | `gpt4omini_barrier_endpoints_demos.json`, `--input data/bench_e3.json` | standard | 500 | 같음 |
| `llama3-repe2-v1` (예정, Vast) | Llama-3-8B | `llama3_table1_upstream.json` | repe2 | 1000 | `scripts/run_vast_llama_extra.sh` |
| `llama3-endpoint-demos-standard-v1` (예정, Vast) | Llama-3-8B | `llama3_barrier_endpoints_demos.json` | standard | 1000 | 같음 |
| `llama3-repe2_origdemos-v1` (예정, Vast) | Llama-3-8B | `llama3_table1_upstream.json` | repe2_origdemos | 1000 | 같음 |
| `gpt4omini-table1-upstream-v1` | GPT-4o mini | `gpt4omini_table1_upstream.json` | standard cot repe | 스모크 (실시간 API) | batch의 `--reuse-cache` 기본값 |
| `gpt4omini-vot-zeroshot-smoke-v1` | GPT-4o mini | `gpt4omini_vot_zeroshot.json` | vot | 10 (지도 0번) | – |
| `gpt4omini-vot-waypoint-limit-smoke-v1` | GPT-4o mini | `gpt4omini_vot_waypoint_limit.json` | vot | 10 (지도 0번) | – |
| `gpt4omini-vot-map-smoke-v1` | GPT-4o mini | `gpt4omini_vot_map.json` | vot | 9/10 (중단) | – |
| `gpt4omini-vot-map-waypoint-limit-smoke-v1` | GPT-4o mini | `gpt4omini_vot_map_waypoint_limit.json` | vot | 10 (지도 0번) | – |
| `gpt4omini-vot-coarse-waypoint-limit-smoke-v1` | GPT-4o mini | `gpt4omini_vot_coarse_waypoint_limit.json` | vot | diverse10 | – |
| `gpt4omini-candidate-waypoints-smoke-v1` | GPT-4o mini | `gpt4omini_candidate_waypoints.json` | standard cot repe | diverse10 | `exp/claude/analyze_candidate_smoke.py` |
| `gpt4omini-barrier-endpoints-smoke-v1` | GPT-4o mini | `gpt4omini_barrier_endpoints.json` | standard cot repe | diverse10 | `exp/claude/endpoint_adherence.py` |
| `gpt4omini-barrier-endpoints-demos-smoke-v1` | GPT-4o mini | `gpt4omini_barrier_endpoints_demos.json` | standard cot repe | diverse10 | `exp/claude/endpoint_adherence.py` |
| `gpt4omini-repe2-smoke-v1` | GPT-4o mini | `gpt4omini_repe2.json` | repe2 | diverse10 | – |

diverse10 = `data/smoke_diverse10.json` (서로 다른 지도 10개, 문제 ID 유지). 스모크 수치는 보고서 본문에 쓰지 않는다.

명령 형식:
```bash
.conda/envs/llm-astar/bin/python scripts/run_batch_table1.py --config configs/<config> --output outputs/<run> --prompts <...> --zip
.conda/envs/llm-astar/bin/python scripts/run_api_table1.py --config configs/<config> --input data/smoke_diverse10.json --prompts <...> --output outputs/<run>
```

## 오프라인 실험 (`exp/claude/`, 새 추론 없음)
| 스크립트 | 입력 응답 | 결과 (`exp/claude/results/`) | 문서 |
|---|---|---|---|
| `offline_improve.py` → `analyze.py`, `gate_analysis.py` | Llama v2 | `per_sample.jsonl.gz`, `summary_*.csv`, `bootstrap.json`, `gate.json` | `exp/claude/REPORT.md` |
| `logic_variants.py` → `logic_analyze.py` | Llama v2 | `logic_*` | `exp/claude/LOGIC_REPORT.md` |
| `general_bench.py`, `union_test.py`, `wall_component_h.py`, `closed_form_*.py` | Llama v2 / 없음 | `general_bench*.json`, `wall_component_h.json` | `exp/claude/GENERAL_REPORT.md` |
| `fairness_check.py`, `cost_check*.py`, `wallc_win_check.py`, `dwa_check.py` | 없음 | 같은 이름 `.json` | `docs/EXPERIMENTS.md` B1, B5 |
| `random_waypoints.py`, `per_map_compare.py`, `topology_*.py`, `heuristic_quality.py`, `waypoint_*.py` | Llama v2 | 같은 이름 `.json` | B3 |
| `combine_llm_wallc.py` (`LLM_SRC`, `TAG`) | Llama / GPT | `combine_llm_wallc*.json` | B6 |
| `post_inference_gpt.py` | GPT 전 세트 | `post_inference_gpt.json/.txt` | B7 |
| `egraph_gpt.py` | GPT 전 세트 | `egraph_gpt.json/.txt` | B9 |
| `failure_taxonomy.py` → `failure_case_data.py` → `plot_failure_cases.py`* | 전 세트 | `failure_taxonomy.*`, `failure_cases.json`, `figs/failure_cases.png`, `report/fig_failures.png` | B10 |
| `near_miss.py` | GPT 전 세트 | `near_miss.txt` | B11 |
| `repe2_selfcheck.py` | RePE2, E1-b | `repe2_selfcheck.txt` | A8 |
| `candidate_points.py`, `endpoint_ceiling.py`, `endpoint_llmastar_ceiling.py` | 없음 / GPT | 같은 이름 `.json` | A6, A7 |
| `demo_representativeness.py`, `build_repe2_demos.py`, `render_repe2_demos.py` | 없음 | `demo_representativeness.txt`, `repe2_demo*.json` | A8 |
| `exp/gpt/*` (이전 담당 GPT의 오프라인 실험) | Llama v2 | `exp/gpt/` | `exp/gpt/REPORT.md` |

\* `plot_*.py`는 matplotlib가 필요함. 이 서버에서는 `/home/skkai/anaconda3/bin/python3`로 실행(계산 단계는 `.conda/envs/llm-astar`).
