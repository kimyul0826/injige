새 프롬프트 실험: 8방향 이동과 장벽 양 끝 좌표를 명시하는 `configs/llama3_8way_walls.json`을 추가했습니다. 실행 명령은 [Vast 가이드 §8](README_VAST.md#8-8방향-이동--명시적-장벽-좌표-실험)을 확인하세요.

# LLM-A* 과제 구현

**Vast.ai RTX 5090 / 멀티 GPU 실행은 [README_VAST.md](README_VAST.md)를 참고하세요.**

```bash
bash scripts/setup_vast.sh
source .venv/bin/activate
python scripts/run_multigpu.py --model-path /workspace/models/Meta-Llama-3-8B-Instruct --gpus 0,1,2,3 --output outputs/llama3-5090-v1 --zip
```

GPU마다 모델 1개를 로드하고 입력을 동적으로 분배합니다. 완료된 입력은 중단 후 GPU 수가 달라져도 재사용합니다. 결과 ZIP만 내려받아 원문 응답·경로·집계·로그를 확인할 수 있습니다. 체크포인트는 Git에 포함하지 않으며 외부 볼륨에서도 읽을 수 있습니다.

GitHub 업로드용 코드 패키지: `python scripts/package_source.py` → `dist/llm-astar-source.zip`.
기존 연구 결과는 로컬에 보존하지만 Git/코드 ZIP에서 제외합니다. `exp/gpt`, `exp/claude`의 개선 연구 코드는 포함하며 원본 추론 결과는 별도로 준비해야 합니다.

아래는 기존 단일 실행/API 사용법과 연구 배경입니다. 특정 서버 경로·가중치 보유 상태는 과거 로컬 환경의 설명이며 Git clone에는 적용되지 않습니다.


논문 Algorithm 1과 공식 구현을 바탕으로 A*, LLM 단독, LLM-A*를 비교합니다.
GPT-3.5 Turbo, GPT-4o mini, Llama 3 8B Instruct는 JSON 설정만 바꾸어 사용합니다.
공식 Few-Shot(standard), CoT, RePE 프롬프트와 100개 지도 × 10개 시작/목표의
1,000문제 데이터셋을 포함합니다. 이것은 원 논문 재현용 구현이며, 과제에서 요구하는
독자적인 개선 알고리즘과 보고서는 별도 작업입니다.

## 실행 환경

현재 경로는 `/home/skkai/Mobility/2026_2/ykim/assign`입니다.
상위 폴더 이름을 변경하여 경로의 `:`를 제거했습니다.
Llama 실제 실험은 전용 Conda 환경을 사용하며, 자세한 준비·실행·사후 검증 절차는
[README_LLAMA.md](README_LLAMA.md)를 따르세요.

```bash
source scripts/activate_llama.sh
python -m llm_astar --help
```

전용 환경에서는 `.deps`를 등록하는 `run.py` 대신 `python -m llm_astar` 또는
`python scripts/run_llama_table1.py`를 사용합니다.
별도 실행 진입점인 `run.py`는 폴더 내부 `.deps`의 API 의존성을 사용합니다.
모델 가중치와 캐시, 결과는 모두 이 프로젝트 폴더 안에 저장합니다.
API 키는 환경변수로 읽고 결과 파일에 저장하지 않습니다.

## 모델 교체

| 설정 파일 | 모델 / 방식 |
|---|---|
| `configs/gpt35.json` | `gpt-3.5-turbo`, Chat Completions |
| `configs/gpt4omini.json` | `gpt-4o-mini`, Chat Completions |
| `configs/llama3.json` | `meta-llama/Meta-Llama-3-8B-Instruct`, 로컬 Transformers, bfloat16 |
| `configs/llama3_server.json` | 같은 Llama 모델의 OpenAI 호환 서버 |
| `configs/gpt35_upstream.json` | 공식 코드의 `gpt-3.5-turbo-instruct`, legacy Completions |

### GPT-3.5 Turbo / GPT-4o mini

`OPENAI_API_KEY`를 터미널 환경에 설정한 뒤 실행합니다. 키를 코드나 JSON에 넣지 마세요.
모델 접근 가능 여부는 계정/API에 따라 달라지며, 접근 실패 시 다른 모델로 자동 대체하지 않습니다.

```bash
python run.py --methods astar llm llm-astar --config configs/gpt35.json --output outputs/gpt35
python run.py --methods astar llm llm-astar --config configs/gpt4omini.json --output outputs/gpt4omini
```

### 로컬 Llama 3 8B

먼저 Meta 모델 접근 권한을 승인받고 필요하면 `HF_TOKEN` 환경변수를 설정합니다.
현재 서버에는 가중치 다운로드가 완료되어 있습니다. 로컬 가중치를 사용하는 전체 실행은 README_LLAMA.md를 따르세요.
16비트 가중치 자체는 약 16 GB이며 실행에는 KV cache 등 추가 메모리가 필요합니다.
`device_map: "auto"`로 사용 가능한 장치에 배치하고, 특정 GPU는 `CUDA_VISIBLE_DEVICES`로 선택할 수 있습니다.
하드웨어에 따라 PyTorch 설치 방식은 조정해야 합니다.

```bash
mkdir -p .tmp .cache/pip
TMPDIR="$PWD/.tmp" python -m pip install --target .deps --cache-dir .cache/pip -r requirements-local.txt
python run.py --methods astar llm llm-astar --config configs/llama3.json --output outputs/llama3
```

이미 실행 중인 OpenAI 호환 Llama 서버를 사용할 수도 있습니다.
`configs/llama3_server.json`의 `base_url`과 `model`을 서버에 맞추고 `LLAMA_API_KEY`를 설정합니다.
인증이 없는 로컬 서버라도 SDK용으로 `export LLAMA_API_KEY=local`을 설정합니다.

```bash
python run.py --methods astar llm llm-astar --config configs/llama3_server.json --output outputs/llama3-server
```

원본 Llama 프롬프트의 역할/예시를 chat messages로 변환해 전달합니다.
서버의 chat template, precision, token limit에 따라 로컬 실행과 달라질 수 있습니다.

## Table 1 실험

먼저 `--limit 3`으로 작은 실험을 확인합니다. 기본 입력은 예제 1개입니다.
공식 전체 데이터셋은 `--input data/environment_50_30.json`으로 선택합니다.

```bash
python run.py \
  --input data/environment_50_30.json \
  --config configs/gpt35.json \
  --config configs/gpt4omini.json \
  --config configs/llama3.json \
  --methods astar llm llm-astar \
  --prompts standard cot repe \
  --limit 3 \
  --output outputs/table1-small
```

전체 실험은 `--limit 3`을 제거하고 `--output outputs/table1-full`을 사용합니다.
세 모델 × 세 프롬프트 × 1,000문제로, 캐시가 없으면 총 9,000번의 생성 호출입니다.
한 응답을 LLM 단독과 LLM-A*에서 함께 사용합니다. A* 기준값은 항상 함께 계산합니다.
`--offset`으로 시작 위치를 지정할 수 있으며, 서로 다른 설정/범위에는 다른 출력 폴더를 사용합니다.
같은 명령을 재실행하면 이미 받은 응답은 캐시에서 읽고 탐색을 다시 계산합니다.
독립적인 새 생성 실험은 다른 `--cache-dir`와 `--output`을 지정하세요.

결과 파일:

- `manifest.json`: 데이터 SHA-256, 모델 설정, 문제 목록, 탐색 모드
- `results.jsonl`: 문제별 경로·유효성·operation·storage·length·시간·토큰 사용량
- `table1.csv`, `summary.json`: 방법/모델/프롬프트별 집계
- `outputs/cache/*.json`: 정확한 입력 프롬프트·원문 응답·생성 설정·종료 사유

Operation/storage/path-length ratio는 A* 대비 비율의 **기하평균 × 100**입니다.
Valid path ratio는 전체 문제 중 유효 경로 비율입니다. 비율은 두 방법 모두 유효한 쌍으로
계산하며 `paired_valid_samples`에 분모를 기록합니다. 경로 길이가 0인 쌍은 길이 비율에서 제외합니다.
실패가 많은 방법의 조건부 평균을 전체 성능처럼 해석하지 마세요.
LLM 단독은 연속 선분 경로이므로 8방향 격자 A*보다 짧아 길이 비율이 100 미만일 수 있습니다.

`operation`은 heap에서 꺼낸 횟수, `storage`는 g 딕셔너리 크기입니다.
LLM 생성 비용이나 실제 RAM/GPU 메모리를 의미하지 않습니다.
`generation_seconds`는 캐시 최초 생성 당시 시간이며, `cache_hit`로 재사용 여부를 구분합니다.
API 실패는 실험을 중단하고 성공한 응답은 캐시에 남깁니다.
파싱 실패는 기록하며 LLM 단독은 실패, LLM-A*는 시작/목표만 가진 target 목록으로 탐색합니다.

## 알고리즘 및 공식 구현과의 차이

A*는 `f(s) = g(s) + distance(s, goal)`입니다.
LLM-A*는 `f(s) = g(s) + distance(s, goal) + distance(s, current_target)`입니다.
LLM이 한 번 생성한 waypoint 목록을 따라, 이웃이 현재 target에 도달하면 다음 target으로
옮기고 OPEN의 우선순위를 전부 다시 계산합니다. 이 추가 휴리스틱은 최단경로를 보장하지 않습니다.

기본 `--search-mode upstream`은 공식 코드의 중복 heap pop, 막힌 이웃을 포함하는 storage,
waypoint 필터와 target 전환 순서를 유지합니다. 공식 코드와 수치를 비교할 때 사용합니다.
`--search-mode paper`는 논문 의사코드에 가깝게 CLOSED 중복을 건너뛰고 실제 충돌 검사로
waypoint를 필터링합니다. 두 모드의 수치는 서로 다른 실험으로 보고해야 합니다.

논문은 GPT-3.5-TURBO라고 기재했지만 공식 경로 생성 코드는 `gpt-3.5-turbo-instruct`를 사용합니다.
`gpt35.json`은 사용자가 요청한 chat 모델이고 `gpt35_upstream.json`은 공식 코드의
completion API와 newline stop 설정을 보존합니다. 원문의 Table 1 수치와 정확히 같다고 주장하지 않습니다.
자세한 변경점과 출처는 [THIRD_PARTY.md](THIRD_PARTY.md)에 있습니다.

## 입력·출력 및 Python 사용

입력은 공식 저장소의 JSON query 또는 `start_goal`을 포함한 지도 배열입니다.
`examples/query.json`을 참고하세요. `range_x`, `range_y`의 상한은 exclusive입니다.
예를 들어 `[0,51]`의 외벽은 0과 50이고 이동 가능한 정수 좌표는 1~49입니다.
수평 장벽 `[y,x_start,x_end]`, 수직 장벽 `[x,y_start,y_end]`는 끝점을 포함해 충돌 판정합니다.
시작/목표는 내부의 장애물 없는 정수 좌표여야 합니다. 도달 불가능하면 `valid:false`와 빈 경로를 반환합니다.

```python
import json
from llm_astar import AStar, LLMAStar
from llm_astar.models import Model

query = json.load(open('examples/query.json'))
result = AStar().searching(query)
# API 패키지가 Python 경로에 있어야 합니다. CLI에서는 run.py가 .deps를 등록합니다.
model = Model(json.load(open('configs/gpt4omini.json')))
result = LLMAStar(model, prompt='standard').searching(query)
```

기존 결과 키 `operation`, `storage`, `length`, `llm_output`을 유지하며 `path`, `valid` 등을 추가합니다.
자연어 query 파싱과 PNG 자동 생성은 지원하지 않습니다. 지도별 수정 없이 동일 JSON 형식을 처리합니다.
