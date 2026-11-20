# Vast.ai RTX 5090 실행

## 1. 인스턴스 선택

Vast의 **PyTorch 템플릿, CUDA 12.8 이상, 호스트 CPU에 맞는 아키텍처**를 선택한다. 일반적인 x86_64 서버에서는 x86_64/AMD64 이미지를 선택한다. 첨부 화면의 CUDA 12.4 / ARM 설정을 그대로 사용하지 않는다. GPU 모델만으로 CPU 아키텍처가 결정되는 것은 아니므로 임대한 호스트 정보를 확인한다.

Vast는 RTX 50 시리즈에 CUDA 12.8 이상 및 PyTorch 2.7 이상을 요구한다. 이 프로젝트는 기존 실험과 버전을 맞춘 **torch 2.10.0+cu128, transformers 4.57.3, bf16** 조합을 설치한다. Python 3.12 권장(3.11–3.13 지원). 설치 후 실제 작은 CUDA 연산으로 드라이버 호환성을 확인한다. 호스트 NVIDIA 드라이버는 컨테이너 안의 pip로 교체할 수 없으므로 부적합한 호스트는 선택하지 않는다.

- [Vast RTX 50 공식 안내](https://github.com/vast-ai/docs/blob/main/rtx-5-series.mdx)
- [PyTorch Blackwell 지원 안내](https://pytorch.org/blog/pytorch-2-7/)
- [PyTorch CUDA 12.8 wheel 목록](https://download.pytorch.org/whl/cu128/torch/)

가중치 디스크 용량은 모델 한 벌 약 16GB이며 GPU 수만큼 디스크에 복사할 필요는 없다. 각 프로세스가 같은 파일을 읽어 각 GPU VRAM에 별도 모델을 로드한다. GPU별로 모델+KV cache가 들어가야 하므로 여러 GPU의 VRAM을 합쳐 사용하는 방식은 아니다. 모델 파일, Python 환경, 캐시를 고려해 여유 있는 디스크를 준비한다(예: 50GB 이상). CPU RAM도 동시에 로딩하는 복제본 수에 영향을 받는다.

## 2. Git clone 후 설치

GitHub 등록은 사용자가 별도로 한다. 저장소를 올린 다음:

```bash
git clone <YOUR_REPOSITORY_URL> llm-astar
cd llm-astar
bash scripts/setup_vast.sh
source .venv/bin/activate
```

Python 3.12 실행 파일을 직접 지정해야 하면:

```bash
PYTHON_BIN=python3.12 bash scripts/setup_vast.sh
```

템플릿 기본 Python 환경은 변경하지 않고 저장소 내부 `.venv`에 설치한다. `requirements.txt`는 기존 CPU/API 용도, `requirements-vast.txt`는 RTX 5090 용도다. Torch는 먼저 공식 cu128 인덱스에서 설치하므로 설치 스크립트를 사용하는 것이 권장된다. matplotlib은 추론에 필요 없으며 연구용 그림을 다시 만들 경우에만 별도 설치한다.

## 3. 모델 가중치 준비

### 이미 보유한 모델 전송

Hugging Face 형식의 폴더 전체를 복사한다. `config.json`, tokenizer 관련 파일, `model.safetensors.index.json`과 모든 safetensors shard가 필요하다. GGUF 한 파일은 이 실행기의 입력이 아니다.

로컬 또는 원래 서버에서 Vast의 SSH host/port를 사용한다:

```bash
rsync -avP -e 'ssh -p <VAST_SSH_PORT>' \
  /path/to/Meta-Llama-3-8B-Instruct/ \
  root@<VAST_SSH_HOST>:/workspace/models/Meta-Llama-3-8B-Instruct/
```

SSH 사용자명은 Vast 인스턴스 접속 정보에 맞춘다. rsync가 없다면 설치하거나 scp로 폴더 전체를 보낸다. 전송이 끝난 뒤 추론을 시작한다.

### 서버에서 직접 다운로드 — 토큰을 파일에 저장하지 않는 방법

Meta Llama 3 접근 권한이 승인된 Hugging Face read 토큰이 있다면 보통 직접 다운로드가 편하다:

```bash
python scripts/download_weights.py
```

토큰을 숨김 입력으로 받고 `hf auth login`이나 Git credential 저장을 하지 않는다. 가중치는 `models/Meta-Llama-3-8B-Instruct`에 저장한다. 기존 재현 revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`로 고정한다. 이미 `HF_TOKEN`을 환경 변수로 넣었다면 사용 후 직접 `unset HF_TOKEN`을 실행한다. 이 스크립트는 다른 사람의 기존 인증을 삭제하지 않는다.

다운로드 재실행 시 Hub의 로컬 메타데이터를 이용한다. 직접 전달한 가중치는 파일 존재·크기·JSON 메타데이터를 기록하지만 전체 가중치 SHA256 검증을 했다고 표시하지 않는다. 같은 revision이라도 다른 GPU/라이브러리 조합에서는 생성 결과가 달라질 수 있다.

[Hugging Face 공식 다운로드 안내](https://huggingface.co/docs/huggingface_hub/guides/download)

## 4. 멀티 GPU 실험 시작

SSH 연결과 분리하려면 먼저 `tmux new -s llama`를 실행한다. GPU 4개 예시:

```bash
source .venv/bin/activate
python scripts/run_multigpu.py \
  --model-path /workspace/models/Meta-Llama-3-8B-Instruct \
  --gpus 0,1,2,3 \
  --output outputs/llama3-5090-v1 \
  --zip
```

기본 다운로드 폴더를 사용하고 모든 보이는 GPU를 사용할 경우:

```bash
python scripts/run_multigpu.py --output outputs/llama3-5090-v1 --zip
```

- GPU당 독립 프로세스 1개, 독립 모델 1개. 각 프로세스는 자기 GPU 하나만 보고 `cuda:0`에 모델 전체를 배치한다. DDP나 tensor parallel/NCCL은 사용하지 않는다.
- 작업 단위는 `(지도·시작/목표, 프롬프트)`다. 기본 1,000문제 × standard/cot/repe = **3,000회 생성**, A* 및 LLM/LLM-A* **7,000개 결과 행**이다.
- 먼저 끝난 프로세스가 다음 작업을 받아 긴 RePE 응답에 따른 GPU 불균형을 줄인다. 마지막 몇 작업의 길이 차이에 따른 유휴 시간은 남을 수 있다.
- 한 GPU에서 입력을 한 개씩 처리한다. 모델을 여러 GPU에 쪼개거나 여러 요청을 배치 처리하지 않으며, 기존 원문 프롬프트·greedy·max_new_tokens=8000·upstream 파싱/탐색 로직을 유지한다.
- `--prompts standard`처럼 일부 프롬프트만 지정하거나 `--limit 100`으로 앞의 100문제를 선택할 수 있다. 설정을 바꿀 때는 새 출력 폴더를 사용한다.
- `CUDA_VISIBLE_DEVICES`가 이미 지정돼 있으면 그 범위만 사용한다. `--gpus`도 해당 변수의 ID/UUID 토큰 중에서 지정한다.
- 기존 OpenAI API 모델은 종전 `python -m llm_astar --config ...`로 실행한다. API 요청을 GPU에 나누는 것은 지원하지 않는다.
- 기존 오프라인 개선 알고리즘은 `exp/gpt`, `exp/claude`에 유지한다. 새 실행기는 공식 기준의 응답과 Table 1 결과를 생성하며, 61개 개선 조건을 자동 실행하는 옵션은 아니다. 연구 스크립트가 참조하는 원본 결과 경로는 해당 스크립트에 맞춰 준비해야 한다.

## 5. 중단 및 이어서 실행

Ctrl-C 또는 SIGTERM 시 완료된 작업을 모아 결과표와 ZIP을 만든다. 같은 명령을 다시 실행하면 완료된 입력은 건너뛰고 남은 것만 실행한다. 작업 도중 중단된 생성 한 건은 다시 할 수 있다. 응답 저장 직후 중단됐으면 캐시를 재사용한다.

GPU 개수/ID를 바꾸어도 같은 출력 폴더에서 이어갈 수 있다. 데이터, 프롬프트 선택, 모델 설정/메타데이터, 코드 또는 주요 라이브러리 버전이 바뀌면 섞이지 않도록 거부한다. 기존 V1/V2 폴더를 재사용하지 말고 새 멀티 GPU 폴더로 시작한다.

OOM/프로세스 오류가 나면 전체 실행을 실패로 기록하고 남은 작업은 다음 실행에서 처리한다. 고장 난 GPU를 제외하고 같은 폴더로 재실행할 수 있다. 종료가 불가능한 SIGKILL/호스트 종료 시에는 마지막 ZIP 갱신은 보장되지 않지만 개별 작업/응답 파일은 보존된다.

## 6. 결과 확인과 ZIP 다운로드

```text
outputs/llama3-5090-v1/
  status.json         # running/completed/interrupted/failed, 완료 작업 수
  manifest.json       # 설정·원본 데이터/코드 지문·모델 메타데이터
  input.json          # 입력 지도 데이터 사본
  results.jsonl       # 합쳐진 경로·성공 여부·탐색량·길이
  summary.json
  table1.csv
  responses/          # 실제 프롬프트, 원문 응답, 토큰 수, 생성 시간
  jobs/               # 중단 복구용 입력별 결과
  logs/               # GPU별 로그
outputs/llama3-5090-v1.zip
```

종료 시 결과표를 합친다. 실행 중에는 `status.json`, `jobs/`, GPU 로그를 본다. 중단된 결과표는 완료된 입력만의 통계이며 최종 Table 1로 해석하면 안 된다. `status.json`의 완료 상태와 3,000작업/7,000행을 확인한다.

이미 저장된 결과를 추론 없이 다시 압축:

```bash
python scripts/run_multigpu.py --output outputs/llama3-5090-v1 --zip-only
```

실행 중인 출력 폴더는 잠금 때문에 압축하지 않는다. ZIP에는 지정한 결과 파일·입력 사본·응답·로그만 포함하며 가중치/가상환경/토큰/심볼릭 링크는 포함하지 않는다. SIGKILL 직후에는 재실행해 병합을 마친 다음 압축하는 것이 좋다.

서버 파일 브라우저에서 ZIP 하나만 내려받거나 로컬에서:

```bash
scp -P <VAST_SSH_PORT> root@<VAST_SSH_HOST>:/workspace/llm-astar/outputs/llama3-5090-v1.zip .
```

저장소 위치에 맞게 경로를 바꾼다. ZIP 다운로드가 완료되고 내용이 확인된 뒤 서버를 종료한다.

## 7. GitHub에 올릴 파일

`.gitignore`에서 체크포인트, 응답/출력, 환경, 인증 파일, 공식 저장소의 별도 clone, 과제 PDF를 제외한다. 실행에 필요한 공식 데이터/프롬프트/라이선스는 포함돼 있어 `upstream/` 폴더 없이 실행된다.

코드만 모은 작은 패키지도 만들 수 있다:

```bash
python scripts/package_source.py
```

`dist/llm-astar-source.zip`의 `llm-astar/` 내용이 GitHub에 올릴 코드다. 모델과 기존 실험 출력은 들어 있지 않다. 사용자가 저장소를 만든 뒤 해당 폴더에서 git init/add/commit/push하거나, 현재 폴더에서 `.gitignore`를 적용해 직접 등록하면 된다. 공유 서버의 다른 사람 GitHub 인증은 자동으로 사용하지 않는다.

## 검증 범위

프로세스 두 개 배분, 작업 중단 상태에서 GPU 수 변경 재개, 동일 응답 동시 저장, worker 실패 처리, ZIP 내용/제외 규칙을 CPU fixture로 회귀 테스트한다:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

이 테스트는 실제 LLM 추론을 하지 않고 임시 결과를 자동 삭제한다. 지속적으로 유지할 테스트 코드만 남긴다. RTX 5090 실장 환경의 실제 Llama 추론은 이 서버에서 실행하지 않았으며, `setup_vast.sh`의 GPU 사전 점검 후 실제 실험을 시작해야 한다.

## 8. 8방향 이동 + 명시적 장벽 좌표 실험

`configs/llama3_8way_walls.json`은 `prompt_variant=eight_way_explicit_walls`를 사용한다.
기존 지시문 형식을 유지하면서 `You can move horizontally, vertically, and diagonally.`와 `Barriers cannot be crossed.` 두 문장을 추가한다. 실제 문제와 few-shot 예제의 장벽 목록 모두를 `B1: horizontal barrier at y = ..., from (...) to (...)` 형식으로 변환한다. 이동 후보 열거, 비용, 별도 충돌 규칙 설명은 추가하지 않는다.
원본 예제 응답/CoT 설명, 생성 설정, 필터, 탐색 로직은 유지한다. 따라서 예제 응답에 있던 오류까지 교정한 실험은 아니다. 두 변경을 동시에 적용하므로 개별 변경의 효과를 분리하는 실험도 아니다.
원본 프롬프트는 기본 설정에서 그대로 사용된다. 새 설정은 논문 원본 재현과 구별되는 프롬프트 개선 실험이다.

Standard 1,000문제로 먼저 비교하는 실행 명령:

```bash
tmux new -s llama-8way
source .venv/bin/activate
python scripts/run_multigpu.py \
  --config configs/llama3_8way_walls.json \
  --model-path /workspace/models/Meta-Llama-3-8B-Instruct \
  --gpus all \
  --prompts standard \
  --output outputs/llama3-8way-walls-standard-v1 \
  --zip
```

`Ctrl-B` 다음 `D`로 tmux에서 빠져나오며, `tmux attach -t llama-8way`로 다시 연결한다.
중단 후 같은 명령으로 재개한다. 완료 시 1,000회 생성 / 3,000개 결과 행과 `status.json`의 completed 상태를 확인한다.
다운로드할 파일은 `outputs/llama3-8way-walls-standard-v1.zip`이다.
세 프롬프트 전체를 비교하려면 `--prompts standard cot repe`와 **새 출력 폴더** `outputs/llama3-8way-walls-all-v1`을 사용한다(3,000회 생성 / 7,000행).
가중치를 기본 다운로드 경로에 넣었다면 `--model-path`를 생략한다.
동일 서버에서 원본 대조군을 새로 실행할 때는 `--config configs/llama3_table1_upstream.json --prompts standard --output outputs/llama3-original-standard-v1`을 사용한다.

새 Standard 프롬프트 전체 예시는 `examples/prompt_8way_walls.txt`에서 확인할 수 있다.


## 9. 원본 장벽 좌표 + 8방향 이동 문장만 추가

`configs/llama3_8way_only.json`은 원본 프롬프트에 다음 한 문장만 추가한다:

```text
You can move horizontally, vertically, and diagonally.
```

원본 장벽 설명 바로 다음에 한 번 삽입한다. `Horizontal Barriers: [[y, x_start, x_end], ...]`와 `Vertical Barriers: [[x, y_start, y_end], ...]` 표기, few-shot 예제와 응답, CoT·RePE 지시는 모두 원본과 같다. `Barriers cannot be crossed.` 문장은 추가하지 않는다. 기존 명시적 장벽 실험과 원본 설정은 그대로 유지한다.

```bash
python3 scripts/run_multigpu.py \
  --config configs/llama3_8way_only.json \
  --model-path /workspace/Meta-Llama-3-8B-Instruct \
  --gpus all \
  --prompts standard cot repe \
  --output outputs/llama3-8way-only-v1 \
  --zip
```

총 3,000회 생성이며, 결과는 `outputs/llama3-8way-only-v1.zip`에 저장된다. Standard만 먼저 확인하려면 `--prompts standard --output outputs/llama3-8way-only-standard-v1`을 사용한다. 원본 대조군도 같은 GPU·소프트웨어에서 실행하면 생성 환경 차이를 통제할 수 있다.

Standard 프롬프트 전문은 `examples/prompt_8way_only.txt`에서 확인할 수 있다.
