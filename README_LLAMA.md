> 이 문서는 기존 GB10/Conda 단일 GPU 환경의 기록입니다. RTX 5090/Vast 멀티 GPU 실행은 [README_VAST.md](README_VAST.md)를 따르세요. 현재 코드로는 과거 완료 폴더를 재개하지 말고 새 출력 폴더를 사용하세요.

# Llama 3 8B Instruct 실제 Table 1 검증

작업 경로: `/home/skkai/Mobility/2026_2/ykim/assign`.
Conda 환경과 가중치, 캐시, 결과는 이 폴더 안에 둡니다. 심볼릭 링크는 사용하지 않습니다.
이 문서는 사용자가 직접 추론을 시작하고, 종료 후 결과를 검증하는 절차입니다.

## 1. Conda 환경 활성화

```bash
cd /home/skkai/Mobility/2026_2/ykim/assign
source scripts/activate_llama.sh
```

Python 3.12, PyTorch 2.10.0 CUDA 13.0, Transformers 4.57.3,
Accelerate 1.12.0을 사용합니다. 이 실행에서는 이전 `.deps`를 사용하지 않습니다.
현재 하드웨어는 NVIDIA GB10이며 bfloat16을 사용합니다.

## 2. 인증 및 실제 가중치 다운로드

공식 모델: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
Meta 모델 접근 권한이 승인된 계정의 read 토큰이 필요합니다.
현재 공식 가중치 다운로드와 SHA-256 검증이 완료되었고, 프로젝트 토큰 파일은 삭제되었습니다.
이미 준비된 이 서버에서는 2단계를 건너뛰고 3단계 추론 명령을 실행하면 됩니다.
아래 인증·다운로드 명령은 추후 다시 다운로드할 때만 필요합니다.
다음 명령은 토큰을 화면에 표시하지 않고 입력받으며, 모델 접근을 확인한 뒤
현재 폴더의 `.cache/huggingface/token`에만 저장합니다.

```bash
python scripts/hf_auth_local.py
python scripts/download_llama3.py
```

고정 revision: `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`.
가중치: `models/Meta-Llama-3-8B-Instruct/` 아래 약 16.1GB safetensors.
다운로드 후 파일 크기와 SHA-256을 검증하고 `models/llama3-verified.json`을 생성합니다.
이 파일이 없으면 실험 실행기가 추론을 시작하지 않습니다.
인증을 설정한 후 Codex에 알려주면 다운로드·검증까지 맡길 수 있습니다.

## 3. 사용자가 추론 시작

논문 Table 1의 **Llama 부분**을 실행합니다. GPT 행은 이 실행에 포함하지 않습니다.
공식 1,000문제 × standard/CoT/RePE = **3,000번 생성**하며,
각 응답으로 LLM 단독과 LLM-A*를 평가합니다. A*까지 총 **7,000개 결과 행**입니다.

```bash
python scripts/run_llama_table1.py
```

이 명령은 모델·토크나이저를 로컬에서만 읽으며 추론 중 다운로드하지 않습니다.
기본 결과 폴더는 `outputs/llama3-table1-upstream-v2`입니다. 터미널에서 모든 추론 로그를 따라가지 않아도
상세 로그와 상태 파일이 저장됩니다. 종료 시 성공/실패를 알려줍니다.
세션을 닫아도 실행을 유지하려면:

```bash
mkdir -p outputs/llama3-table1-upstream-v2
nohup python -u scripts/run_llama_table1.py \
  > outputs/llama3-table1-upstream-v2/launcher.log 2>&1 &
echo $!
```

기본 생성 설정은 공식 코드처럼 greedy, bfloat16, max_new_tokens=8000입니다.
8,192 context 한도를 넘지 않도록 입력 길이만큼 출력 상한을 줄이며 실제 상한은 응답에 기록합니다.
공식 `upstream` 탐색 모드와 원본 프롬프트를 사용합니다.
같은 출력 폴더에서 재실행하면 저장된 응답을 재사용해 누락된 생성부터 진행합니다.
탐색과 집계는 다시 계산합니다. 코드나 설정이 바뀌면 새 출력 폴더를 지정해야 합니다.

## 4. 종료 후 결과 확인 및 검증

```bash
cat outputs/llama3-table1-upstream-v2/status.json
python scripts/verify_table1.py outputs/llama3-table1-upstream-v2
```

이 검증 명령은 모델을 로드하지 않습니다. 결과 누락·중복, synthetic 응답 여부,
원문 응답 캐시, 경로 충돌·길이, 집계 일치 여부를 검사하고 논문 Llama 행과의 차이를 계산합니다.
논문 수치와의 차이만으로 실패 판정하지는 않습니다. 차이는 별도 분석 대상입니다.

완료 후 Codex에 **“Llama 추론 끝났으니 결과 검증해줘”**라고 알려주세요.
다음 파일로 실제 결과를 확인할 수 있습니다.

| 파일 | 내용 |
|---|---|
| `status.json` | running/completed/failed/interrupted, 종료 코드·시각 |
| `run-*.log` | 실행별 전체 추론 로그 및 오류 |
| `manifest.json` | 실험 설정과 문제 목록·데이터 해시 |
| `experiment-code.json` | 소스 해시, 모델 revision, 생성 설정 |
| `environment-*.json` | 실행 Python·패키지 버전 |
| `results.jsonl` | 문제별 경로·유효성·연산 수·저장 수·길이 |
| `table1.csv`, `summary.json` | Table 1 형식 집계 |
| `responses/*.json` | 실제 원문 응답·프롬프트·토큰 수·종료 사유 |
| `verification.json` | 사후 검증 결과와 논문 대비 차이 |

`table1.csv`와 `summary.json`은 추론이 정상 완료되었을 때 생성됩니다.
중단되어도 이미 완료한 응답과 문제별 결과, 종료 상태 및 로그는 남습니다.

주의할 재현 조건: 논문에 보고된 모델과 현재 고정 가중치의 세부 버전,
원본 코드의 GPT 명칭 불일치, metric 정의 등은 `THIRD_PARTY.md`에 정리되어 있습니다.
이 실행은 Llama 3 8B Instruct의 원 논문 재현용 기준 실험이며 독자 개선 방법 실험은 아닙니다.

## 과거 일회성 대조 파일 정리

공식 코드와의 충돌·탐색 로직 대조를 위해 만들었던 일회성 재평가 스크립트와 중간 출력은 정리했습니다. 실제 V1/V2 추론 결과와 최종 보고서·검증 기록은 보존했습니다. Vast의 새 실행은 [README_VAST.md](README_VAST.md)를 사용하세요.

## 공개 코드 정합 실험 v2

현재 런처 기본 설정은 `configs/llama3_table1_upstream.json`, 출력은
`outputs/llama3-table1-upstream-v2`입니다. 기존 추론 캐시와 분리되어 전체 응답을 새로 생성합니다.
변경 사항: 우선순위 g+(h+t), 응답 전체의 마지막 완성 좌표 목록을 사용하는 공식 파서,
입력 길이를 빼지 않는 max_new_tokens=8000. greedy, bfloat16, 가중치 revision은 유지합니다.
공식 파서의 제한된 숫자 뺄셈은 직접 계산하며 모델 문자열을 eval로 실행하지 않습니다.
Shapely 2.2.0 충돌 판정도 유지합니다. LLM 단독에는 목표점 필터를 추가하지 않습니다.

```bash
tmux new -s llama-upstream-v2
cd /home/skkai/Mobility/2026_2/ykim/assign
source scripts/activate_llama.sh
python scripts/run_llama_table1.py
```

세션에서 나올 때 Ctrl+b, d. 돌아올 때 `tmux attach -t llama-upstream-v2`.
중단되면 같은 명령으로 v2 캐시에 완료된 응답을 재사용해 이어갑니다.
종료 후 `python scripts/verify_table1.py outputs/llama3-table1-upstream-v2`로 검증합니다.

출력이 길면 입력+출력이 모델의 명시된 8192 문맥 범위를 넘을 수 있습니다.
이는 공개 코드의 출력 한도에 맞춘 실험 조건이며, 문맥 내 제한을 적용한 이전 실험과 구분합니다.
