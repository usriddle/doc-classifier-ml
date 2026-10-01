"""
colab_train.ipynb 생성 스크립트. (노트북을 고칠 때는 이 파일을 고치고 다시 실행하세요)

    python training/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# =============================================================================
md("""
# ⑤ Qwen QLoRA 학습 — Colab T4

`Qwen/Qwen3.5-4B` 위에 LoRA 어댑터를 학습해 **도구 호출 JSON 의 인자(content / location / id 등) 추출 품질**을
올리는 노트북입니다. 의도·카테고리는 소량만 섞어서 기존 능력이 흔들리지 않게 붙잡아 둡니다.

**위에서부터 차례로 실행하세요.** 2번 셀 다음에 런타임 재시작이 한 번 있습니다.

| 단계 | 하는 일 | 확인할 것 |
|---|---|---|
| 0~3 | GPU 확인, 드라이브 연결, 설치, 재시작 | `Tesla T4`, 버전 |
| 4 | 이번 학습 설정 (이름·데이터·하이퍼파라미터) | |
| 5 | 정답 데이터 읽기·검증·분할 | 오류 0건, 분할 경고 |
| 6 | ④ 후보 계산 (bge-m3) | ④ 가 정답을 놓친 건수 |
| 6-1 · 6-2 | (선택) ④ 방식 측정 → 가장 나은 설정을 `.env` 에 반영 | 놓침·망가뜨림, 카테고리별 놓침률 |
| 7 | 학습 샘플 만들기 | **학습 대상** 표시가 정답 부분에만 있는지 |
| 8 | 학습 전 기준선 평가 | 기준 점수 |
| 9 | LoRA 붙이기 | 어디에 몇 개 붙었는지 |
| 10 | **학습** | 실시간 loss 그래프, 경고, 중간 출력 |
| 11 | 학습 후 평가 · 전후 비교 · 판정 | 목표 지표 ↑, 방어 지표 유지, 지어냄 ↓ |
| 12 | 저장 · 버전 비교 | `adapters/<이름>/` |
| 13 | (재시작 후) 운영 경로로 최종 확인 | 서버와 똑같이 올렸을 때도 같은 결과인지 |

### 시작 전에 드라이브에 있어야 하는 파일 (`내 드라이브/backend/`)

| 경로 | 필수 | 용도 | 없으면 |
|---|---|---|---|
| `app/` | ✅ | 서버 코드. 프롬프트·모델 로드·판정을 학습과 평가가 그대로 씀 | 3-1 에서 `ModuleNotFoundError: app` |
| `training/` | ✅ | 학습 코드 (`records.py`, `samples.py`, `evaluate.py`, `trainer.py`) | 4 에서 `ModuleNotFoundError: training` |
| `requirements.txt`, `requirements-model.txt` | ✅ | 2 단계 설치 목록 | 2 에서 `No such file` |
| `data/train/records.jsonl` | ✅ | **학습 정답 데이터** (5 단계) | 예시 90건을 복사해서 진행 (시험 운전용) |
| `data/cases.csv` | 권장 | 벡터DB 사례 (6 단계 후보 재정렬) | 벡터DB 없이 ④ 후보만 사용 |
| `.env` | 선택 | 모델 설정 | 3-1 이 서버 노트북과 같은 값으로 자동 생성 |

3-1 단계 바로 아래 **파일 점검 셀**이 위 표를 자동으로 확인해 줍니다.

### 실행하면 드라이브에 새로 생기는 것

| 경로 | 언제 | 내용 | 지워도 되나 |
|---|---|---|---|
| `data/cases.npy`, `data/cases.meta.json` | 6 | 벡터DB (cases.csv 임베딩) | 됨 - 다시 만들어짐 |
| `data/train/cache/candidates.json` | 6 | 문장별 ④ 후보 3개 캐시 | 됨 - 다시 계산 |
| `data/train/cache/calib_queries.npz`, `calibration.json` | 6-1 | 측정용 질의문 벡터 캐시, 측정 결과 기록 | 됨 - 다시 계산 |
| `data/train/cache/baseline_*.json`, `*_rows.csv` | 8 | 학습 전 채점 결과 캐시 | 됨 - 다시 채점 |
| `adapters/<이름>/` | 12 | **학습 결과물** (어댑터 + 기록 + 평가) | 지우면 그 버전은 사라짐 |
| `adapters/_checkpoints/<이름>/` | 10 | 중간 저장 (`CHECKPOINT_TO_DRIVE=True` 일 때만) | 학습이 끝나면 지워도 됨 |
| `data/complaints.db` | 13 | 파이프라인 확인 때 접수된 테스트 민원 | 됨 |

모델 가중치(bge-m3 약 2GB, Qwen 약 8~10GB)는 드라이브가 아니라 **세션 디스크**에 받습니다. 런타임이 끊기면 다시 받습니다.

> 이 노트북은 서버 코드(`app/`)를 그대로 불러 씁니다. 학습 입력은 `prompts.py` 와 `llm_engine.py` 의
> 함수로 조립되므로 **추론 때 입력과 토큰 단위로 같습니다.** 평가도 운영 함수로 하므로 여기 점수가 곧 서버 동작입니다.
""")

md("""
## 0. GPU 확인

**[런타임] → [런타임 유형 변경] → 하드웨어 가속기** 에서 GPU 를 고르고 저장하세요.
무료는 `T4 GPU`, Colab Pro 면 `L4 GPU`(T4 보다 2~3배 빠름, 24GB) 또는 `A100 GPU`(가장 빠름, 40GB, 컴퓨팅 단위를 가장 많이 씀)를 고를 수 있습니다.
코드는 GPU 에 맞춰 자동으로 설정됩니다(L4·A100 은 bfloat16, T4 는 float16).

**📂 필요한 파일** — 없음

**📋 실행 결과**
- GPU 정보 표가 나옵니다. 가운데 GPU 이름(`Tesla T4` / `NVIDIA L4` / `NVIDIA A100-SXM4-40GB` 등)과 메모리가 보이면 정상입니다.
- `NVIDIA-SMI has failed` / `command not found` 가 나오면 런타임이 CPU 입니다. 위 메뉴에서 T4 로 바꾸세요.
- 파일이나 변수는 아무것도 만들지 않습니다.
""")
code("!nvidia-smi")

md("""
## 1. 코드 올리기 (구글 드라이브)

`MyDrive/backend` 폴더를 씁니다. 서버 노트북(`colab_backend.ipynb`)과 같은 폴더입니다.

**📂 필요한 파일**
- `내 드라이브/backend/` 폴더 전체 (맨 위 표의 파일들). 처음이면 PC 의 `backend` 폴더를 통째로 `내 드라이브` 바로 아래에 올리세요.
- 폴더 이름이 `backend` 가 아니거나 다른 폴더 안에 있으면 아래 `%cd` 경로를 그에 맞게 고치세요.

**📋 실행 결과**
- 구글 계정 선택 창 → 드라이브 접근 허용을 누르면 `Mounted at /content/drive` 가 나옵니다.
- 작업 경로가 `/content/drive/MyDrive/backend` 로 바뀌고, `ls` 로 폴더 목록(`app`, `training`, `data`, `requirements.txt` …)이 보입니다.
- `No such file or directory` 가 나오면 드라이브에 `backend` 폴더가 없거나 위치가 다른 것입니다.
""")
code("""
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/backend
!ls
""")

md("""
## 2. 설치

서버 노트북과 같은 설치에 학습용 `peft` · `accelerate` 를 더합니다. 5~10분 걸립니다.

**📂 필요한 파일** — `requirements.txt`, `requirements-model.txt` (backend 폴더 안)

**📋 실행 결과**
- 설치 로그가 조용히 흘러가고 마지막에 `설치 완료` 가 나옵니다.
- 중간의 `ERROR: pip's dependency resolver ...` 빨간 경고는 Colab 기본 패키지와의 버전 차이 알림이라 대부분 무시해도 됩니다. 셀이 끝까지 실행됐는지만 보세요.
- 패키지는 **세션 디스크**에 깔립니다. 드라이브에는 아무것도 쓰지 않으며, 런타임이 끊기면 다시 설치해야 합니다.
""")
code("""
!pip install -q -r requirements.txt
!grep -v '^torch' requirements-model.txt > /tmp/req_model.txt
!pip install -q -r /tmp/req_model.txt
!pip install -q bitsandbytes peft accelerate
!pip install -q --upgrade 'git+https://github.com/huggingface/transformers.git'
print('설치 완료 - 아래 셀로 런타임을 재시작하세요')
""")

md("""
## 3. 런타임 재시작 ⚠️

새 transformers 를 적용하려면 한 번 재시작해야 합니다. 끊긴 뒤 **3-1 셀부터** 이어서 실행하세요.

**📂 필요한 파일** — 없음

**📋 실행 결과**
- 셀이 곧바로 끊기고 `세션이 다운되었습니다` / `세션이 다시 시작되었습니다` 같은 알림이 뜹니다. **일부러 끊은 것이라 정상입니다.**
- 설치한 패키지는 남고, 메모리의 변수와 작업 경로(`%cd`)는 모두 초기화됩니다. 그래서 3-1 에서 드라이브 경로를 다시 잡습니다.
""")
code("""
import os
os.kill(os.getpid(), 9)   # '세션이 다시 시작되었습니다' 는 정상입니다
""")

md("""
### 3-1. 재시작 후 — 경로 다시 잡고 환경 확인

학습은 `.env` 의 모델 설정을 서버와 **똑같이** 씁니다. 아래 세 값이 서버와 다르면 학습한 어댑터가 서버에서 제대로 동작하지 않습니다.

- `LLM_MODEL` — 어댑터는 학습한 베이스 모델에서만 동작
- `LLM_ENABLE_THINKING=false` — 채팅 템플릿이 달라짐
- `LLM_LOAD_4BIT=true` — QLoRA 는 4bit 베이스 위에서 학습

**📂 필요한 파일**
- `app/` 폴더 (설정·프롬프트 코드를 불러옴)
- `.env` — 있으면 그대로 씁니다. **없으면 서버 노트북 5번 셀과 같은 값으로 드라이브에 새로 만듭니다.**

**📋 실행 결과**
- 설치된 버전(torch · transformers · peft · bitsandbytes)과 GPU 이름
- 모델 설정 네 줄. `실제 4bit 사용: True`, `연산 dtype` 은 T4 면 `torch.float16`, L4·A100 이면 `torch.bfloat16` 이 정상, `LLM_ENABLE_THINKING: False` 이어야 합니다.
- `프롬프트 지문` — `prompts.py`·`categories.py` 내용의 지문(16자리). 학습한 어댑터에 기록되고, 나중에 프롬프트가 바뀌었는지 비교할 때 씁니다.
- 마지막 줄이 `문제 없음` 이면 통과, `확인 필요` 면 나온 항목을 `.env` 에서 고친 뒤 이 셀을 다시 실행하세요.
- 메모리: GPU 사용량을 찍어 주는 `gpu()` 함수가 만들어집니다. (이후 단계에서 씀)
""")
code("""
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/backend

import os, sys, json, math, time, shutil, hashlib
from pathlib import Path

if not Path('.env').exists():
    # 서버 노트북 5번 셀과 같은 값
    Path('.env').write_text('''DEBUG=true
APP_ENV=local
DEVICE=auto
MODEL_CACHE_DIR=
PRELOAD_MODELS=false
EMBED_MODEL=BAAI/bge-m3
LLM_MODEL=Qwen/Qwen3.5-4B
LLM_LOAD_4BIT=true
LLM_4BIT_COMPUTE_DTYPE=auto
LLM_ENABLE_THINKING=false
CANDIDATE_MIN_SCORE=0.65
CASE_BLEND_WEIGHT=0.2
CANDIDATE_SCORING=multi
CASE_MODE=low_confidence
CASE_TOP_K=5
CASE_MIN_SCORE=0.5
GATE_ENABLED=true
''', encoding='utf-8')
    print('.env 가 없어 서버 노트북과 같은 값으로 만들었습니다.')

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')   # 메모리 조각화 완화 (torch 보다 먼저)
import torch, transformers, peft, bitsandbytes
from app.config import settings
from app.services import runtime, prompts

print('torch         :', torch.__version__, '| CUDA', torch.cuda.is_available())
print('GPU           :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '없음')
print('transformers  :', transformers.__version__)
print('peft          :', peft.__version__)
print('bitsandbytes  :', bitsandbytes.__version__)
print()
print('LLM_MODEL          :', settings.LLM_MODEL)
print('LLM_LOAD_4BIT      :', settings.LLM_LOAD_4BIT, '-> 실제 4bit 사용:', runtime.use_4bit())
print('연산 dtype          :', runtime.compute_dtype(), '(T4 는 float16, L4·A100 은 bfloat16 이 정상)')
print('LLM_ENABLE_THINKING:', settings.LLM_ENABLE_THINKING)
print('프롬프트 지문        :', prompts.prompt_fingerprint(), '(prompts.py 가 바뀌면 달라짐)')
print('④ 설정              :', f'CANDIDATE_SCORING={settings.CANDIDATE_SCORING}',
      f'CASE_MODE={settings.CASE_MODE}', f'CANDIDATE_MIN_SCORE={settings.CANDIDATE_MIN_SCORE}',
      f'CASE_BLEND_WEIGHT={settings.CASE_BLEND_WEIGHT}')

problems = []
if settings.LLM_ENABLE_THINKING: problems.append('LLM_ENABLE_THINKING 을 false 로 두세요.')
if not runtime.use_4bit(): problems.append('4bit 로 올라가지 않습니다. LLM_LOAD_4BIT=true 와 GPU 를 확인하세요.')
if settings.LLM_ADAPTER_PATH:
    print('\\n참고: .env 에 LLM_ADAPTER_PATH 가 있지만, 학습은 항상 베이스 모델에서 새로 시작합니다.')
print('\\n' + ('문제 없음' if not problems else '확인 필요:\\n - ' + '\\n - '.join(problems)))

def gpu(label=''):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**3
        m = torch.cuda.max_memory_allocated() / 1024**3
        t = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f'[GPU {label}] 사용 {a:.1f}GB | 최대 {m:.1f}GB | 전체 {t:.1f}GB')
""")

md("""
### 3-2. 필요한 파일 점검

맨 위 표의 파일이 드라이브에 있는지 한 번에 확인합니다. **필수(✅) 항목이 `없음`이면 여기서 멈추고** 파일을 올린 뒤 다시 실행하세요.

**📂 필요한 파일** — 점검 대상 자체 (맨 위 표)

**📋 실행 결과**
- 파일마다 `있음` / `없음` 과 크기·줄 수가 표로 나옵니다. (`.env` 는 3-1 이 이미 만들었으면 `있음`)
- `records.jsonl` 은 줄 수(= 정답 레코드 수)도 보여 줍니다. 보강 데이터라면 5000 이 나와야 합니다.
- 필수 파일이 빠져 있으면 마지막에 `❌ 필수 파일 없음` 과 함께 셀이 멈춥니다.
""")
code("""
def _info(p):
    p = Path(p)
    if p.is_dir():
        return '있음', f'{sum(1 for _ in p.rglob(\"*.py\"))}개 .py'
    if p.is_file():
        size = p.stat().st_size
        lines = sum(1 for _ in open(p, encoding='utf-8-sig', errors='ignore')) if p.suffix in ('.jsonl', '.csv') else None
        return '있음', f'{size/1024:.0f}KB' + (f' / {lines}줄' if lines is not None else '')
    return '없음', ''

CHECKS = [
    ('app/', True, '서버 코드'),
    ('training/', True, '학습 코드'),
    ('requirements.txt', True, '설치 목록'),
    ('requirements-model.txt', True, '설치 목록(모델)'),
    ('data/train/records.jsonl', False, '학습 정답 데이터 - 없으면 5단계가 예시 90건을 복사'),
    ('data/cases.csv', False, '벡터DB 사례 - 없으면 벡터DB 없이 진행'),
    ('.env', False, '모델 설정 - 3-1 이 없으면 자동 생성'),
    ('training/sample_data/sample_records.jsonl', False, '예시 데이터'),
]
rows, missing = [], []
for path, required, note in CHECKS:
    status, detail = _info(path)
    if required and status == '없음':
        missing.append(path)
    rows.append({'경로': path, '필수': '✅' if required else '', '상태': status, '크기': detail, '설명': note})
import pandas as pd
display(pd.DataFrame(rows))
if missing:
    raise SystemExit(f'❌ 필수 파일 없음: {missing} - 드라이브의 backend 폴더에 올린 뒤 다시 실행하세요.')
print('필수 파일 모두 있음')
""")

# =============================================================================
md("""
## 4. 이번 학습 설정

한 번에 **하나만** 바꿔 가며 실험하세요. 여러 값을 같이 바꾸면 무엇 때문에 달라졌는지 알 수 없습니다.

| 값 | 기본 | 이렇게 보이면 이렇게 |
|---|---|---|
| `learning_rate` | 2e-4 | loss 가 튀거나 `nan` → 1e-4 로 / loss 가 거의 안 줄면 3e-4 로 |
| `epochs` | 1 | 시간이 되면 2~3. 검증 loss 가 중간부터 오르면(과적합) → 줄이기 |
| `lora_r` | 16 | 좁은 과제라 보통 8~16 이면 충분. 올리면 과적합·기존 능력 손상 위험 ↑ |
| `MIX` | 도구 0.6 / 의도 0.2 / 카테고리 0.2 | 의도·카테고리 정확도가 떨어지면 두 값을 올리기 (도구 JSON 은 적은 샘플로도 잘 배움) |
| `NONE_BOOST` | 3 | 카테고리 정답이 '없음' 인 샘플 배수. `카테고리 없음 재현율` 이 낮으면 올리기 |
| `max_length` | 4096 | 이보다 긴 샘플은 버립니다 (정답이 잘린 채 학습되지 않게) |
| `TRAIN_SAMPLE_LIMIT` | 1000 | 학습 샘플 수 상한. **학습 시간이 이 값에 거의 비례**합니다. 과제·의도 비율은 유지한 채 줄입니다. None 이면 전부(약 5,000개) |
| `VAL_SAMPLE_LIMIT` | 100 | 학습 중 검증 loss 를 잴 샘플 수. 검증은 학습 중 여러 번 반복되므로 줄일수록 빨라집니다 |
| `evals_per_epoch` | 2 | 한 epoch 에 검증 loss 를 몇 번 잴지. 검증 1번 = 검증 샘플 전부 계산이라 횟수만큼 시간이 늘어납니다 |
| `gradient_checkpointing` | True | False 면 학습이 약 20~30% 빠르지만 메모리를 훨씬 더 씁니다. Qwen3.5 의 선형 어텐션 층이 전용 커널 없이 돌면 **L4(24GB)에서도 메모리가 부족**했습니다. A100(40GB 이상)에서만 False 를 시도하세요 |
| `EVAL_LIMIT` | None | 8·11·13 단계 채점에 쓸 평가셋 문장 수. None 이면 전부. 시간이 부족하면 200~300 |

**`RUN_NAME`** 은 어댑터 폴더 이름입니다. 비워 두면 `v번호_날짜` 로 자동으로 정합니다.
런타임이 끊겨 이어서 학습하려면 **끊긴 실행의 이름을 그대로 적고** `RESUME=True` 로 두세요.

**⏱ 걸리는 시간 (T4 기준, 대략. L4 는 약 2~3배, A100 은 그보다 더 빠름)** — 샘플마다 고정 프리픽스(약 2,000자, 샘플 토큰의 대부분)까지 매번 계산하므로 느립니다.
- 학습: `학습 샘플 수 ÷ grad_accum` = epoch 당 스텝 수. 샘플 하나에 대략 3~7초(추정)가 걸립니다.
  - `TRAIN_SAMPLE_LIMIT=1000`(기본) → epoch 당 약 125스텝, **약 1~2시간**
  - 제한 없이 약 5,000개 → epoch 당 약 600스텝, **약 5~10시간 이상** (무료 Colab 에서는 끝까지 가기 어려움)
  - 위 숫자는 추정입니다. 10단계 첫 몇 스텝 뒤 표시되는 **남은 시간**이 실제 값이니 그걸 보고 조절하세요.
  - 남은 시간에는 **검증 loss 측정 시간이 빠져 있습니다.** 검증 1번 ≈ 검증 샘플 수 × (학습 샘플 1개 시간의 약 1/3).
- Colab 무료 런타임은 오래 돌리면 끊길 수 있습니다. 긴 학습이면 `CHECKPOINT_TO_DRIVE=True` 를 권합니다.
- 채점(8·11단계): 도구 JSON 을 문장마다 실제로 생성하므로 **평가셋 약 700건이면 수십 분~1시간 이상**. `EVAL_LIMIT` 으로 줄일 수 있습니다.

**📂 필요한 파일** — 없음 (값만 정하는 셀). `adapters/` 에 기존 버전이 있으면 그 개수로 다음 번호를 정합니다.

**📋 실행 결과**
- 이번 실행 이름, 어댑터가 저장될 경로, 체크포인트 경로가 출력됩니다.
- 드라이브에 빈 폴더 `adapters/`, `data/train/cache/` 가 없으면 만듭니다.
- `⚠️ … 가 이미 있습니다` 가 나오면 같은 이름의 결과가 이미 있다는 뜻입니다. 그대로 가면 12단계에서 덮어쓰니 `RUN_NAME` 을 바꾸세요.
- 메모리: `RUN_NAME`, `CFG`(학습 설정), `MIX`, `RUN_DIR`, `CKPT_DIR` 등이 만들어져 이후 단계가 씁니다.
""")
code("""
from training.trainer import TrainConfig

RUN_NAME = ''                       # 비우면 자동 (예: v3_0929)
DATA_FILE = 'data/train/records.jsonl'
SPLIT_RATIOS = (0.7, 0.15, 0.15)    # 학습 / 검증 / 평가 (group 해시로 고정 배정)
MIX = {'tool': 0.6, 'intent': 0.2, 'category': 0.2}   # 의도·카테고리가 떨어지면 두 값을 올리기
NONE_BOOST = 3               # 카테고리 정답이 '없음' 인 샘플을 몇 배로 (조회·수정·삭제의 없음 판정 연습)

CFG = TrainConfig(
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    learning_rate=2e-4,
    epochs=1,                # 5,000건 기준 T4 에서 한 epoch 도 여러 시간. 시간이 되면 2~3
    grad_accum=8,
    max_length=4096,
    evals_per_epoch=2,       # 검증 loss 를 epoch 당 몇 번 잴지 (많을수록 느림)
    spot_check_every=2,      # 검증 2번마다 실제 출력 몇 건을 뽑아 봄 (0 이면 끔)
    gradient_checkpointing=True,   # 메모리 절약. False 면 약 20~30% 빠르지만 L4(24GB)에서도 메모리 부족이 날 수 있음
    full_kbit_prep=False,    # loss 가 nan 이면 True 로 (메모리 더 씀)
    seed=42,
)
SPOT_CHECK_N = 3             # 중간 점검에 쓸 검증 레코드 수
TRAIN_SAMPLE_LIMIT = 1000    # 학습 샘플 수 상한 - 학습 시간이 거의 비례 (None 이면 전부, 약 5,000개)
VAL_SAMPLE_LIMIT = 100       # 학습 중 검증 loss 를 잴 샘플 수 (None 이면 전부)
EVAL_LIMIT = None            # 8·11·13 단계 채점 문장 수 (None 이면 평가셋 전부, 예: 300)
RESUME = False               # 끊긴 학습 이어서 하기 (같은 RUN_NAME 필요)
CHECKPOINT_TO_DRIVE = False  # True 면 체크포인트를 드라이브에 저장 (끊겨도 이어서 가능, 대신 수백 MB 사용)

ADAPTERS_DIR = Path('adapters')
CACHE_DIR = Path('data/train/cache')
ADAPTERS_DIR.mkdir(exist_ok=True); CACHE_DIR.mkdir(parents=True, exist_ok=True)

if not RUN_NAME:
    n = len([p for p in ADAPTERS_DIR.glob('v*') if p.is_dir()]) + 1
    RUN_NAME = f"v{n}_{time.strftime('%m%d')}"
RUN_DIR = ADAPTERS_DIR / RUN_NAME
CKPT_DIR = (ADAPTERS_DIR / '_checkpoints' / RUN_NAME) if CHECKPOINT_TO_DRIVE else Path('/content/checkpoints') / RUN_NAME

print('이번 실행      :', RUN_NAME)
print('어댑터 저장    :', RUN_DIR)
print('체크포인트     :', CKPT_DIR, '(드라이브)' if CHECKPOINT_TO_DRIVE else '(세션 디스크 - 끊기면 사라짐)')
if RUN_DIR.exists() and not RESUME:
    print(f'\\n⚠️ {RUN_DIR} 가 이미 있습니다. 덮어쓰지 않으려면 RUN_NAME 을 바꾸세요.')
""")

# =============================================================================
md("""
## 5. 정답 데이터 — 읽기 · 검증 · 분할

정답 데이터는 **민원 문장 + Qwen 이 내놓아야 할 올바른 답(의도·카테고리·도구 인자)** 을 한 줄에 한 건씩 적은 파일입니다.
이 셀이 그 파일을 읽어 **검사**하고, **학습 / 검증 / 평가** 세 묶음으로 나눕니다. 정답을 새로 만들지는 않습니다.

```json
{"id": "r0001", "group": "g0001", "text": "행복로 23길 가로등이 꺼졌어요", "intent": "접수", "category": "국토교통",
 "arguments": {"content": "가로등 꺼짐", "location": "행복로 23길"}, "tags": ["구어체"]}
{"id": "r0002", "group": "g0002", "text": "어제 넣은 가로등 민원 취소해 주세요", "intent": "삭제", "category": "국토교통",
 "arguments": {"complaint_id": 0, "keyword": "가로등", "period": "어제", "reason": ""}}
```

| intent | category | arguments |
|---|---|---|
| 접수 | 7종 중 하나 (필수) | `content`, `location` |
| 조회 | 7종 또는 `없음` (필수) | `complaint_id` (없으면 0), `keyword`, `period`, `field` (status / history / list) |
| 수정 | 7종 또는 `없음` (필수) | `complaint_id`, `keyword`, `period`, `content`, `location` (안 바꾸는 항목은 `""`) |
| 삭제 | 7종 또는 `없음` (필수) | `complaint_id`, `keyword`, `period`, `reason` (없으면 `""`) |
| 문의 · 해당없음 | 적지 않음 | 적지 않음 |

조회·수정·삭제는 시민이 번호 대신 **"어제 넣은 가로등 민원"** 처럼 말하는 경우가 대부분이라, 찾기 조건을 함께 적습니다.
- `keyword` — 찾을 민원의 **대상 명사만** 원문 그대로 (예: `가로등`, `엘리베이터`, `주차`). 번호·동·호·층·노선(`3402번`, `101동`), 장소 수식, 상태·요청 말(`고장`, `불법`, `지연`)은 빼고, `맨홀 뚜껑`·`방범 CCTV`처럼 한 덩어리 명사는 그대로. 대상을 알 수 없는 말("그 민원", "안전 문제")뿐이면 `""`.
- `period` — 접수 시점(원문 그대로, 예: `어제`, `지난주`, `방금`). 날짜 계산은 서버 코드가 합니다.
- `category` — 대상이 7종 중 하나로 분명하면 그 이름, 주제가 안 드러나거나 애매하면 `없음`.

**라벨링 규칙** — 정답이 흔들리면 모델도 흔들립니다.
- `location` 은 원문 표현을 **그대로** 옮깁니다. "우리 동네", "여기", "근처"처럼 장소를 좁혀 주지 않는 말만 있으면 `""`. (검증이 원문에 없는 위치를 잡아냅니다)
  민원 대상 시설물·물건(가로수, 담장, 전봇대, CCTV …)은 넣지 않고 장소까지만 (`들꽃길 주택 담장이 기울어서` → `들꽃길 주택`). 어긋나 보이면 `🔎` 경고가 나옵니다.
- 의도 경계: 전입신고·여권·대관 같은 **행정 업무의 처리 지연·절차 불편**은 `접수`, **이미 넣은 민원의 진행**을 묻는 말은 `조회`.
- 카테고리 경계: 공사장 소음 → 주택건축, 도로 먼지·거리 청소 → 환경·위생, 홈페이지 장애·공무원 칭찬 → 기타.
- 장소가 두 번 나오면 **민원이 발생한 곳**이 정답입니다. 수정의 "A에서 B로"는 새 위치 B 가 정답입니다.
- `content` 는 "무엇이 어떤 상태인지"를 명사형으로 짧게 (예: `가로등 꺼짐`). 위치·기간·요청 문구는 넣지 않습니다.
- 말만 바꿔 늘린 변형은 같은 `group` 으로 묶으세요. 같은 group 은 같은 쪽(학습/검증/평가)에 들어갑니다.
- 헷갈리기 쉬운 문장에는 `"tags": ["경계"]` — 의도 학습 샘플에 우선 들어갑니다.

분할은 `group` 의 해시로 정하므로 **데이터를 더 넣어도 기존 평가셋이 바뀌지 않습니다.** (학습 전후·버전 간 비교가 공정해짐)
특정 줄을 원하는 곳에 넣으려면 `"split": "train" / "val" / "test"` 를 적습니다.

**📂 필요한 파일**
- **`data/train/records.jsonl`** (필수) — 보강 데이터 5,000건 파일을 이 위치에 두세요. 경로를 바꾸려면 4단계의 `DATA_FILE`.
  - 없으면 예시 90건(`training/sample_data/sample_records.jsonl`)을 이 위치로 복사하고 경고를 띄웁니다. 예시는 시험 운전용이라 학습 효과를 판단할 수 없습니다.
- `data/cases.csv` (선택) — 검증·평가 문장이 벡터DB 사례와 똑같은지 확인할 때만 씁니다.

**📋 실행 결과**
- `레코드 N건 | 검증 오류 0건` — 오류가 있으면 `[레코드 id] 이유` 목록이 나오고 셀이 멈춥니다. 파일을 고쳐 다시 올린 뒤 이 셀만 다시 실행하세요.
  (예: location 이 원문에 없음, 필수 인자 누락, 없는 카테고리 이름)
- **의도별 건수 표** — 열이 학습/검증/평가. 보강 데이터면 대략 3,630 / 674 / 696 입니다. 검증·평가 열에 0 인 의도가 있으면 그 의도는 채점되지 않습니다.
- **태그 표** — 위치있음/없음, id있음/없음, 경계, 구어체 등 어려운 경우가 세 묶음에 고르게 들어갔는지.
- `⚠️` 경고 — 평가셋에 없는 의도, 평가셋이 30건 미만, 벡터DB 사례와 같은 문장 등.
- `EVAL_LIMIT` 을 정했으면 평가셋을 의도 비율대로 그 수만큼 줄였다는 줄이 나옵니다.
- 아래 셀은 의도별로 2건씩 문장과 정답 JSON 을 보여 줍니다. 라벨이 규칙대로인지 눈으로 확인하세요.
- 메모리: `records`(전체), `train_recs` · `val_recs` · `test_recs`(세 묶음), `DATA_SHA`(파일 지문). 파일은 새로 만들지 않습니다.
""")
code("""
import pandas as pd
from training import records as R

if not Path(DATA_FILE).exists():
    Path(DATA_FILE).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy('training/sample_data/sample_records.jsonl', DATA_FILE)
    print(f'⚠️ {DATA_FILE} 가 없어 예시 데이터를 복사했습니다. 실제 학습 전에 직접 만든 데이터로 바꾸세요.\\n')

records = R.load_records(DATA_FILE)
errors = R.validate(records)
print(f'레코드 {len(records)}건 | 검증 오류 {len(errors)}건')
for e in errors[:30]:
    print('  -', e)
assert not errors, '위 오류를 고친 뒤 이 셀을 다시 실행하세요.'

DATA_SHA = hashlib.sha256(Path(DATA_FILE).read_bytes()).hexdigest()[:16]
parts = R.split_records(records, SPLIT_RATIOS)
train_recs, val_recs, test_recs = parts['train'], parts['val'], parts['test']

summary = pd.DataFrame({side: pd.Series(R.summarize(recs)['의도']) for side, recs in parts.items()}).fillna(0).astype(int)
summary.loc['합계'] = summary.sum()
display(summary)

tags = pd.DataFrame({side: pd.Series(R.summarize(recs)['태그']) for side, recs in parts.items()}).fillna(0).astype(int)
display(tags[~tags.index.str.startswith('의도:')])

for w in R.lint(records)[:20]:
    print('🔎', w)
for w in R.split_warnings(parts):
    print('⚠️', w)

if EVAL_LIMIT and len(test_recs) > EVAL_LIMIT:
    # 의도 비율을 유지하며 고정된 부분집합만 채점 (매번 같은 문장이 뽑힘)
    import random as _random
    _rng, by_intent = _random.Random(0), {}
    for r in sorted(test_recs, key=lambda r: r.id):
        by_intent.setdefault(r.intent, []).append(r)
    picked = []
    for rs in by_intent.values():
        picked += _rng.sample(rs, min(len(rs), max(1, round(EVAL_LIMIT * len(rs) / len(test_recs)))))
    print(f'평가셋 {len(test_recs)}건 중 {len(picked)}건만 채점합니다 (EVAL_LIMIT={EVAL_LIMIT}, 의도 비율 유지)')
    test_recs = sorted(picked, key=lambda r: r.id)
TEST_IDS = [r.id for r in test_recs]
overlap = R.cases_overlap(val_recs + test_recs, settings.CASE_CSV_PATH)
if overlap:
    print(f'⚠️ 검증/평가 문장 {len(overlap)}건이 벡터DB 사례(cases.csv)에 그대로 있습니다 - 카테고리 점수가 부풀 수 있습니다: {overlap[:10]}')
""")

md("의도별 예시 몇 건 — 정답이 라벨링 규칙대로 적혔는지 눈으로 확인하세요.")
code("""
rows = []
for intent in [i.name for i in prompts.INTENTS]:
    for r in [x for x in records if x.intent == intent][:2]:
        rows.append({'의도': r.intent, '문장': r.text, '카테고리': r.category,
                     '정답 JSON': prompts.format_tool_call(r.tool, r.tool_arguments()) if r.tool else '-'})
pd.set_option('display.max_colwidth', 120)
display(pd.DataFrame(rows))
""")

# =============================================================================
md("""
## 6. ④ 후보 계산 (bge-m3)

카테고리 학습 샘플의 후보 3개는 **운영과 같은 코드**(`pipeline.run_candidates`, 벡터DB 재정렬 포함)로 만듭니다.
그래야 학습 때 보는 후보 분포가 서버와 같습니다. 결과는 드라이브에 캐시돼서 다음부터는 바로 읽습니다.

끝나면 bge-m3 를 GPU 에서 내려 학습에 메모리를 몰아줍니다.

- **정답 위치 분포** — 정답이 대부분 1번이면, 학습 샘플에서 후보 순서를 섞어 '무조건 1번' 버릇을 막습니다. (7단계에서 자동)
- **④ 놓침** — 정답이 후보 3개 밖. ⑤ 가 고를 수 없는 경우라 평가에서는 따로 셉니다.

**📂 필요한 파일**
- `data/cases.csv` — 벡터DB 사례. `.env` 의 `CASE_STORE_ENABLED=true`(기본)일 때 읽습니다. 없으면 벡터DB 없이 ④ 후보만 씁니다.
- `data/train/cache/candidates.json` — 있으면 캐시로 씁니다 (없으면 새로 만듦).
- bge-m3 가중치(약 2GB) — 처음에는 인터넷에서 내려받아 세션 디스크에 둡니다.

**📋 실행 결과**
- 처음 실행: bge-m3 다운로드 → 벡터DB 생성 → `④ 후보 계산` 진행 막대(레코드 수만큼). 5,000건이면 **수 분~십수 분**.
  두 번째부터는 같은 문장·같은 설정이면 캐시에서 읽어 **몇 초**. `cases.csv`·카테고리 정의·임베딩 설정이 바뀌면 자동으로 다시 계산합니다.
- `카테고리 정답 위치 분포` — 예: `{1: 1500, 2: 300, 3: 100}`. 1 이 가장 많으면 정상입니다.
- `④ 가 정답을 후보 3개에 못 올린 건수` — 많으면(수백 건) 학습으로 해결되지 않는 ④ 의 한계입니다. **6-1 측정 셀**로 벡터DB 사용 방식을 비교해 보세요. (`categories.py` 설명을 계속 고치는 것보다 효과가 큽니다)
- `low_confidence (벡터DB 조회) 건수` — ④ 가 애매해 벡터DB 로 후보 순서를 다시 정한 문장 수.
- `[GPU bge-m3 내린 뒤] 사용 0.x GB` — bge-m3 를 내린 뒤라 거의 0 이어야 합니다.
- 드라이브에 생기는 파일: `data/cases.npy`, `data/cases.meta.json`(벡터DB), `data/train/cache/candidates.json`(후보 캐시)
- 메모리: `cands`(레코드 id → 후보 3개·점수), `CAND_SIG`(후보 설정 지문)
""")
code("""
from training import samples as S
from app.services import embedder, case_store

if settings.CASE_STORE_ENABLED:
    case_store.load_or_build()
cands = S.compute_candidates(records, cache_path=CACHE_DIR / 'candidates.json')
CAND_SIG = S._candidate_signature()

pos, miss = {}, []
for r in records:
    names = cands[r.id]['names']
    if r.category and r.category != R.NONE:   # '없음' 정답은 ④ 후보와 무관
        if r.category in names: pos[names.index(r.category) + 1] = pos.get(names.index(r.category) + 1, 0) + 1
        else: miss.append(r.id)
print('카테고리 정답 위치 분포 (1이 가장 좋음):', dict(sorted(pos.items())))
print(f'④ 가 정답을 후보 3개에 못 올린 건수: {len(miss)}', miss[:10])
print('low_confidence (벡터DB 조회) 건수:', sum(1 for v in cands.values() if v['low_confidence']))

embedder.unload()
gpu('bge-m3 내린 뒤')
""")

# =============================================================================
md("""
### 6-1. (선택) ④ 방식 측정 — 벡터DB 를 언제·어떻게 쓸지 고르기

6단계의 `④ 놓침` 이 많을 때 실행합니다. 아래 방식들을 records 로 **한 번에** 비교해 가장 나은 설정을 찾습니다.
학습과 무관하고 bge-m3 만 씁니다.

| 설정 | 후보 | 뜻 |
|---|---|---|
| `CANDIDATE_SCORING` | `single` / `multi` | ④ 카테고리 점수. `single` 은 정의문 벡터 1개, `multi` 는 정의문 + 키워드 하나하나 중 가장 가까운 것 |
| `CASE_MODE` | `low_confidence` / `always` / `union` | 벡터DB 를 애매할 때만 / 매번 섞기 / 매번 섞고 사례 1위를 후보에 반드시 포함 |
| `CANDIDATE_MIN_SCORE` | 0.45 ~ 0.70 | `low_confidence` 일 때 벡터DB 를 여는 문턱 (높을수록 자주 열림) |
| `CASE_BLEND_WEIGHT` | 0 ~ 0.8 | 섞을 때 사례 점수 비중 |

**어떻게 재나**
1. 접수 레코드(정답 카테고리가 있는 것)마다 질의문 벡터를 한 번만 만들고, 카테고리 점수(두 방식)와 비슷한 사례 10개를 **문턱과 상관없이** 모두 구해 둡니다.
2. 조합 94가지마다 top-3 를 다시 계산합니다. 운영과 **같은 함수**(`candidates.score_matrix`, `case_store.combine`)를 쓰므로 여기 결과가 곧 서버 동작입니다. 산수라 몇 초면 끝납니다.
3. **학습+검증(train+val)으로 고르고, 평가(test)는 확인만** 합니다. 평가셋으로 고르면 평가 점수가 부풀기 때문입니다.
4. 고르는 규칙: 놓침이 가장 적은 조합 근처(0.2%p 이내) 중에서 **망가뜨림이 적은 것 → 단순한 방식(single, low_confidence < always < union) → 작은 가중치** 순.

**📂 필요한 파일**
- 5·6 단계를 이 세션에서 실행한 상태 (`records`, `parts`, `cands` 를 씀)
- `data/cases.csv` 와 벡터DB 파일 (6단계에서 만든 것)
- `data/train/cache/calib_queries.npz` — 있으면 질의문 벡터를 여기서 읽습니다 (없으면 새로 만듦)

**📋 실행 결과**
- `현재 설정 재현` — 여기서 계산한 현재 설정의 top-3 가 6단계(운영 코드) 결과와 몇 건 같은지. **거의 전부 일치**해야 이 측정을 믿을 수 있습니다. (배치/1건 임베딩의 소수점 차이로 동점 근처 몇 건은 다를 수 있음)
- **비교표 (학습+검증)** — 첫 줄이 현재 설정, 아래는 놓침이 적은 순서 15개. 열의 뜻:
  - `놓침` · `후보3 적중률` — 정답이 후보 3개 밖인 건수 / 안에 든 비율
  - `1위 적중률` — 정답이 1위인 비율 (참고. ⑤ 가 최종 선택)
  - `살림` · `망가뜨림` — 현재 설정 대비 새로 맞힌 건수 / 새로 놓친 건수. **순이익만 보지 말고 망가뜨림을 따로 보세요.**
  - `최악 카테고리` — 놓침률이 가장 높은 카테고리. 모두 10% 이하가 목표
  - `벡터DB 열림` — 벡터DB 를 연 비율
- **선택한 방식** 한 줄과 **평가셋(test) 확인** — 현재 vs 선택. 학습+검증과 비슷하게 좋아져야 합니다. 평가셋에서만 크게 나빠지면 과하게 맞춘 것입니다.
- **카테고리별 놓침표** — 현재 vs 선택, 학습+검증 / 평가 각각
- **선택한 방식으로도 남는 놓침 예시** — `사례 1위` 가 `(없음)` 이면 비슷한 사례가 없다는 뜻이라 `cases.csv` 에 그 유형을 넣으면 됩니다. 사례 1위가 정답과 다르면 사례 라벨이나 경계를 확인하세요.
- `.env 에 넣을 값` — 다음 6-2 셀이 이 값을 씁니다.
- 드라이브에 생기는 파일: `data/train/cache/calib_queries.npz`(질의문 벡터 캐시), `data/train/cache/calibration.json`(측정 기록)
- 메모리: `table`, `sw`(전체 결과), `chosen`(선택한 방식)
""")
code("""
from training import calibrate as C

split_of = {r.id: side for side, recs in parts.items() for r in recs}
table = C.collect(records, split_of, cache_path=CACHE_DIR / 'calib_queries.npz')
same, total, diff = C.consistency(table, cands)
print(f'현재 설정 재현: {same}/{total} 일치' + (f'  (다른 id 예: {diff[:5]})' if diff else ''))

sw = C.sweep(table)
chosen = C.choose(sw)
ranked = sorted(sw.results, key=lambda r: (r.miss, r.broken))[:15]
print(f'\\n비교표 - 학습+검증 {sw.baseline.n}건 (정답 카테고리가 있는 레코드), 조합 {len(sw.results)}가지')
display(pd.DataFrame(C.rows([sw.baseline] + ranked), index=['현재 설정'] + [f'{i+1}' for i in range(len(ranked))]))

print('\\n선택한 방식:', chosen.strategy.label())
chosen_test = C.evaluate(table, chosen.strategy, sw.test_index, set(sw.baseline_test.miss_ids))
print(f'평가셋(test) 확인 - {sw.baseline_test.n}건')
display(pd.DataFrame(C.rows([sw.baseline_test, chosen_test]), index=['현재 설정', '선택']))

print('카테고리별 놓침 (놓침/전체)')
display(pd.DataFrame(C.per_category({
    '현재(학습+검증)': sw.baseline, '선택(학습+검증)': chosen,
    '현재(평가)': sw.baseline_test, '선택(평가)': chosen_test,
})).set_index('카테고리'))

print('선택한 방식으로도 남는 놓침 예시 (학습+검증)')
display(pd.DataFrame(C.miss_examples(table, chosen.strategy, sw.tune_index, limit=30)))

print('.env 에 넣을 값:')
for k, v in chosen.strategy.env().items():
    print(f'  {k}={v}')
C.save(sw, chosen, CACHE_DIR / 'calibration.json')

embedder.unload()
gpu('bge-m3 내린 뒤')
""")

md("""
### 6-2. (선택) 고른 설정을 `.env` 에 반영

6-1 의 결과가 마음에 들면 `APPLY_ENV = True` 로 바꾸고 실행합니다. 드라이브의 `.env` 에서 해당 줄만 바꿉니다.
(주석과 다른 설정은 그대로. 없는 키는 파일 끝에 추가)

- 비교표의 다른 줄을 쓰고 싶으면 `PICK` 을 직접 적으세요. 예) `PICK = C.Strategy(scoring='single', mode='union', weight=0.3)`
- **바꾼 뒤에는 런타임 → 세션 다시 시작 → 3-1, 4, 5, 6 을 다시 실행**하세요. 설정은 서버 코드를 불러올 때 한 번 읽히고,
  ④ 후보가 바뀌므로 6단계 후보 캐시도 자동으로 다시 계산됩니다. 6단계의 `④ 놓침` 이 6-1 표의 값과 비슷하게 줄었는지 확인하세요.
- **서버도 같은 값이어야 합니다.** 서버 노트북(colab_backend)은 같은 드라이브 `.env` 를 쓰므로 그대로 적용되고,
  PC 서버를 따로 쓰면 PC 의 `.env` 에도 같은 줄을 넣으세요. 학습 때와 서버의 ④ 후보가 다르면 카테고리 학습이 어긋납니다.

**📂 필요한 파일** — `.env` (없으면 새로 만들어 값을 적음)

**📋 실행 결과**
- `APPLY_ENV = False` 면 바뀔 내용만 보여 주고 아무것도 쓰지 않습니다.
- `APPLY_ENV = True` 면 `CASE_MODE: low_confidence -> union` 같은 변경 목록과 다음에 할 일이 나옵니다.
""")
code("""
APPLY_ENV = False          # True 로 바꾸면 .env 를 고칩니다
PICK = chosen.strategy # 다른 조합: C.Strategy(scoring='single', mode='union', weight=0.3)

print('적용할 값:', PICK.label())
for k, v in PICK.env().items():
    print(f'  {k}={v}   (지금: {getattr(settings, k)})')
if APPLY_ENV:
    for line in C.update_env('.env', PICK.env()) or ['바뀐 값 없음']:
        print('  변경 -', line)
    print('\\n다음: 런타임 → 세션 다시 시작 → 3-1, 4, 5, 6 을 다시 실행하세요.')
else:
    print('\\nAPPLY_ENV = False - .env 는 그대로입니다.')
""")

# =============================================================================
md("""
## 7. 베이스 모델 올리기 + 학습 샘플 만들기

베이스는 `llm_engine.load_base_model()` 로 올립니다. **서버와 같은 양자화 설정**입니다.

아래 출력에서 꼭 볼 것
- `【학습 대상 ▶ ... ◀】` 부분만 loss 가 걸립니다. 정답 JSON · 번호 뒤에 끝 토큰(`<|im_end|>`)이 붙어 있어야 합니다.
- 너무 길어서 버린 샘플이 있으면 `max_length` 를 올리거나 문장을 줄이세요.

**학습 샘플**은 레코드를 "모델이 받을 입력 전체 + 내놓아야 할 답" 토큰으로 바꾼 것입니다. 레코드 하나에서 과제별로 최대 3개가 나옵니다.

| 과제 | 입력 | 답 | 몇 개 |
|---|---|---|---|
| 도구 JSON | 문장 + 정답 의도 + 카테고리 | 정답 JSON | 접수·조회·수정·삭제 레코드 전부 |
| 의도 | 문장 | 번호 1개 | `MIX` 비율만큼. 의도별로 고르게(적은 의도를 조금 더), 각 의도 안에서 '경계' 문장 먼저 |
| 카테고리 | 문장 + ④ 후보 3개 (조회·수정·삭제는 + `없음`) | 번호 1개 | `MIX` 비율만큼 (학습용은 순서 섞은 샘플 추가) |

**📂 필요한 파일** — 없음 (앞 단계의 `records`, `cands` 를 씀). Qwen 가중치(약 8~10GB)를 처음에는 내려받아 세션 디스크에 둡니다.

**📋 실행 결과**
- 첫 실행은 Qwen 다운로드로 **5~15분**, 이후 로드는 1~2분. `[GPU Qwen 4bit 로드 후] 사용 약 3GB` 안팎이 정상입니다.
- `고정 프리픽스 길이: N 토큰` — 역할 지시·도구 정의·카테고리 정의·의도 정의 (FAQ 는 들어가지 않음). 약 2,000자라 대략 1천 토큰 안팎이며, 모든 샘플 앞에 붙고 loss 에서는 빠집니다.
- `학습 샘플을 N개 → 1000개로 줄였습니다` — `TRAIN_SAMPLE_LIMIT` 이 적용된 경우. 과제별·의도별 비율은 그대로입니다.
- **샘플 수 표** — `학습(최종)`이 실제로 학습할 수입니다. `학습(줄이기 전)`은 보강 데이터면 약 4,800개(도구 약 3,050 · 의도 약 1,060 · 카테고리 약 650),
  1,000개로 줄이면 대략 도구 640 · 의도 220 · 카테고리 140 입니다.
- `길이(토큰)` — 최소·중간·최대와 정답 부분 최대 길이. 최대가 `max_length` 보다 작아야 합니다.
- `정답을 후보에 끼워 넣은 건수` — 학습용 카테고리 샘플 중 ④ 가 놓친 정답을 후보에 넣어 만든 수.
- `검증 샘플을 N개로 줄였습니다` — `VAL_SAMPLE_LIMIT` 이 적용된 경우.
- 두 번째 셀: 과제별 샘플 하나를 글자로 풀어 보여 줍니다. **`【학습 대상 ▶ … ◀】` 안에 정답과 `<|im_end|>` 만 있어야** 합니다.
- 메모리: `tokenizer`, `base_model`(원본 Qwen), `train_samples`, `val_samples`. 파일은 만들지 않습니다.
""")
code("""
from app.services import llm_engine

tokenizer, base_model = llm_engine.load_base_model()
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
gpu('Qwen 4bit 로드 후')

builder = S.SampleBuilder(tokenizer, max_length=CFG.max_length)
print('고정 프리픽스 길이:', builder.prefix_tokens, '토큰 (모든 샘플 앞에 붙음, loss 제외)')

train_samples, train_stats = S.build_samples(train_recs, builder, cands, train=True, mix=MIX, seed=CFG.seed,
                                             none_boost=NONE_BOOST)
val_samples, val_stats = S.build_samples(val_recs, builder, cands, train=False, mix=MIX, seed=CFG.seed)
train_before = len(train_samples)
train_samples = S.limit_samples(train_samples, TRAIN_SAMPLE_LIMIT, train_recs, seed=CFG.seed)
if len(train_samples) < train_before:
    print(f'학습 샘플을 {train_before}개 → {len(train_samples)}개로 줄였습니다 (TRAIN_SAMPLE_LIMIT, 과제·의도 비율 유지)')
if VAL_SAMPLE_LIMIT and len(val_samples) > VAL_SAMPLE_LIMIT:
    import random as _random
    val_samples = _random.Random(CFG.seed).sample(val_samples, VAL_SAMPLE_LIMIT)
    print(f'검증 샘플을 {VAL_SAMPLE_LIMIT}개로 줄였습니다 (VAL_SAMPLE_LIMIT) - 학습 중 검증 시간을 줄이기 위함')

from collections import Counter as _Counter
display(pd.DataFrame({
    '학습(최종)': pd.Series(_Counter(x.task for x in train_samples)),
    '학습(줄이기 전)': pd.Series(dict(train_stats.counts)),
    '학습(가능한 최대)': pd.Series(dict(train_stats.available)),
    '검증(최종)': pd.Series(_Counter(x.task for x in val_samples)),
}).fillna(0).astype(int))
print('길이(토큰) - 학습:', S.length_report(train_samples))
print('길이(토큰) - 검증:', S.length_report(val_samples))
print('학습 카테고리: 정답을 후보에 끼워 넣은 건수', len(train_stats.category_injected))
if train_stats.too_long or val_stats.too_long:
    print('⚠️ 너무 길어 버린 샘플:', (train_stats.too_long + val_stats.too_long)[:10])
assert train_samples and val_samples, '학습/검증 샘플이 비었습니다. 데이터 양과 분할을 확인하세요.'
""")
code("""
# 과제별로 하나씩 - 학습 대상이 정답 부분에만 있는지 확인
for task in ('tool', 'intent', 'category'):
    s = next((x for x in train_samples if x.task == task), None)
    if s:
        print('=' * 100)
        print(builder.describe(s, tail_chars=350))
""")

# =============================================================================
md("""
## 8. 학습 전 기준선 평가

**평가셋(test)** 을 지금의 베이스 모델로 채점합니다. 학습 후 같은 평가셋으로 다시 채점해 비교합니다.
같은 평가셋·같은 프롬프트·같은 베이스면 결과를 캐시해 두고 다음부터는 건너뜁니다.

- **목표** 지표: 좋아져야 하는 것 (location · content · id · field · **keyword · period** · **카테고리 없음 재현율** 등)
- **방어** 지표: 떨어지면 안 되는 것 (의도·카테고리 정확도, JSON 파싱 성공률, 카테고리 없음 오판율)
- `location 지어냄 비율` · `찾기 조건 지어냄 비율` 은 낮을수록 좋습니다. 원문에 없는 위치를 채우면 현장 출동이 엉뚱한 곳으로 가고,
  원문에 없는 keyword·period 를 채우면 엉뚱한 민원이 수정·취소될 수 있습니다.
- `카테고리 없음 재현율` = 조회·수정·삭제에서 주제가 안 드러난 문장("어제 넣은 거 취소")을 `없음` 으로 고른 비율.
  `카테고리 없음 오판율` = 주제가 있는데("가로등 민원") `없음` 으로 고른 비율. 베이스 모델은 `없음` 선택지를 처음 보므로 낮게 나올 수 있습니다.

문장 하나당 도구 JSON 생성이 있어 오래 걸립니다. (평가셋 약 700건이면 수십 분~1시간 이상, `EVAL_LIMIT` 으로 조절)

**📂 필요한 파일**
- `data/train/cache/baseline_<지문>.json` — 있으면 채점을 건너뛰고 읽습니다. 지문은 평가셋 문장·정답, 프롬프트, 베이스 모델, 후보 설정으로 만들어서,
  하나라도 바뀌면 새로 채점합니다.

**📋 실행 결과**
- `평가(학습 전)` 진행 막대 → 끝나면 걸린 시간과 KV 캐시 상태(`ready: True`, `verified` 의 `same_choice: True` 면 정상)
- **지표 표** — 지표 / 역할(목표·방어·보조) / 좋은 방향(↑↓) / 점수 / 건수
  - 점수는 0~1 (0.85 = 85%). `건수` 가 작은 지표는 한 건 차이로 크게 흔들리니 함께 보세요.
  - 이 점수 자체로 잘한다·못한다를 판단하지 않습니다. **11단계와 비교할 출발점**입니다.
- 드라이브에 생기는 파일: `data/train/cache/baseline_<지문>.json`(지표), `baseline_<지문>_rows.csv`(문장별 채점 — 엑셀로 열어 볼 수 있음)
- 메모리: `baseline`(학습 전 채점 결과), `test_key`
""")
code("""
from training import evaluate as E

llm_engine.use_model(tokenizer, base_model)

def metrics_frame(*reports):
    rows = []
    for name, spec in E.METRICS.items():
        row = {'지표': name, '역할': spec['role'], '좋은 방향': '↑' if spec['better'] == 'high' else '↓'}
        for rep in reports:
            m = rep.metrics.get(name, {})
            row[rep.label] = m.get('value')
            row[f'{rep.label} 건수'] = m.get('n')
        rows.append(row)
    return pd.DataFrame(rows)

test_key = hashlib.sha256(json.dumps(
    [[r.id, r.text, r.intent, r.category, r.arguments] for r in test_recs]
    + [prompts.prompt_fingerprint(), settings.LLM_MODEL, CAND_SIG, str(runtime.compute_dtype())],
    ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
baseline_file = CACHE_DIR / f'baseline_{test_key}.json'

if baseline_file.exists():
    baseline = E.EvalReport.load(baseline_file)
    print(f'캐시된 기준선을 씁니다: {baseline_file}')
else:
    baseline = E.evaluate(test_recs, cands, label='학습 전')
    baseline.save(CACHE_DIR, baseline_file.stem)
    print(f'{baseline.seconds/60:.1f}분 | KV 캐시: {baseline.kv_cache}')
display(metrics_frame(baseline))
""")

# =============================================================================
md("""
## 9. LoRA 붙이기

어댑터는 **언어 모델의 선형층에만** 붙입니다. (비전 인코더·출력층·임베딩 제외)
아래 표에서 층 종류별로 몇 개 붙었는지, 학습되는 파라미터가 전체의 몇 %인지 확인하세요. (보통 1% 안팎)

**📂 필요한 파일** — 없음 (7단계의 `base_model` 에 붙입니다)

**📋 실행 결과**
- `붙은 층 수` — 수백 개가 정상 (층 수 × 층마다 선형층 여러 개)
- **층 종류별 개수 표** — `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` 같은 이름이 층 수만큼 나옵니다.
  Qwen3.5 는 일부 층이 선형 어텐션이라 다른 이름도 섞일 수 있습니다. `visual` · `vision` 이 들어간 이름은 **없어야** 합니다.
- `학습되는 파라미터` 비율 — 1% 안팎. (4bit 가중치는 개수가 절반으로 세어져 비율이 약간 크게 나옵니다)
- 학습 파라미터가 float32 가 아니면 여기서 멈춥니다. (T4 fp16 학습 오류를 미리 막는 검사)
- 메모리: `model`(LoRA 가 붙은 모델), `targets`. 파일은 만들지 않습니다.
""")
code("""
from training import trainer as T

model, targets = T.attach_lora(base_model, CFG)
print('붙은 층 수:', len(targets))
display(pd.DataFrame([T.summarize_targets(targets)]).T.rename(columns={0: '개수'}))
print(T.trainable_report(model))
not_fp32 = T.check_lora_dtype(model)
assert not not_fp32, f'float32 가 아닌 학습 파라미터가 있습니다(fp16 학습 오류 원인): {not_fp32[:5]}'
llm_engine.use_model(tokenizer, model)   # 평가·중간 점검이 이 모델을 쓰도록
gpu('LoRA 부착 후')
""")

# =============================================================================
md("""
## 10. 학습

출력 칸에서 제자리로 갱신됩니다.

- **왼쪽 그래프** — 회색/파랑: 학습 loss(원값/이동평균), 주황: 검증 loss
  - 둘 다 내려가다 완만해짐 → 정상
  - 학습은 내려가는데 검증이 오름 → 과적합. **검증 loss 가 가장 낮던 체크포인트가 자동 선택**됩니다
  - 둘 다 안 내려감 → 학습률이 너무 낮거나 데이터 형식 문제 (7단계 출력 확인)
  - 튀거나 `nan` → 학습률이 너무 높음
- **오른쪽 그래프** — 학습률 (warmup 후 cosine 으로 줄어듦)
- **경고** — 위 이상 신호를 자동으로 잡아 글로 보여 줍니다
- **중간 점검 표** — 검증 몇 번마다 고정된 몇 문장의 실제 출력을 뽑습니다. 학습이 진행될수록 `모델 출력` 이 `정답 JSON` 에 가까워져야 합니다

T4 에서는 샘플마다 프리픽스까지 매번 계산하므로 느립니다. 첫 몇 스텝 뒤 **남은 시간**이 표시됩니다.
중간에 멈추려면 셀 정지 버튼을 누르세요 — 그때까지의 가중치로 11단계를 진행할 수 있습니다.

**📂 필요한 파일**
- 없음 (7단계의 `train_samples`·`val_samples`, 9단계의 `model` 을 씀)
- 이어서 학습(`RESUME=True`)할 때만 `CKPT_DIR` 의 `checkpoint-*` 폴더가 필요합니다. 세션 디스크에 두었다면 런타임이 끊기면서 사라지므로, 이어서 하려면 처음부터 `CHECKPOINT_TO_DRIVE=True` 여야 합니다.

**📋 실행 결과**
- 첫 줄: `학습 샘플 N | 검증 샘플 N | epoch 당 N 스텝 | 총 N 스텝 | 검증 N 스텝마다 | fp16`
- 제자리 갱신 영역
  - 상태: `step 40/125 | epoch 0.32 | 경과 35.2분 | 남은 시간 약 75분` / 최근 학습 loss / 최근·최저 검증 loss
  - 그래프 (글꼴 문제로 영어 표기): `train loss`, `train loss (moving avg)`, `eval loss`, 오른쪽 `learning rate`
  - 경고: `nan`, 검증 loss 3회 연속 상승(과적합), 학습 loss 정체
  - 중간 점검 표: 검증 `spot_check_every` 번마다 고정 문장의 `예측 의도`, `정답 JSON`, `모델 출력`
- loss 숫자 읽는 법: 의도·카테고리 샘플은 정답이 토큰 2개라 금방 0 에 가까워지고, 곡선 모양은 주로 도구 JSON 샘플이 정합니다.
  **절대값보다 추세**(내려가다 완만해지는지)를 보세요.
- 끝나면 모델은 **검증 loss 가 가장 낮았던 체크포인트**의 가중치로 돌아가 있습니다.
- 다음 셀: 학습 시간, 가장 좋은 체크포인트와 그 검증 loss, 학습 중 경고, **첫 점검 문장의 출력이 스텝마다 어떻게 바뀌었는지**와 정답.
- 디스크에 생기는 파일: `CKPT_DIR/checkpoint-<스텝>/` (어댑터 + 옵티마이저 상태, 최근 2개만 유지). 기본은 세션 디스크라 드라이브 용량을 쓰지 않습니다.
- 메모리: `trainer`, `monitor`(loss 기록), `TRAIN_MINUTES`, `model`(학습된 모델로 교체됨)
""")
code("""
def pick_spot(recs, n):
    chosen, seen = [], set()
    for r in recs:                       # 도구 종류가 겹치지 않게 먼저 고름
        key = (r.tool, 'loc' if (r.arguments or {}).get('location') else '')
        if r.tool and key not in seen:
            chosen.append(r); seen.add(key)
    return (chosen + [r for r in recs if r not in chosen])[:n]

spot_recs = pick_spot(val_recs, SPOT_CHECK_N)
monitor = T.LiveMonitor(spot_records=spot_recs, cands=cands, spot_every=CFG.spot_check_every)
trainer, plan = T.build_trainer(model, tokenizer, train_samples, val_samples, CFG, CKPT_DIR, monitor)
print(f"학습 샘플 {len(train_samples)} | 검증 샘플 {len(val_samples)} | epoch 당 {plan['steps_per_epoch']} 스텝 "
      f"| 총 {plan['total_steps']} 스텝 | 검증 {plan['eval_steps']} 스텝마다 | {'bf16' if plan['bf16'] else 'fp16'}")

resume = T.latest_checkpoint(CKPT_DIR) if RESUME else None
if resume:
    print('이어서 학습:', resume)

t0 = time.time()
interrupted = False
try:
    trainer.train(resume_from_checkpoint=resume)
except KeyboardInterrupt:
    interrupted = True
    print('\\n중단했습니다. 지금까지의 가중치로 다음 단계를 진행할 수 있습니다. (가장 좋은 체크포인트 자동 선택은 적용되지 않음)')
TRAIN_MINUTES = (time.time() - t0) / 60
model = T.finish_training(trainer)       # 학습용 autocast 를 풀어 평가가 서버와 같은 경로로 돌게 함
llm_engine.use_model(tokenizer, model)
gpu('학습 후')
""")
code("""
state = trainer.state
best_ckpt, best_loss = state.best_model_checkpoint, state.best_metric
print(f'학습 시간 {TRAIN_MINUTES:.1f}분 | 최종 스텝 {state.global_step}')
print(f'가장 좋은 체크포인트: {best_ckpt} (검증 loss {best_loss})')
if monitor.warnings:
    print('\\n학습 중 경고:')
    for w in monitor.warnings: print(' -', w)
if monitor.history['spot']:
    print('\\n중간 점검 - 첫 번째 문장의 출력 변화:')
    for h in monitor.history['spot']:
        item = h['items'][0]
        print(f"  step {h['step']:>4} | {item.get('예측 의도')} | {str(item.get('모델 출력', ''))[:110]}")
    print(f"  정답              | {spot_recs[0].intent} | {prompts.format_tool_call(spot_recs[0].tool, spot_recs[0].tool_arguments()) if spot_recs[0].tool else '-'}")
""")

# =============================================================================
md("""
## 11. 학습 후 평가 · 전후 비교 · 판정

8단계와 **같은 평가셋**을 학습된 모델로 다시 채점합니다.

판단 규칙
- 차이가 2%p 보다 작거나 **평가 한 건(1/n) 이하**면 `비슷` — 평가셋이 작으면 한 건으로 점수가 크게 흔들립니다
- **성공** = 목표 지표 하나 이상 `좋아짐` + 방어 지표 `나빠짐` 없음 + `location 지어냄 비율` 이 나빠지지 않음

**📂 필요한 파일** — 없음 (8단계의 `baseline`, 학습된 `model`). 채점 시간은 8단계와 같습니다.

**📋 실행 결과**
- **전후 비교표** — 지표 / 역할 / 학습 전 / 학습 후 / 평가 건수 / 변화 / 판단. `판단` 칸이 좋아짐(초록)·나빠짐(빨강)·비슷.
- **판정** — `성공 - 운영에 써 볼 만합니다` 또는 `보류 - …` 와 이유 (좋아진 목표 지표, 나빠진 방어 지표 등)
- **태그별 표** (다음 셀) — `위치없음`, `경계`, `id있음`, `구어체` 등 어려운 경우만 따로 본 학습 전후 점수
- **틀린 사례** (그다음 셀) — 위치를 지어낸 것 / 위치가 틀린 것 / content 점수가 낮은 것 / 의도 오답 / JSON 파싱 실패 / id 오답을 문장·정답·예측으로 나란히
  - 같은 유형(예: 줄임말, 장소가 두 개인 문장)이 몰려 있으면 그 유형 데이터를 보강하면 됩니다.

| 결과 | 할 일 |
|---|---|
| 성공 | 12단계 저장 → 13단계 확인 → 서버 적용 |
| 목표 ↑ 인데 의도·카테고리 ↓ | 4단계 `MIX` 의 intent · category 비율을 올려 재학습 |
| 전부 `비슷` | 데이터·epoch 부족 또는 평가셋이 작음 |
| 지어냄 ↑ | 위치 없는 문장(정답 `""`)을 더 넣어 재학습. 이 어댑터는 쓰지 않기 |

- 파일은 만들지 않습니다 (12단계에서 저장). 메모리: `after`(학습 후 채점), `table`, `VERDICT`, `reasons`
""")
code("""
after = E.evaluate(test_recs, cands, label='학습 후')
table = pd.DataFrame(E.compare(baseline, after))

def color(v):
    return {'좋아짐': 'background-color:#d9f2d9', '나빠짐': 'background-color:#f8d7da'}.get(v, '')
display(table.style.map(color, subset=['판단']).format({'학습 전': '{:.3f}', '학습 후': '{:.3f}', '변화': '{:+.3f}'}, na_rep='-'))

VERDICT, reasons = E.verdict(E.compare(baseline, after))
print('\\n판정:', VERDICT)
for r in reasons: print(' -', r)
""")

md("태그별로 쪼개 보기 — 전체 평균에 가려진 어려운 경우(위치 없음, 경계 문장, 카테고리없음·키워드없음·시점있음 등)를 따로 봅니다.")
code("""
KEY = ['의도 정확도', '카테고리 정확도', 'JSON 파싱 성공률', 'location 정확 일치', 'location 지어냄 비율',
       '위치없음 빈칸 유지율', 'content 유사도', 'complaint_id 정확 일치', 'keyword 정확 일치', 'period 정확 일치',
       '찾기 조건 지어냄 비율']
rows = []
for tag in sorted(set(after.by_tag) | set(baseline.by_tag)):
    for name in KEY:
        b = baseline.by_tag.get(tag, {}).get(name, {})
        a = after.by_tag.get(tag, {}).get(name, {})
        if a.get('value') is None and b.get('value') is None:
            continue
        rows.append({'태그': tag, '지표': name, '학습 전': b.get('value'), '학습 후': a.get('value'), '건수': a.get('n')})
display(pd.DataFrame(rows))
""")

md("""
틀린 사례 보기 — 숫자만으로는 **왜** 틀렸는지 모릅니다. 틀린 유형이 몰려 있으면 그 유형의 학습 데이터를 보강하세요.
`content` 는 자동 점수(글자 ROUGE-L)가 거칠기 때문에 **사람이 직접 몇 건 읽어 보는 것**이 가장 정확합니다.
""")
code("""
def show(task, cols, n=8):
    rows = E.failures(after, task, n)
    print(f'--- {task}: {len(rows)}건 (최대 {n}건 표시)')
    if rows: display(pd.DataFrame(rows)[[c for c in cols if c in rows[0]]])

show('hallucination', ['id', 'text', 'gold_location', 'pred_location', 'gold_keyword', 'pred_keyword', 'gold_period', 'pred_period'])
show('location', ['id', 'text', 'gold_location', 'pred_location'])
show('content', ['id', 'text', 'gold_content', 'pred_content', 'content_score'])
show('keyword', ['id', 'text', 'gold_keyword', 'pred_keyword'])
show('period', ['id', 'text', 'gold_period', 'pred_period'])
show('intent', ['id', 'text', 'gold_intent', 'pred_intent', 'intent_score'])
show('parse', ['id', 'text', 'raw'])
show('id', ['id', 'text', 'gold_id', 'pred_id'])
show('category', ['id', 'text', 'candidates', 'gold_category', 'pred_category', 'category_score'])
""")

# =============================================================================
md("""
## 12. 저장 · 버전 비교

`adapters/<RUN_NAME>/` 에 저장합니다.

| 파일 | 내용 |
|---|---|
| `adapter_model.safetensors`, `adapter_config.json` | 학습된 어댑터 (서버가 읽는 것) |
| `run_info.json` | 학습 조건 · 데이터 지문 · 점수 · 판정. **서버가 로드할 때 베이스 모델 / 프롬프트 지문을 대조해 다르면 경고**합니다 |
| `eval_before.json`, `eval_after.json`, `*_rows.csv` | 평가 결과와 문장별 채점 (CSV 는 엑셀로 열어 content 를 사람이 검토) |
| `loss.png`, `loss_history.json` | 학습 곡선 |

덮어쓰지 않고 버전마다 새 폴더를 만드세요. 서버에 적용하거나 되돌리는 건 `.env` 의 `LLM_ADAPTER_PATH` 한 줄입니다.

**📂 필요한 파일** — 없음 (메모리의 학습 결과를 파일로 씁니다)

**📋 실행 결과**
- `저장 완료: adapters/<이름> (N MB)` 와 폴더 목록. 어댑터 크기는 설정에 따라 수십~백수십 MB 입니다.
- 드라이브에 생기는 파일 (`adapters/<이름>/`): 위 표의 파일들 + peft 가 만드는 `README.md`
- `run_info.json` 에 들어가는 것: 베이스 모델, 프롬프트 지문, 학습 설정, 데이터 파일 지문(`data_sha`), 레코드·샘플 수,
  채점한 평가셋 id(`test_ids`), 최저 검증 loss, 학습 전후 지표, 판정과 이유, 경고
- **버전 비교 표** (다음 셀) — 지금까지 저장한 버전마다 판정·설정·주요 지표 한 줄씩. `프롬프트 일치` 가 False 인 버전은 지금 프롬프트와 다르게 학습된 것이라 재학습이 필요합니다.
""")
code("""
metrics_after = after.metrics
info = {
    'run_name': RUN_NAME,
    'verdict': VERDICT,
    'verdict_reasons': reasons,
    'interrupted': interrupted,
    'train_minutes': round(TRAIN_MINUTES, 1),
    'train_config': CFG.to_dict(),
    'mix': MIX,
    'split_ratios': SPLIT_RATIOS,
    'data_file': DATA_FILE,
    'data_sha': DATA_SHA,
    'record_counts': {k: len(v) for k, v in parts.items()},
    'sample_counts': {'train': len(train_samples), 'val': len(val_samples)},
    'train_sample_limit': TRAIN_SAMPLE_LIMIT,
    'val_sample_limit': VAL_SAMPLE_LIMIT,
    'train_task_counts': dict(_Counter(x.task for x in train_samples)),
    'candidate_signature': CAND_SIG,
    'test_key': test_key,
    'test_ids': TEST_IDS,
    'eval_limit': EVAL_LIMIT,
    'lora_targets': T.summarize_targets(targets),
    'best_checkpoint': best_ckpt,
    'best_eval_loss': best_loss,
    'final_step': trainer.state.global_step,
    'metrics_before': baseline.metrics,
    'metrics_after': metrics_after,
    'warnings': monitor.warnings,
}
T.save_run(model, RUN_DIR, info)
baseline.save(RUN_DIR, 'eval_before')
after.save(RUN_DIR, 'eval_after')
monitor.save_figure(RUN_DIR / 'loss.png')
(RUN_DIR / 'loss_history.json').write_text(json.dumps(monitor.history, ensure_ascii=False, default=str), encoding='utf-8')

size = sum(p.stat().st_size for p in RUN_DIR.rglob('*') if p.is_file()) / 1024**2
print(f'저장 완료: {RUN_DIR} ({size:.0f}MB)')
!ls -la {RUN_DIR}
""")
code("""
# 지금까지 만든 버전들
runs = pd.DataFrame(T.list_runs(ADAPTERS_DIR))
display(runs if not runs.empty else '저장된 버전이 없습니다.')
# '프롬프트 일치' 가 False 인 버전은 지금 prompts.py 와 다른 프롬프트로 학습된 것입니다. 서버에 쓰려면 재학습하세요.
""")

md("""
### 서버에 적용하기

아래 셀의 `APPLY = True` 로 바꿔 실행하면 `.env` 의 `LLM_ADAPTER_PATH` 를 이번 어댑터로 바꿉니다.
서버 노트북(`colab_backend.ipynb`)을 다시 실행하면 이 어댑터가 올라갑니다. **되돌리려면 값을 비우면** 베이스 모델로 돌아갑니다.
판정이 `보류` 면 적용하지 않는 것을 권합니다.

**📂 필요한 파일** — `.env` (없으면 3-1 이 이미 만들어 둠)

**📋 실행 결과**
- `현재 LLM_ADAPTER_PATH = …` 로 지금 설정을 보여 줍니다.
- `APPLY = True` 일 때만 드라이브의 `.env` 에서 그 줄을 이번 어댑터 경로로 바꾸고 `변경 -> adapters/<이름>` 을 출력합니다.
""")
code("""
APPLY = False
ADAPTER_VALUE = str(RUN_DIR)          # backend 폴더 기준 상대경로

env_path = Path('.env')
lines = env_path.read_text(encoding='utf-8').splitlines()
current = next((l.split('=', 1)[1] for l in lines if l.startswith('LLM_ADAPTER_PATH=')), '')
print('현재 LLM_ADAPTER_PATH =', current or '(비어 있음 - 베이스 모델)')
if APPLY:
    lines = [l for l in lines if not l.startswith('LLM_ADAPTER_PATH=')] + [f'LLM_ADAPTER_PATH={ADAPTER_VALUE}']
    env_path.write_text('\\n'.join(lines) + '\\n', encoding='utf-8')
    print('변경 ->', ADAPTER_VALUE)
""")

# =============================================================================
md("""
## 13. (권장) 재시작 후 운영 경로로 최종 확인

학습 세션의 모델은 학습용 처리(gradient checkpointing, LoRA dropout 설정 등)가 섞여 있어 **서버에서 올린 모델과 미세하게 다를 수 있습니다.**
런타임을 재시작한 뒤, 서버와 **똑같은 경로**(`llm_engine.get_model()` + `LLM_ADAPTER_PATH`)로 올려서 다시 확인합니다.

1. **[런타임] → [세션 다시 시작]**
2. 아래 `CHECK_RUN` 에 확인할 어댑터 이름을 적고 실행

확인하는 것
- 어댑터 학습 조건 대조 결과 (`mismatches` 가 비어 있어야 함)
- KV 캐시 검증 통과 여부 (어댑터를 얹은 상태에서도 캐시 결과가 전체 계산과 같은지)
- 평가셋 점수가 11단계와 비슷한지 (같은 판정이 나오는지)
- 실제 파이프라인(②→⑥) 동작

**📂 필요한 파일**
- `adapters/<이름>/` — 12단계 결과 (`adapter_model.safetensors`, `adapter_config.json`, `run_info.json`)
- `data/train/records.jsonl` — **학습 때와 같은 파일**이어야 같은 평가셋이 나옵니다. 그 사이 파일을 바꿨다면 경고가 나옵니다.
- `data/train/cache/candidates.json` — 6단계 캐시 (없으면 bge-m3 로 다시 계산)
- 재시작 후라 **0~3-1 단계는 다시 할 필요 없이** 이 셀부터 실행하면 됩니다. (설치는 남아 있음)

**📋 실행 결과**
- 첫 셀: `확인할 어댑터`, **어댑터 조건 대조** (`mismatches: []` 이면 통과. 항목이 있으면 베이스 모델·프롬프트·사고 모드 중 무엇이 학습 때와 다른지),
  **KV 캐시 상태** (`ready: True`, `verified.same_choice: True`, `disabled_reason: null` 이면 어댑터를 얹어도 캐시가 정상)
- 둘째 셀: 같은 평가셋을 어댑터 켜고/끄고 채점한 비교표 + `학습 세션 값` 열. **판정(운영 경로)과 판정(학습 세션)이 같으면 통과**입니다.
  채점을 두 번 하므로 8단계의 약 두 배 시간이 걸립니다.
- 셋째 셀: 문장 3개를 ②→⑥ 전체 파이프라인에 넣은 결과 (위치 있는 접수 / 위치 없는 접수 / 잡담 반려)
- 드라이브에 생기는 파일: `data/complaints.db` 에 셋째 셀의 접수 민원이 실제로 들어갑니다 (`user_id='train-check'`). 테스트 기록이라 지워도 됩니다.
""")
code("""
CHECK_RUN = ''          # 예: 'v1_0929'. 비우면 가장 최근 버전

from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/backend
import os, json
from pathlib import Path
import pandas as pd

runs = sorted([p for p in Path('adapters').glob('v*') if (p / 'run_info.json').exists()], key=lambda p: p.stat().st_mtime)
assert CHECK_RUN or runs, 'adapters/ 에 저장된 어댑터가 없습니다. 12단계까지 먼저 실행하세요.'
run_dir = Path('adapters') / CHECK_RUN if CHECK_RUN else runs[-1]
os.environ['LLM_ADAPTER_PATH'] = str(run_dir)    # .env 보다 우선합니다 (이 세션에서만)
print('확인할 어댑터:', run_dir)

from app.services import llm_engine, case_store
from training import records as R, samples as S, evaluate as E

info = json.loads((run_dir / 'run_info.json').read_text(encoding='utf-8'))
assert not llm_engine.is_loaded(), '학습 세션의 모델이 아직 메모리에 있습니다. [런타임] → [세션 다시 시작] 후 이 셀부터 실행하세요.'
llm_engine.warmup()                               # 서버 기동과 같은 경로: 로드 + 어댑터 + KV 캐시 prefill/검증
st = llm_engine.status()
print('어댑터 조건 대조:', json.dumps(st['adapter_check'], ensure_ascii=False, indent=1))
print('KV 캐시        :', st['kv_cache'])
""")
code("""
# 학습 때와 같은 평가셋으로 어댑터 켜고/끄고 채점 (같은 세션에서 전후 비교)
import hashlib
records = R.load_records(info['data_file'])
if hashlib.sha256(Path(info['data_file']).read_bytes()).hexdigest()[:16] != info.get('data_sha'):
    print('⚠️ 학습 뒤 데이터 파일이 바뀌었습니다. 평가셋이 학습 때와 다를 수 있어 점수 비교가 정확하지 않습니다.')
parts = R.split_records(records, tuple(info['split_ratios']))
test_recs = parts['test']
if info.get('test_ids'):
    wanted = set(info['test_ids'])
    test_recs = [r for r in test_recs if r.id in wanted]   # 학습 때 채점한 문장과 같게 (EVAL_LIMIT 포함)
print(f'채점할 평가셋: {len(test_recs)}건')
cands = S.compute_candidates(records, cache_path=Path('data/train/cache/candidates.json'))

with E.adapter(True):
    prod_after = E.evaluate(test_recs, cands, label='운영경로-어댑터')
with E.adapter(False):
    prod_before = E.evaluate(test_recs, cands, label='운영경로-베이스')

table = pd.DataFrame(E.compare(prod_before, prod_after))
table['학습 세션 값'] = [ (info['metrics_after'].get(n) or {}).get('value') for n in table['지표'] ]
display(table)
print('판정(운영 경로):', E.verdict(E.compare(prod_before, prod_after))[0])
print('판정(학습 세션):', info['verdict'])
""")
code("""
# 실제 파이프라인 (②→⑥) - 어댑터를 얹은 상태로
from app.services import pipeline
for 문장 in ['아파트 앞 도로에 포트홀이 생겨서 차가 덜컹거립니다. 보수해 주세요.',
            '동네 골목이 너무 어두워요. 가로등 좀 더 달아주세요.',
            '오늘 점심 뭐 먹지']:
    r = pipeline.run(문장, user_id='train-check')
    print(pipeline.render_result_text(r))
    print('-' * 80)
""")

# =============================================================================
md("""
## 문제가 생기면

| 증상 | 원인과 해결 |
|---|---|
| `CUDA out of memory` (학습 중) | 먼저 4단계 `CFG` 의 `gradient_checkpointing=True` 인지 확인하고 **런타임 재시작 후 3-1부터**. 그래도 나면 `max_length` 를 줄이기(긴 샘플 제외). bge-m3 를 내렸는지(`embedder.unload()`) 확인 |
| loss 가 `nan` / 크게 튐 | `learning_rate` 를 1e-4 로. 그래도 나면 `full_kbit_prep=True` (비양자화 층을 float32 로, 메모리 더 씀). T4 는 fp16 이라 bf16 GPU 보다 불안정할 수 있습니다 |
| loss 가 거의 안 줄어듦 | 7단계 출력에서 `학습 대상` 이 정답 부분에만 있는지 확인 → 맞으면 `learning_rate` 를 올리기 |
| 검증 loss 가 계속 오름 | 과적합. `epochs` 를 줄이거나 데이터를 늘리기. 가장 좋은 체크포인트는 자동 선택됨 |
| 목표 지표는 오르는데 의도 정확도가 떨어짐 | `MIX` 의 intent · category 비율을 올려 기존 능력을 더 붙잡기 |
| 전후 차이가 전부 `비슷` | 평가셋이 작거나 데이터가 부족. 도구별로 수백 건 이상 모아 다시 |
| 학습이 너무 느림 | 정상입니다(T4, 샘플마다 프리픽스까지 계산). 4단계 `TRAIN_SAMPLE_LIMIT` 을 줄이세요 - 학습 시간이 거의 비례합니다 |
| 런타임이 끊김 | `CHECKPOINT_TO_DRIVE=True` 로 두고 다시 학습하면, 다음에는 `RUN_NAME` 을 같게 + `RESUME=True` 로 이어서 할 수 있습니다 |
| 13단계에서 `mismatches` 가 나옴 | 학습 뒤 `prompts.py` 나 `.env` 모델 설정이 바뀜. 그 어댑터는 재학습하세요 |
| `KeyError: 'qwen3_5'` | transformers 가 낡음. 2단계 설치 후 3단계 재시작을 했는지 확인 |
""")


def build() -> dict:
    cells = []
    for kind, text in CELLS:
        lines = text.split("\n")
        source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "colab_train.ipynb"
    out.write_text(json.dumps(build(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written {out} ({len(CELLS)} cells)")
