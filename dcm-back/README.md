# Document AI Backend

> **팀원용 빠른 안내** — Colab 에서 ⑤ 판정 모델(Qwen3.5-4B)을 학습시키는 방법은 바로 아래 **[0. Colab 에서 모델 학습하기](#0-colab-에서-모델-학습하기-팀원용)**,
> Qwen 대신 다른 모델로 시험해 보려면 **[0-9. 다른 모델로 바꿔서 학습하기](#0-9-다른-모델로-바꿔서-학습하기)** 를 보세요.
> 서버 구조·설치·API 설명은 1장부터 이어집니다.

---

## 0. Colab 에서 모델 학습하기 (팀원용)

PC 에 GPU 가 없어도 됩니다. **구글 드라이브에 이 폴더를 올리고, Colab 에서 `colab_train.ipynb` 를 위에서부터 실행**하면
LoRA 어댑터가 드라이브의 `backend/adapters/<이름>/` 에 저장됩니다.

```
① backend.zip 을 내 드라이브에 올려 풀기  →  ② Colab 에서 colab_train.ipynb 열기  →  ③ GPU 고르기
→ ④ 0~3 단계(설치·재시작)  →  ⑤ 3-1 부터 13 단계까지 차례로  →  ⑥ adapters/<이름>/ 결과 공유
```

### 0-1. 준비물

| 항목 | 설명 |
|---|---|
| 구글 계정 | 드라이브 여유 공간 **1GB 이상** 권장 (코드·데이터 약 10MB + 캐시 + 어댑터 버전마다 수십~백수십 MB) |
| `backend.zip` | 이 폴더 전체를 압축한 파일. `.env` 는 들어 있지 않습니다 (노트북이 자동으로 만듦) |
| Colab | 무료도 되지만(T4) 학습이 끊기기 쉽습니다. **Colab Pro 의 L4** 를 권장합니다 (0-3 참고) |
| Hugging Face 토큰 | Qwen3.5-4B 는 **필요 없음**. Llama·Gemma 같은 승인제 모델로 바꿀 때만 필요 (0-9) |

### 0-2. 드라이브에 파일 올리기

노트북은 **`내 드라이브/backend/`** 경로를 씁니다. (`/content/drive/MyDrive/backend`)
폴더 이름이나 위치가 다르면 노트북의 `%cd /content/drive/MyDrive/backend` 줄을 모두 고쳐야 하니, 가능하면 이 위치를 그대로 쓰세요.

**방법 A (권장) — zip 을 올리고 Colab 에서 풀기.** 작은 파일 수백 개를 브라우저로 올리는 것보다 훨씬 빠릅니다.

1. [drive.google.com](https://drive.google.com) → **내 드라이브** 맨 위(폴더 안이 아닌 곳)에 `backend.zip` 을 끌어다 놓습니다.
2. Colab 에서 새 노트북을 하나 열고 (파일 → 새 노트북) 아래 셀을 실행합니다. 드라이브 접근 허용 창이 뜨면 허용하세요.

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !unzip -q -o /content/drive/MyDrive/backend.zip -d /content/drive/MyDrive/
   !ls /content/drive/MyDrive/backend
   ```

   `app  colab_backend.ipynb  colab_train.ipynb  data  requirements.txt ...` 가 보이면 끝입니다. 이 임시 노트북은 닫아도 됩니다.
   (`-o` 는 같은 이름 파일을 덮어씁니다. zip 에 `.env` 가 없으므로 **이미 쓰던 `.env`·`adapters/`·캐시는 그대로** 남습니다.)

**방법 B — 폴더째 올리기.** PC 에서 zip 을 풀고, 드라이브 웹에서 **새로 만들기 → 폴더 업로드** 로 `backend` 폴더를 고릅니다.
`내 드라이브/backend/app/...` 처럼 **`backend` 가 한 겹**이어야 합니다. (`backend/backend/app` 이 되면 안 됩니다)

**팀원끼리 폴더를 공유하지 마세요.** 공유 폴더를 함께 쓰면 `.env`, 캐시(`data/train/cache/`), `adapters/`, `data/complaints.db` 를
여러 사람이 동시에 덮어씁니다. **각자 자기 드라이브에 따로** 올리고, 결과만 0-8 처럼 주고받으세요.

올린 뒤 드라이브 구조:

```
내 드라이브/
└── backend/
    ├── app/                       서버 코드 (학습·평가가 그대로 씀) ✅ 필수
    ├── training/                  학습 코드 ✅ 필수
    ├── data/
    │   ├── train/records.jsonl    학습 정답 데이터 (4,643건) ✅
    │   └── cases.csv              벡터DB 사례 (④ 후보 재정렬)
    ├── requirements.txt           ✅ 필수
    ├── requirements-model.txt     ✅ 필수
    ├── colab_train.ipynb          ← 학습 노트북
    └── colab_backend.ipynb        ← 서버(API) 노트북
```

### 0-3. 노트북 열기 · GPU 고르기

1. 드라이브에서 `backend/colab_train.ipynb` 를 **더블클릭 → 연결 앱: Google Colaboratory** 로 엽니다.
   (처음이면 "연결할 앱 더보기"에서 Colaboratory 를 설치. 또는 [colab.research.google.com](https://colab.research.google.com) → 파일 → 노트북 열기 → Google Drive 탭)
2. 메뉴 **런타임 → 런타임 유형 변경 → 하드웨어 가속기** 에서 GPU 를 고르고 저장합니다.

| GPU | 메모리 | 속도 (기본 설정 1,000 샘플 = 125 스텝) | 비고 |
|---|---|---|---|
| T4 (무료) | 16GB | 가장 느림. 학습만 몇 시간 | 무료는 사용 시간 제한·중간 끊김이 잦아 끝까지 가기 어렵습니다 |
| **L4 (Pro)** | 24GB | 스텝당 약 22초 → 학습 약 45~50분 | **권장.** 컴퓨팅 단위를 시간당 약 1.7 정도 씀 |
| A100 (Pro) | 40GB | L4 보다 빠름 | 컴퓨팅 단위를 가장 많이 씀 |

학습 외에 **8단계(학습 전 평가)와 11단계(학습 후 평가)** 가 각각 평가셋 전부(약 700건) 기준 꽤 오래 걸립니다(L4 에서도 수십 분 이상).
처음 시험 삼아 돌릴 때는 4단계에서 `EVAL_LIMIT = 300` 처럼 줄이세요. 8단계 결과는 캐시되므로 같은 조건이면 두 번째부터는 건너뜁니다.

### 0-4. 실행 순서

**위에서부터 한 셀씩** 실행합니다 (셀 왼쪽 ▶ 또는 `Shift+Enter`). 각 단계 위의 설명 칸에 **필요한 파일 / 실행 결과(정상일 때 보이는 것)** 가 적혀 있으니 결과가 같은지 확인하며 넘어가세요.

| 단계 | 할 일 | 정상이면 |
|---|---|---|
| 0 | GPU 확인 | `NVIDIA L4` 같은 GPU 이름이 보임 |
| 1 | 드라이브 연결 | 계정 선택 → 허용 → `Mounted at /content/drive`, `ls` 에 `app`, `training` … |
| 2 | 설치 (5~10분) | 마지막에 `설치 완료`. 중간의 빨간 `pip's dependency resolver` 경고는 무시 |
| 3 | **런타임 재시작** | `세션이 다운되었습니다` 알림 — **일부러 끊는 것**입니다 |
| 3-1 | 재시작 후 경로·환경 확인 | `.env` 가 없으면 자동 생성. 마지막 줄 `문제 없음` |
| 3-2 | 필수 파일 점검 | `필수 파일 모두 있음`, `records.jsonl` 4,643줄 |
| 4 | 이번 학습 설정 | 처음에는 **기본값 그대로** (아래 0-5) |
| 5 | 정답 데이터 검증·분할 | `검증 오류 0건` |
| 6 | ④ 후보 계산 (bge-m3) | 처음 수 분~십수 분, 다음부터 캐시로 몇 초. `④ 놓침` 이 수십 건 이하 |
| 6-1 · 6-2 | (선택) ④ 설정 측정·반영 | 기본값(`multi` · `low_confidence` · 0.65 · 0.2)이 이미 측정해서 고른 값이라 **건너뛰어도 됩니다** |
| 7 | Qwen 올리기 + 학습 샘플 | 첫 실행은 다운로드 5~15분. `【학습 대상 ▶ … ◀】` 안에 정답과 끝 토큰만 |
| 8 | 학습 전 기준선 평가 | 지표 표 (비교 출발점) |
| 9 | LoRA 붙이기 | `visual`·`vision` 이름이 없어야 함 |
| 10 | **학습** | loss 그래프가 내려가다 완만해짐. 남은 시간 표시 |
| 11 | 학습 후 평가·판정 | `판정: 성공` 또는 `보류 - 이유` |
| 12 | 저장 · 버전 비교 | `저장 완료: adapters/<이름>` |
| (12 아래) | 서버에 적용 | `APPLY = True` 로 실행하면 `.env` 의 `LLM_ADAPTER_PATH` 가 바뀜 |
| 13 | **런타임 재시작 후** 운영 경로로 재확인 | `mismatches: []`, 판정(운영 경로)과 판정(학습 세션)이 같음 |

**런타임이 끊겼을 때** (오래 자리를 비움, 메모리 부족 등): 설치는 대부분 남아 있으므로 **3-1 부터** 다시 실행하면 됩니다.
`ModuleNotFoundError` 나 `KeyError: 'qwen3_5'` 가 나면 세션이 새로 할당된 것이니 **2 → 3 → 3-1** 순서로 다시 하세요.
`Transport endpoint is not connected` 는 드라이브 연결이 끊긴 것 — 런타임 → 세션 다시 시작 후 3-1 부터.

### 0-5. 4단계 설정 — 무엇을 바꿔 볼까

처음에는 기본값으로 한 번 끝까지 돌려 보고, 그다음부터 **한 번에 하나씩만** 바꾸세요. 여러 개를 같이 바꾸면 무엇 때문에 좋아졌는지 알 수 없습니다.

| 값 | 기본 | 언제 바꾸나 |
|---|---|---|
| `RUN_NAME` | `''` (자동 `v번호_날짜`) | 구분하고 싶으면 `v5_lr1e4` 처럼. **`v` 로 시작**해야 13단계가 자동으로 찾습니다 |
| `TRAIN_SAMPLE_LIMIT` | 1000 | 학습 시간이 거의 비례. 본 학습은 `None`(전부, 약 4천~5천 개 → L4 에서 3~4시간) |
| `EVAL_LIMIT` | None | 시험 삼아 돌릴 때 300 |
| `learning_rate` | 2e-4 | loss 가 튀거나 `nan` → 1e-4 |
| `epochs` | 1 | 데이터를 전부 쓸 때 2~3 |
| `MIX` | 도구 0.6 / 의도 0.2 / 카테고리 0.2 | 학습 후 의도·카테고리 정확도가 떨어지면 의도·카테고리 비중을 올리기 |
| `NONE_BOOST` | 3 | `카테고리 없음 재현율` 이 낮으면 올리기 |
| `gradient_checkpointing` | True | **끄지 마세요** — L4 에서도 메모리 부족이 났습니다 (A100 에서만 시도) |
| `CHECKPOINT_TO_DRIVE` | False | 오래 걸리는 학습(전부 사용)이면 True — 끊겨도 `RESUME=True` + 같은 `RUN_NAME` 으로 이어서 |

### 0-6. 데이터를 고쳐서 다시 학습할 때

- 정답 데이터는 `data/train/records.jsonl` 한 파일입니다. 형식·라벨링 규칙은 노트북 5단계 설명과 `training/records.py` 맨 위에 있습니다.
  특히 `keyword` 는 **대상 명사만** (`101동 엘리베이터 고장` → `엘리베이터`), `location`·`keyword`·`period` 는 **원문에 있는 글자 그대로**여야 검증을 통과합니다.
- 파일을 고쳐 드라이브에 다시 올렸으면 **런타임 → 세션 다시 시작 → 3-1 부터**. 5단계가 검증하고, 6·8단계는 바뀐 문장만 다시 계산합니다.
- `prompts.py`·`categories.py` 를 고치면 **프롬프트 지문**이 바뀌어 이전 어댑터는 서버에서 경고가 나고 다시 학습해야 합니다.
- `categories.py`·`cases.csv` 를 고치면 ④ 후보가 바뀌므로 6단계가 후보를 자동으로 다시 계산합니다.

### 0-7. 결과 읽기

- **판정이 `성공`** (목표 지표가 하나 이상 좋아지고, 방어 지표·location 지어냄이 나빠지지 않음) → 12단계 저장 → 13단계 확인.
- **`보류`** 면 11단계 아래의 **틀린 사례 표**를 먼저 보세요. 같은 유형이 몰려 있으면 그 유형의 데이터를 보강하는 게 가장 효과적입니다.
- 10단계 그래프보다 **11단계 전후 비교표가 기준**입니다. loss 는 과제마다 크기가 달라 모양이 들쭉날쭉할 수 있습니다.
- 버전별 비교는 12단계 아래 **버전 비교 표** (판정·베이스 모델·주요 지표·`프롬프트 일치`).

### 0-8. 결과를 팀과 공유하기

드라이브의 `backend/adapters/<이름>/` 폴더 하나가 결과 전부입니다.

| 파일 | 내용 |
|---|---|
| `adapter_model.safetensors`, `adapter_config.json` | 어댑터 (서버가 읽는 것, 수십~백수십 MB) |
| `run_info.json` | 베이스 모델·프롬프트 지문·학습 설정·데이터 지문·전후 지표·판정 |
| `eval_before/after.json`, `*_rows.csv` | 지표와 문장별 채점 (CSV 는 엑셀로 열림) |
| `loss.png` | 학습 곡선 |

- 공유할 때는 이 폴더를 통째로 (우클릭 → 다운로드 하면 zip) 보내고, **어떤 설정을 바꿨는지 한 줄**을 함께 적어 주세요. (`run_info.json` 에도 남아 있음)
- 받은 사람은 자기 드라이브의 `backend/adapters/` 아래에 폴더째 넣고, `.env` 의 `LLM_ADAPTER_PATH=adapters/<이름>` 으로 바꾸면 서버(`colab_backend.ipynb`)가 그 어댑터로 올라갑니다.
- 어댑터는 **같은 베이스 모델 + 같은 프롬프트(`prompts.py`·`categories.py`)** 에서만 제대로 동작합니다. 서버가 올릴 때 `run_info.json` 과 대조해 다르면 경고합니다.

### 0-9. 다른 모델로 바꿔서 학습하기

코드는 특정 모델에 묶여 있지 않고 **`.env` 의 `LLM_MODEL` 한 줄**로 베이스 모델을 정합니다. 학습·평가·서버가 모두 이 값을 읽습니다.

#### 어떤 모델이 되나

아래 조건을 모두 만족해야 합니다.

| 조건 | 이유 | 안 맞으면 |
|---|---|---|
| **대화형(Instruct / chat / `-it`) 모델** | 채팅 템플릿으로 프롬프트를 조립합니다 | base 모델은 템플릿이 없어 7단계에서 오류 |
| 채팅 템플릿이 **system 역할**을 받음 | 고정 프리픽스(역할·도구·카테고리·의도 정의)를 system 에 넣습니다 | `System role not supported` 같은 오류 (예: Gemma 2. Gemma 3 는 가능) |
| 숫자 `1`~`6` 이 **각각 다른 토큰** | 의도·카테고리를 "다음 토큰 1개의 확률"로 고릅니다 | 7단계에서 `번호 토큰이 서로 겹칩니다` 오류 — 그 모델은 쓸 수 없음 |
| transformers 가 지원 | `AutoModelForCausalLM` (안 되면 자동으로 `AutoModelForImageTextToText`) 으로 올립니다 | 저장소 자체 코드가 필요한 모델은 `LLM_TRUST_REMOTE_CODE=true` |
| **4bit 로 GPU 에 들어가는 크기** | QLoRA 는 4bit 베이스 위에서 학습합니다 | 학습 중 `CUDA out of memory` |
| 한국어를 잘함 | 민원 문장 이해가 판정 품질을 좌우합니다 | 학습해도 점수가 낮음 |

크기 대략 (4bit QLoRA 학습, `max_length=4096`, gradient checkpointing 켬 기준 — 모델 구조에 따라 다름):

| GPU | 무난한 크기 |
|---|---|
| T4 16GB | ~4B |
| L4 24GB | ~8B |
| A100 40GB | ~14B |

시험해 볼 만한 예 (이름은 Hugging Face 저장소 id. **최신 이름·라이선스는 모델 페이지에서 확인**하세요):

| 모델 | 비고 |
|---|---|
| `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B` | Qwen3.5 이전 세대. 사고 모드가 있어 `LLM_ENABLE_THINKING=false` 그대로 |
| `Qwen/Qwen2.5-7B-Instruct` | 사고 모드 없음 |
| `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct`, `…-7.8B-Instruct` | 한국어 특화. 3.5 판은 `LLM_TRUST_REMOTE_CODE=true` 필요. 라이선스(비상업) 확인 |
| `google/gemma-3-4b-it` | **승인제** — 아래 토큰 설정 필요 |
| `meta-llama/Llama-3.1-8B-Instruct` | **승인제**. 한국어 토큰 효율이 낮아 입력이 길어지고 느림 |

#### 바꾸는 순서

1. **(승인제 모델만) Hugging Face 토큰 준비**
   - 모델 페이지에서 라이선스 동의(Access 요청) → 승인 메일 확인
   - [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 에서 **Read** 토큰 생성
   - Colab 왼쪽 **🔑(보안 비밀)** → 새 보안 비밀 → 이름 `HF_TOKEN`, 값에 토큰 붙여넣기 → **노트북 액세스 켜기**
   - (토큰을 노트북 셀이나 `.env` 에 직접 적지 마세요 — 노트북을 공유하면 같이 퍼집니다)

2. **드라이브의 `backend/.env` 를 고칩니다.** (드라이브에서 우클릭 → 연결 앱 → 텍스트 편집기, 또는 Colab 왼쪽 📁 에서 더블클릭)

   ```bash
   LLM_MODEL=Qwen/Qwen2.5-7B-Instruct    # 바꿀 모델의 저장소 id
   LLM_TRUST_REMOTE_CODE=false           # EXAONE 3.5 처럼 자체 코드가 필요한 모델만 true
   LLM_ENABLE_THINKING=false             # 그대로 (사고 모드가 없는 모델은 무시됨)
   LLM_LOAD_4BIT=true                    # 그대로
   LLM_ADAPTER_PATH=                     # 반드시 비우기 - Qwen 으로 학습한 어댑터는 다른 모델에 못 얹습니다
   ```

   `.env` 가 아직 없으면 노트북 3-1 을 한 번 실행하면 만들어집니다. 그 뒤 고치세요.
   (3-1 셀 안의 `.env` 기본값 글은 **파일이 없을 때만** 쓰이므로 셀을 고칠 필요는 없습니다)

3. **런타임 → 세션 다시 시작 → 3-1 부터** 실행합니다. 3-1 출력의 `LLM_MODEL` 이 바꾼 이름인지 확인하세요.
   설정은 서버 코드를 불러올 때 한 번만 읽히므로, 재시작 없이 `.env` 만 고치면 반영되지 않습니다.

4. **4단계 `RUN_NAME` 에 모델 이름을 넣으세요.** 예: `v1_qwen25_7b`, `v1_exaone24`. 버전 비교 표에서 구분하기 쉽습니다. (`v` 로 시작)

5. 단계별로 **다르게 보이는 것**

   | 단계 | 확인할 것 |
   |---|---|
   | 6 | 바뀌지 않음 (④ 는 bge-m3 라 모델과 무관, 캐시 그대로 씀) |
   | 7 | `고정 프리픽스 길이` 가 토크나이저마다 다릅니다 (한국어 토큰 효율이 낮은 모델은 더 길고 느림). **`【학습 대상 ▶ … ◀】` 끝의 토큰이 그 모델의 발화 끝 토큰**인지 확인 — Qwen `<\|im_end\|>`, Llama 3 `<\|eot_id\|>`, Gemma `<end_of_turn>`, EXAONE `[\|endofturn\|]` |
   | 8 | 기준선 캐시 지문에 모델 이름이 들어 있어 **자동으로 새로 채점**합니다. 학습 전 점수 자체가 모델 비교의 첫 자료입니다 |
   | 8 | `KV 캐시` 의 `verified.same_choice` 가 False 거나 `disabled_reason` 이 있으면, 그 모델에서는 캐시가 자동으로 꺼진 것입니다. 결과는 맞고 느려질 뿐입니다 |
   | 9 | `붙은 층 수` 와 층 이름이 모델마다 다릅니다. `visual`·`vision` 이 들어간 이름이 없으면 정상 |
   | 10 | 메모리가 부족하면 더 작은 모델로 바꾸거나 `max_length` 를 줄이세요 |
   | 12 | `run_info.json` 의 `base_model` 에 모델 이름이 기록됩니다 |

6. **모델끼리 비교할 때**는 같은 `records.jsonl`, 같은 `EVAL_LIMIT`, 같은 `TRAIN_SAMPLE_LIMIT` 으로 돌린 뒤
   11단계 **학습 후** 지표(특히 의도·카테고리 정확도, keyword·period 일치, location 지어냄)를 나란히 보세요.
   학습 시간과 8·11단계 채점 시간(= 응답 속도의 대략적인 비교)도 함께 적어 두면 좋습니다.

7. **원래 Qwen 으로 돌아가려면** `.env` 를 `LLM_MODEL=Qwen/Qwen3.5-4B`, `LLM_TRUST_REMOTE_CODE=false` 로 되돌리고 세션 재시작.
   다른 모델로 학습한 어댑터를 서버에 쓰려면 서버 `.env` 의 `LLM_MODEL` 도 그 모델이어야 합니다.

#### 그래도 안 될 때 고칠 곳

| 증상 | 고칠 곳 |
|---|---|
| 모델 로드 오류 (`Unrecognized configuration class`, `trust_remote_code`) | `.env` 의 `LLM_TRUST_REMOTE_CODE=true` → 그래도 안 되면 `app/services/llm_engine.py` 의 `load_base_model()` / `_load_weights()` |
| 채팅 템플릿 오류 (system 역할 등) | `llm_engine.py` 의 `render_template()` — system 대신 user 앞에 프리픽스를 붙이는 식으로 바꾸면 되지만, **KV 캐시·학습 입력이 이 함수를 같이 쓰므로** 한 곳만 고치면 됩니다 |
| 생성이 끝나지 않고 계속 이어짐 / 7단계 끝 토큰이 이상함 | `llm_engine.py` 의 `_TURN_END_TOKENS` 에 그 모델의 발화 끝 토큰 이름을 추가 |
| LoRA 가 이상한 층(비전·오디오 등)에 붙음 | `training/trainer.py` 의 `_SKIP_PARTS` 에 그 층 이름 일부를 추가 |
| `KeyError: '<모델 종류>'` | transformers 가 그 모델을 모름. 2단계가 최신 transformers 를 설치하므로 2 → 3 → 3-1 을 다시 |

---

문서 인식 및 자동분류 통합시스템의 백엔드입니다.

**현재 구현 범위** : ① 입력 → ② 텍스트 추출 → ③ 임베딩 → ④ 후보 추림 → ⑤ Qwen 판정 → ⑥ 민원 DB 실행

```
① 입력 ──▶ ② 전처리(추출)          텍스트 / 이미지+텍스트 → 원문
              │
              ├─▶ 키워드 3개 + 요약문   kiwipiepy + KeyBERT + TextRank
              │
           ③ 임베딩 bge-m3          문장 → 1024차원 벡터 (학습 없음)
              │
           ④ 후보 추림               카테고리 7종과 코사인 유사도 → top-3
              │   └ 벡터DB             라벨링된 사례로 top-3 재정렬 (data/cases.csv, CASE_MODE)
              │
           ⑤ Qwen3.5-4B             의도 1개(+해당없음 게이트) / 후보 3개 중 카테고리 1개 / 도구 호출 JSON
              │
           ⑥ 민원 DB (SQLite)       도구 호출 실제 실행 - 규칙검사(소유권·상태) 통과 시 즉시 반영
              │
           (⑦ 은 render_result_text() 가 사람이 읽는 텍스트로 대신합니다)
```

⑤ 가 만든 도구 호출 JSON 은 ⑥(`tool_executor.py`)에서 실제 SQLite DB에 실행되고, 그 결과가
`result_text` 와 API 의 `tool_result` 에 담겨 나옵니다. (5.5 참고)

---

## 1. 설치

### 1-0. 어디에 무엇을 설치하나

설치 파일은 두 개로 나뉘어 있고, **같은 윈도우 가상환경에 둘 다 넣으면 안 됩니다.**

| 파일 | 내용 | 설치할 곳 |
|---|---|---|
| `requirements.txt` | 웹 서버 + ①② 문서 추출 (torch 없음) | 모든 환경 |
| `requirements-model.txt` | ③④⑤ 모델 스택 (torch / transformers / sentence-transformers …) | Colab T4 또는 GPU 리눅스 서버 |

| 구성 | 어디서 | 담당 | 설치 순서 |
|---|---|---|---|
| ①② 추출 | 윈도우 | 파일 → 텍스트 (PaddleOCR) | 1-1 → 1-2 → 1-3 |
| ③④⑤ 판정 | Colab T4 | 텍스트 → 의도·카테고리·도구 JSON | 1-5 (노트북이 알아서 설치) |
| ③④⑤ 판정 | GPU 리눅스 서버 | 위와 같음 | 1-1 → 1-2 → 1-4 |
| 벡터DB (pgvector) | PC / 운영 서버 | 라벨링된 사례 저장·검색 (PostgreSQL) | 1-6 |

**왜 나눴나** — PaddleOCR 과 PyTorch 를 같은 윈도우 가상환경에 넣으면 둘 다 Intel MKL / OpenMP DLL 을
각자 들고 와서 충돌하고, 서버가 기동조차 못 합니다.

```
OSError: [WinError 127] 지정된 프로시저를 찾을 수 없습니다.
Error loading "...\torch\lib\shm.dll" or one of its dependencies.
```

`paddleocr` 는 `paddlex → modelscope → torch` 순으로 torch 를 끌어다 쓰기 때문에, torch 가
*설치는 됐는데 DLL 로딩에 실패하는* 상태면 OCR 까지 같이 넘어갑니다. (torch 가 아예 없으면
modelscope 가 조용히 건너뛰므로 OCR 은 멀쩡합니다) 이미 설치했다면 1-8 을 보세요.

### 1-1. 공통 — 가상환경 + 기본 패키지

Python **3.11 / 3.12** 를 권장합니다.

```bash
cd backend

python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

이것만으로 서버는 뜨고 docx·xlsx·pptx·hwp·hwpx·txt·텍스트 PDF 추출은 동작합니다.
이미지와 스캔 PDF 는 1-3 의 PP-OCRv5 가 있어야 합니다.

### 1-2. 환경변수 (.env)

```bash
copy .env.example .env          # Windows
cp .env.example .env            # macOS / Linux
```

모든 항목에 기본값이 있어 `.env` 없이도 동작합니다. 각 값의 의미는 `.env.example` 의 주석을 보세요.
`.env` 는 Git 에 커밋하지 않습니다. (`.gitignore` 에 등록되어 있습니다)

> 함께 들어 있는 `.env` 는 **Colab 용 값**입니다. (`MODEL_CACHE_DIR` 은 드라이브 용량을 아끼도록
> 비워 둬 세션 로컬 디스크를 씁니다 - 필요하면 드라이브 경로로 바꾸세요)
> 윈도우나 서버에서 쓸 때는 위 명령으로 `.env.example` 을 복사해 덮어쓰세요.

### 1-3. PP-OCRv5 설치 (윈도우 · 추출 담당)

PaddlePaddle 은 PyPI 가 아니라 Paddle 전용 인덱스에서 배포되므로 `pip install -r requirements.txt`
로는 설치되지 않습니다. 아래 순서대로 따로 설치합니다.

**설치하지 않아도 서버는 정상 실행되며** 이렇게 동작합니다.

| 입력 | 미설치 시 |
|---|---|
| 이미지 파일 | 503 `OCR_UNAVAILABLE` |
| 스캔 PDF (텍스트 없는 페이지) | 해당 페이지를 건너뛰고 경고로 기록 |
| 그 외 문서 | 영향 없음 |

**1단계) PaddlePaddle — 환경에 맞는 것 하나만 실행**

```bash
# CPU 만 사용 (가장 무난, GPU 없어도 동작)
python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# NVIDIA GPU + CUDA 11.8
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# NVIDIA GPU + CUDA 12.6
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

- `-i` 옵션은 그 명령에서만 인덱스를 바꾸므로 다른 패키지에는 영향이 없습니다.
- 버전/CUDA 조합은 공식 안내를 따르세요: https://www.paddlepaddle.org.cn/en/install/quick

**2단계) PaddleOCR (일반 PyPI)**

```bash
python -m pip install paddleocr
```

반드시 1단계를 먼저 하세요. paddlepaddle 없이 paddleocr 만 설치하면 설치는 되지만 실행 시 ImportError 가 납니다.

**3단계) 설치 확인**

```bash
python -c "import paddle; paddle.utils.run_check()"
python -c "from paddleocr import PaddleOCR; print('paddleocr ok')"
```

**4단계) 첫 실행 시 모델 다운로드**

첫 OCR 요청 때 모델 가중치를 자동으로 내려받습니다. (`~/.paddlex` 에 저장)
인터넷 연결이 필요하고 수 분 걸릴 수 있으며, 이후 요청은 로드된 모델을 재사용합니다.
사용할 모델은 `.env` 의 `OCR_DET_MODEL` / `OCR_REC_MODEL` 로 바꿉니다.
(기본값: `PP-OCRv5_server_det` / `korean_PP-OCRv5_mobile_rec`)

설치 여부는 서버 기동 후 `GET /health` 의 `ocr_available` 로 확인합니다. **설치 여부는 기동할 때
한 번만 확인하므로, 설치한 뒤에는 서버를 재시작하세요.**

**(선택) HEIC / HEIF 이미지를 다룬다면**

```bash
python -m pip install pillow-heif
```

### 1-4. 모델 스택 설치 (GPU 리눅스 서버 · 판정 담당)

Colab 이라면 이 절은 건너뛰고 1-5 로 가세요. 노트북이 같은 일을 대신합니다.

**1단계) 기본 → 모델 스택 순서로 설치**

```bash
python -m pip install -r requirements.txt          # 1-1 에서 했으면 생략
python -m pip install -r requirements-model.txt
```

**2단계) GPU 용 PyTorch** — `requirements-model.txt` 의 `torch>=2.2` 는 PyPI 기본 휠(CPU 판)입니다.
GPU 를 쓸 거라면 CUDA 판으로 다시 설치하세요.

```bash
# CUDA 12.1
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.4
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124

# 확인
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**3단계) 4bit 양자화 (bitsandbytes)** — CUDA 전용이라 requirements 에 넣지 않았습니다.
없어도 서버는 동작하며, 그때는 4bit 를 끄고 올립니다. (CPU 면 float32 - 느리지만 동작)

```bash
python -m pip install "bitsandbytes>=0.43.0"
python -c "import bitsandbytes; print(bitsandbytes.__version__)"
```

설치돼 있고 CUDA 가 보이면 `.env` 의 `LLM_LOAD_4BIT=true` 일 때 4bit NF4(QLoRA 와 같은 설정)로 로드합니다.
`GET /api/v1/analyze/status` 의 `load_4bit` 값으로 확인합니다.

**4단계) transformers 최신판 — Qwen3.5 를 못 알아볼 때만**

Qwen3.5 는 transformers 에 아키텍처가 들어간 지 얼마 되지 않아, PyPI 정식판이 모르는 경우가 있습니다.
아래 오류가 나면 git 판으로 올리세요.

```
KeyError: 'qwen3_5'
ValueError: The checkpoint you are trying to load has model type `qwen3_5`
            but Transformers does not recognize this architecture
```

```bash
python -m pip install --upgrade "git+https://github.com/huggingface/transformers.git"

# 확인
python -c "import transformers; print(transformers.__version__)"
python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('Qwen/Qwen3.5-4B').model_type)"
```

### 1-5. Colab T4 에서 돌리기

**`colab_backend.ipynb`** 를 Colab 에서 [파일] > [노트북 업로드] 로 올리고 셀을 위에서부터 누르면 됩니다.
Qwen3.5-4B 를 4bit 로 올리면 VRAM 약 2.5GB, bge-m3 약 1.2GB 로 T4(16GB)에 넉넉히 들어갑니다.

| 셀 | 하는 일 |
|---|---|
| 0 | `nvidia-smi` 로 T4 확인 (먼저 [런타임] > [런타임 유형 변경] > T4 GPU) |
| 1 | 구글 드라이브 마운트 후 `MyDrive/backend` 로 이동 (코드는 드라이브에 미리 올려 둠) |
| 2 | 설치 (Colab 의 CUDA torch 를 덮어쓰지 않도록 `torch` 줄 제외) |
| 3 | **런타임 재시작** — 새 transformers 적용 |
| 4 | 경로 재설정 + 환경 확인 (`bf16 지원: False` 가 T4 정상) |
| 5 | `.env` 쓰기 |
| 6-A | 터널 없이 `pipeline.run()` 직접 호출 — 가장 빠른 확인 |
| 6-B | cloudflared 터널로 Swagger 열기 |
| 7 | (선택) 벡터DB 파일(`cases.npy`, `cases.meta.json`) 내려받기 — PC 의 pgvector 적재용 (1-6) |

2번 셀이 실제로 실행하는 설치 명령은 이렇습니다.

```bash
!pip install -q -r requirements.txt                       # 웹·추출 (torch 없음)
!grep -v "^torch" requirements-model.txt > /tmp/req_model.txt
!pip install -q -r /tmp/req_model.txt                     # 모델 스택 (Colab 의 CUDA torch 유지)
!pip install -q bitsandbytes                              # 4bit 양자화 (T4 에서 사실상 필수)
!pip install -q --upgrade "git+https://github.com/huggingface/transformers.git"
```

**T4 권장 `.env`** — 핵심은 앞의 세 줄입니다.

```bash
LLM_LOAD_4BIT=true             # 끄면 fp16 으로 약 8GB - bge-m3 와 같이 올리면 위태롭습니다
LLM_4BIT_COMPUTE_DTYPE=auto    # T4 는 bfloat16 이 없어 자동으로 float16. bfloat16 강제 시 느려집니다
LLM_ENABLE_THINKING=false      # Qwen3.5 는 기본이 사고 모드 - 켜두면 의도/카테고리 판정이 깨집니다
DEVICE=auto
DEBUG=true
PRELOAD_MODELS=true            # 기동이 느린 대신 첫 요청이 빨라집니다 (노트북 기본값은 false)

# MODEL_CACHE_DIR : 비우면 세션 로컬 디스크(~/.cache/huggingface) - 드라이브 용량 걱정 없음,
#                    런타임 끊기면 약 10GB 를 다시 받습니다. (노트북 기본값)
MODEL_CACHE_DIR=
# 드라이브 용량에 여유가 있다면 이렇게 바꾸면 런타임이 끊겨도 다시 받지 않습니다.
# MODEL_CACHE_DIR=/content/drive/MyDrive/hf_cache
```

**Swagger 를 여는 방법** — Colab 은 외부에서 바로 접속할 수 없어 터널이 필요합니다.

- **(A) cloudflared** — 가입·토큰이 필요 없어 가장 간단합니다. 노트북 6-B 셀이 이 방식입니다.
  출력된 `https://....trycloudflare.com` 뒤에 `/docs` 를 붙이면 Swagger 가 열립니다.
- **(B) ngrok** — 무료 토큰이 필요합니다. (https://dashboard.ngrok.com)

  ```python
  !pip install -q pyngrok nest_asyncio
  from pyngrok import ngrok
  ngrok.set_auth_token("여기에_본인_토큰")
  print(ngrok.connect(8000).public_url + "/docs")
  # 그 다음 6-B 셀의 uvicorn 스레드 코드를 그대로 실행
  ```

- **(C) 터널 없이** — Swagger 없이 함수만 호출합니다. 노트북 6-A 셀과 같습니다.

  ```python
  from app.services import pipeline
  r = pipeline.run("우리 동네 가로등이 꺼졌어요")
  print(pipeline.render_result_text(r))
  print(pipeline.render_debug_text(r))
  ```

Colab 에는 PaddleOCR 을 설치하지 않으므로 이미지·스캔 PDF 업로드는 503 이 정상입니다.
`/analyze/text` 로 문장을 넣어 테스트하거나, 윈도우 쪽에서 `/extract/text` 로 텍스트를 먼저 뽑아 넘기세요.

### 1-6. PostgreSQL + pgvector 준비 (벡터DB)

벡터DB 는 `.env` 의 `CASE_STORE_BACKEND` 로 저장 방식을 고릅니다. **검색 결과는 둘이 같습니다.**

| 값 | 어디에 저장 | 쓰는 곳 |
|---|---|---|
| `numpy` (기본) | `data/cases.npy` (메모리에서 계산) | Colab 테스트 — 추가 설치 없음 |
| `pgvector` | PostgreSQL 테이블 `complaint_cases` | PC · 운영 서버 |

> **알아둘 점** — 요청을 처리하면서 벡터DB 를 검색하는 곳은 ③④⑤ 가 도는 곳입니다.
> 지금 구성에서 PC(윈도우)는 모델 스택이 없어 ③④⑤ 를 돌리지 않으므로, PC 에서는
> **사례 적재와 검색 동작 확인**(아래 6단계)까지 할 수 있습니다. 실제 요청 처리에서 pgvector 를 쓰는 것은
> 모델과 DB 에 모두 닿는 곳(운영 GPU 서버 등)에서 `CASE_STORE_BACKEND=pgvector` 로 띄울 때입니다.
> Colab 은 PC 의 DB 에 접속할 수 없으므로 `numpy` 로 둡니다.

PostgreSQL 17 을 설치만 한 상태라면 아래를 순서대로 하면 됩니다. (윈도우 기준, 한 번만)

**1단계) pgvector 가 이미 있는지 확인**

시작 메뉴 → **SQL Shell (psql)** 을 열고 엔터를 몇 번 눌러 기본값으로 접속한 뒤
(마지막에 설치할 때 정한 postgres 비밀번호 입력) 아래를 실행합니다.

```sql
SELECT name, default_version FROM pg_available_extensions WHERE name = 'vector';
```

한 줄이 나오면 3단계로, **0 rows** 면 2단계로 가세요. (윈도우용 PostgreSQL 설치 파일에는 기본으로 들어 있지 않습니다)

**2단계) pgvector 설치**

공식 pgvector 는 Windows 용 컴파일된 파일을 배포하지 않아 직접 빌드해야 하지만,
**컴파일러 없이 파일 3개만 복사하는 방법(A)** 이 훨씬 쉽습니다. 안 되면 (B) 직접 빌드로 가세요.

**A. 미리 컴파일된 파일 복사 (권장 — Visual Studio 불필요)**

[andreiramani/pgvector_pgsql_windows](https://github.com/andreiramani/pgvector_pgsql_windows) 저장소가
PostgreSQL 17 용으로 미리 빌드해 둔 파일을 배포합니다. **공식 배포가 아닌 개인 빌드**이므로,
내용이 꺼려지면 아래 B(직접 빌드)로 진행하세요.

1. [Releases 페이지](https://github.com/andreiramani/pgvector_pgsql_windows/releases)에서
   PostgreSQL **17** 대상 최신 버전(예: `pgvector-v0.8.6-pg17-x64-windows` 형태의 이름)을 내려받아 압축을 풉니다.
2. 압축 안의 파일을 PostgreSQL 설치 폴더로 복사합니다. (관리자 권한 필요할 수 있음)
   - `vector.dll` → `C:\Program Files\PostgreSQL\17\lib\`
   - `vector.control`, `vector--*.sql` → `C:\Program Files\PostgreSQL\17\share\extension\`
   (압축 안에 `lib\`, `share\extension\` 폴더 구조가 그대로 들어 있으면 `C:\Program Files\PostgreSQL\17\` 에
   덮어쓰기만 하면 됩니다. 정확한 구조는 압축 안의 `readme.txt` 를 확인하세요)
3. PostgreSQL 서비스를 재시작합니다. (서비스 관리자에서 `postgresql-x64-17` 재시작, 또는 재부팅)
4. 1단계 SQL 을 다시 실행해 한 줄이 나오는지 확인합니다.

버전 폴더(`17`)는 설치한 PostgreSQL 메이저 버전과 같으면 됩니다 — 17.11 처럼 마이너 버전이 달라도
같은 17 계열이면 그대로 동작합니다(PostgreSQL 확장은 메이저 버전 단위로 호환됩니다).

**B. 직접 빌드 (A 가 안 될 때만)**

1. **Visual Studio 2022 Build Tools** 를 설치하고, 설치 화면에서 **"C++를 사용한 데스크톱 개발"** 을 체크합니다.
   https://visualstudio.microsoft.com/ko/visual-cpp-build-tools/
2. **Git for Windows** 가 없으면 설치합니다. https://git-scm.com/download/win
3. 시작 메뉴에서 **x64 Native Tools Command Prompt for VS 2022** 를 찾아 **관리자 권한으로 실행**합니다.
   (일반 cmd·PowerShell 이 아니라 반드시 이 창이어야 nmake 가 동작합니다)
4. 아래를 차례로 실행합니다. `17` 은 설치한 PostgreSQL 버전 폴더입니다.

```bat
set "PGROOT=C:\Program Files\PostgreSQL\17"
cd %TEMP%
git clone --branch v0.8.6 https://github.com/pgvector/pgvector.git
cd pgvector
nmake /F Makefile.win
nmake /F Makefile.win install
```

5. 1단계 SQL 을 다시 실행해 한 줄이 나오는지 확인합니다.

**3단계) 데이터베이스 만들기 + 확장 켜기**

`backend` 폴더에서 아래를 실행합니다. (postgres 비밀번호를 물어봅니다)

```bat
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -f scripts\setup_pgvector.sql
```

`minwon` 데이터베이스를 만들고 `CREATE EXTENSION vector` 를 실행한 뒤, 마지막에 `extversion` 과
거리 계산 결과를 보여 줍니다. pgAdmin 을 쓴다면 같은 SQL 을 Query Tool 에 붙여 넣어도 됩니다.
**테이블은 만들 필요가 없습니다.** 서버나 6단계 스크립트가 자동으로 만듭니다.

**4단계) `.env` 설정**

```bash
CASE_STORE_BACKEND=pgvector
CASE_PG_DSN=postgresql://postgres:비밀번호@localhost:5432/minwon
CASE_PG_TABLE=complaint_cases
```

- 비밀번호에 `@ : / # %` 같은 기호가 있으면 URL 인코딩해서 적으세요. (예: `@` → `%40`)
- 드라이버(`psycopg`)는 `requirements.txt` 에 들어 있습니다. 1-1 을 다시 실행하면 설치됩니다.

**5단계) Colab 에서 만든 벡터 가져오기**

PC 에는 bge-m3 가 없어 사례를 직접 임베딩할 수 없습니다. 그래서 Colab 이 만든 벡터 파일을 가져옵니다.

1. Colab 에서 서버나 `pipeline.run()` 을 한 번 실행하면 `data/cases.npy` 와 `data/cases.meta.json` 이 생깁니다.
   (`MyDrive/backend/data/` 에 그대로 남습니다. 노트북 7번 셀로 내려받을 수도 있습니다)
2. 두 파일을 PC 의 `backend\data\` 에 복사합니다.
3. **`data/cases.csv` 는 Colab 과 똑같은 파일이어야 합니다.** CSV 를 고쳤다면 Colab 에 올려 다시 만든 뒤 가져오세요.
   (내용이 다르면 적재하지 않고 이유를 알려 줍니다)

**6단계) 적재 + 동작 확인**

```bash
python scripts/sync_cases.py
```

테이블을 만들어 사례를 넣고(HNSW 인덱스 포함), 저장된 사례 3개를 **자기 벡터로 검색**해 봅니다.
1위가 자기 자신(유사도 `1.0000`)이고 마지막 줄이 `결과 : 정상` 이면 끝입니다.
(질의문을 새로 임베딩하지 않으므로 PC 에서도 확인할 수 있습니다)

서버를 띄워도 기동할 때 같은 적재를 자동으로 합니다. 이미 같은 내용이 들어 있으면 아무것도 하지 않습니다.
pgAdmin 에서 보려면:

```sql
SELECT category, count(*) FROM complaint_cases GROUP BY category ORDER BY 1;
```

### 1-7. 모델 가중치 미리 받기 / 폐쇄망

아래 두 모델은 pip 가 아니라 **첫 분류 요청 때** HuggingFace 에서 자동으로 내려받습니다.

| 모델 | 단계 | 크기 |
|---|---|---|
| `BAAI/bge-m3` | ③ 임베딩 | 약 2.2GB |
| `Qwen/Qwen3.5-4B` | ⑤ 판정 | 약 8GB (4bit 로 올리면 VRAM 약 2.5GB) |

미리 받아두려면 (인터넷 되는 곳에서 한 번만)

```bash
python -m pip install huggingface_hub
huggingface-cli download BAAI/bge-m3
huggingface-cli download Qwen/Qwen3.5-4B
```

- 저장 위치를 바꾸려면 `.env` 의 `MODEL_CACHE_DIR` 을 지정하세요.
- 폐쇄망이라면 받은 두 폴더를 통째로 옮긴 뒤 `.env` 의 `EMBED_MODEL` / `LLM_MODEL` 에 로컬 경로를 적고,
  환경변수 `HF_HUB_OFFLINE=1` 로 실행하세요.
- 기동할 때 미리 올리려면 `.env` 의 `PRELOAD_MODELS=true` 로 두세요.

### 1-8. 윈도우에 모델 스택을 설치해 버렸다면

지우면 원래대로 돌아옵니다.

```bash
python -m pip uninstall -y torch transformers sentence-transformers accelerate peft keybert
```

서버를 재시작한 뒤 `GET /health` 의 `ocr_available` 이 `true` 로 돌아오는지 확인하세요.
`false` 라면 같은 응답의 `ocr_error` 에 정확한 사유가 찍힙니다.

### 1-9. 설치 확인

서버를 띄운 뒤(2. 실행) 아래 두 주소로 확인합니다.

| 주소 | 확인할 값 |
|---|---|
| `GET /health` | `ocr_available`(PP-OCRv5), `device`, `embed_loaded`, `llm_loaded` |
| `GET /api/v1/analyze/status` | `embedder.installed`, `llm.installed`, `llm.kv_cache`(KV 캐시), `case_store.backend`·`ready`·`cases`(벡터DB), `complaint_store.ready`·`complaints`·`by_status`(민원 DB), `runtime`(GPU 이름·VRAM·`supports_bf16`·`compute_dtype`), `load_4bit` |

T4 라면 `runtime` 에 `load_4bit: true`, `supports_bf16: false`, `compute_dtype: float16` 이 보여야 합니다.

### 1-10. 자주 겪는 문제

| 증상 | 원인과 해결 |
|---|---|
| Windows 에서 `DLL load failed` | Microsoft Visual C++ 재배포 패키지를 설치하세요. https://aka.ms/vs/17/release/vc_redist.x64.exe |
| Windows 에서 `[WinError 127] shm.dll` | torch 와 paddle 의 DLL 충돌입니다. 1-8 로 모델 스택을 지우세요. |
| `ImportError: libGL.so.1` (Linux) | opencv 의존성입니다. `sudo apt-get install -y libgl1 libglib2.0-0` |
| 설치했는데 `ocr_available` 이 false | 가상환경(.venv)을 켠 상태에서 설치했는지, 설치 후 서버를 재시작했는지 확인하세요. |
| kiwipiepy 설치 실패 (Windows) | Microsoft Visual C++ 재배포 패키지를 설치하세요. |
| 첫 분류 요청이 5~15분 / 타임아웃 | 정상입니다. 가중치 약 10GB 를 받는 중입니다. 로그를 확인하고, `MODEL_CACHE_DIR` 을 영구 경로로 두세요. |
| `KeyError: 'qwen3_5'` | transformers 가 낡았습니다. 1-4 의 4단계로 git 판을 설치하세요. (Colab 은 설치 후 **런타임 재시작** 필수) |
| `Unrecognized processing class in BAAI/bge-m3` | `sentence-transformers` 5.4 이상이 설치된 것입니다. 5.4 부터 텍스트 전용 모델에도 AutoProcessor 로딩을 시도해 나는 오류입니다. `pip install "sentence-transformers<5.4"` 로 다시 설치하고 (Colab 은) **런타임 재시작**하세요. `requirements-model.txt` 는 이미 `<5.4` 로 고정돼 있으니, 이전에 더 새 버전을 설치해 둔 세션에서만 겪습니다. |
| `CUDA out of memory` | `LLM_LOAD_4BIT=true` 인지 확인하세요. 그래도 나면 런타임/프로세스를 재시작하거나 `DEVICE=cpu` 로 바꾸세요. |
| CPU 에서 한 요청에 수 분 | 4B 모델은 CPU 가 너무 느립니다. Colab T4 등 GPU 를 쓰세요. |
| bf16 관련 경고/오류 | `LLM_4BIT_COMPUTE_DTYPE=auto` 로 두세요. T4 는 float16 이 맞습니다. |
| 응답에 `<think>` 가 섞임 | `LLM_ENABLE_THINKING=false` 인지 확인하세요. |
| Colab 이미지 업로드가 503 | Colab 에는 PaddleOCR 을 설치하지 않았기 때문입니다. 정상입니다. |
| Colab 90분 방치 후 끊김 | 무료 티어 제한입니다. 노트북 4번 셀부터 다시 실행하세요. |
| 기동 로그에 "벡터DB 를 사용할 수 없습니다" | `numpy` 모드의 윈도우(추출 전용)에서는 정상입니다. `pgvector` 모드라면 1-6 의 5단계(Colab 벡터 파일 복사)를 확인하세요. 그 외에는 `status` 의 `case_store.error` 를 보세요. |
| `pgvector 확장이 설치되어 있지 않습니다` | 1-6 의 2단계로 pgvector 를 설치하세요. (A: 파일 복사 / B: 직접 빌드) |
| `nmake` 를 찾을 수 없음 | 1-6 2단계 B 방식(직접 빌드)을 쓰는 경우, 일반 cmd 가 아니라 **x64 Native Tools Command Prompt for VS 2022** 에서 실행하세요. A 방식(파일 복사)은 `nmake` 가 필요 없습니다. |
| `PostgreSQL 에 접속하지 못했습니다` | PostgreSQL 서비스가 켜져 있는지, `CASE_PG_DSN` 의 비밀번호·포트·DB 이름이 맞는지 확인하세요. |
| 로그에 "KV 캐시를 끄고 전체 입력 계산으로 전환" | KV 캐시 검증이나 사용 중 문제가 있어 스스로 꺼진 것입니다. 판정은 기존 방식으로 정상 동작합니다. 사유는 `status` 의 `llm.kv_cache.disabled_reason` 에 있습니다. |

---

## 2. 실행

```bash
uvicorn app.main:app --reload
```

| 주소 | 설명 |
|---|---|
| http://127.0.0.1:8000/docs | Swagger UI (여기서 바로 테스트) |
| http://127.0.0.1:8000/redoc | ReDoc |
| http://127.0.0.1:8000/health | 서버 상태 + OCR 설치 여부 + 모델 로드 여부 |

## 3. 확장자별 처리 방식

### PDF — 페이지마다 다르게 처리합니다

업로드된 PDF 를 페이지 단위로 돌면서 텍스트 레이어가 있는지 확인합니다.
(기준은 `.env` 의 `PDF_MIN_TEXT_CHARS`, 기본 10자)

| 페이지 상태 | 처리 |
|---|---|
| **텍스트 없음** (스캔본) | 페이지 전체를 이미지로 렌더링해 **PP-OCRv5** 로 보냅니다 |
| **텍스트 있음** | 텍스트는 **PyMuPDF** 가 추출하고, 페이지에 박힌 이미지는 PyMuPDF 가 **정확한 좌표(bbox)로 잘라내** 그 부분만 PP-OCRv5 로 보냅니다 |

- 잘라낸 이미지의 OCR 결과는 `[이미지 1 OCR | 좌표 (50,60)-(400,300)]` 형태로 본문 뒤에 붙습니다.
- 가로·세로가 `OCR_MIN_IMAGE_PX`(기본 80) 보다 작은 이미지는 아이콘·구분선으로 보고 건너뜁니다.
  (PDF 좌표 단위인 pt 기준입니다. 1pt = 1/72 인치)
- `PDF_OCR_EMBEDDED_IMAGES=false` 로 두면 이미지 OCR 없이 텍스트만 빠르게 뽑습니다.
- 실제로 OCR 이 몇 번 돌았는지는 서버 로그의 `ocr(page=…, image=…)` 로 확인합니다.

### 이미지 — 전부 PP-OCRv5

| 확장자 | 처리 |
|---|---|
| `.png` `.jpg` `.jpeg` `.webp` `.gif` `.bmp` `.tif` `.tiff` `.heic` `.heif` | 파일 전체를 **PP-OCRv5** 로 보냅니다 |

- 인식 결과는 좌표를 기준으로 위→아래, 왼쪽→오른쪽 순서로 정렬해 돌려줍니다.
- 신뢰도가 `OCR_MIN_SCORE`(기본 0.5) 미만인 결과는 버립니다.
- PP-OCRv5 미설치 시 **503 `OCR_UNAVAILABLE`** 을 반환합니다.
- `.heic` / `.heif` 는 `pillow-heif` 를 추가로 설치해야 디코딩됩니다.

### 문서 — 전용 라이브러리로 직접 파싱

| 확장자 | 라이브러리 | 추출 내용 |
|---|---|---|
| `.docx` | **docx2txt** | 문단과 표의 텍스트. 표는 행/열 구조 없이 펼쳐집니다 |
| `.xlsx` `.xlsm` | openpyxl | 시트별로 분리. 셀은 탭 구분, 수식은 계산된 값 |
| `.pptx` | python-pptx | 슬라이드별로 분리. 도형 텍스트 + 표 + 발표자 노트 |
| `.hwp` | olefile | HWP 5.0 본문 레코드 직접 파싱. 압축 해제, 제어문자 제거, 암호 파일 감지 |
| `.hwpx` | zipfile + xml | `Contents/sectionN.xml` 을 파싱해 문단 단위로 추출 |
| `.txt` `.md` `.csv` `.tsv` `.json` `.xml` `.html` `.htm` `.log` | **charset-normalizer** | 인코딩 자동 감지 후 디코딩 |
| `.rtf` | **striprtf** | 제어어를 제거하고 본문만 추출 |

**인코딩 판별 순서** — UTF-8 → CP949/EUC-KR(한글이 실제로 나오는지 확인) → charset-normalizer.
charset-normalizer 가 짧은 한글 바이트열을 big5 등으로 잘못 추정하는 경우가 있어
한국어 인코딩을 먼저 확인합니다.

### 거부

| 확장자 | 동작 |
|---|---|
| `.doc` `.xls` `.ppt` | 415 반환 + 최신 포맷으로 변환 안내 |
| 그 외 / 확장자 없음 | 415 반환 |

## 4. API

| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/api/v1/extract/text` | 파일 1개에서 텍스트 추출 (①②) |
| GET | `/api/v1/formats` | 확장자별 처리 방식 + OCR 설치 여부 |
| POST | `/api/v1/analyze/text` | 문장만 넣어 ②~⑤ 확인 (파일 없음) |
| POST | `/api/v1/analyze/file` | 파일 업로드 → ② 추출 → ③④⑤ |
| GET | `/api/v1/analyze/categories` | 카테고리 7종 정의 |
| GET | `/api/v1/analyze/status` | 모델 설치·로드 상태 |
| GET | `/health` | 서버 상태 + OCR 설치 여부 + 사용 중인 모델 |

`/analyze/file` 의 선택 입력칸 `text` 에 문장을 적으면 그림 ① 의 **이미지+텍스트** 입력이 됩니다.
적은 문장이 문서에서 뽑은 텍스트 앞에 붙어 함께 분류됩니다.

### `/extract/text` 성공 응답 예시 (스캔 페이지가 섞인 PDF)

청킹/임베딩에 바로 넘길 수 있도록 본문과 최소 식별 정보만 담습니다.
처리 방식·소요 시간·OCR 횟수·경고는 응답이 아니라 서버 로그(`logs/app.log`)에 남습니다.

```json
{
  "file": {
    "filename": "스마트팜보고서.pdf",
    "extension": ".pdf",
    "category": "pdf"
  },
  "pages": [
    { "page": 1, "label": "page:1", "text": "1. 개요\n...\n\n[이미지 1 OCR | 좌표 (50,60)-(400,300)]\n표 안의 글자" },
    { "page": 2, "label": "page:2 (ocr)", "text": "스캔된 페이지에서 인식한 글자" },
    { "page": 3, "label": "page:3", "text": "3. 결론\n..." }
  ]
}
```

## 5. 분류 파이프라인 (③ ④ ⑤ ⑥)

### 5.1 단계별로 무엇을 쓰는가

| 단계 | 하는 일 | 사용 | 학습 |
|---|---|---|---|
| ②→③ | 원문 → 키워드 3개 + 요약문 | kiwipiepy(형태소) + KeyBERT(bge-m3 재사용) + TextRank | 없음 |
| ③ | 질의문 → 1024차원 벡터 | `BAAI/bge-m3` (sentence-transformers) | 없음 |
| ④ | 카테고리 7종과 코사인 유사도 → top-3 | numpy | 없음 |
| ④ 보조 | 라벨링된 사례로 top-3 재정렬 (`CASE_MODE`) | 벡터DB (numpy `.npy`) + bge-m3 | 없음 |
| ⑤ | 의도 / 카테고리 / 도구 호출 JSON | `Qwen/Qwen3.5-4B` (+QLoRA 어댑터) | **유일한 파인튜닝 대상** |

원문을 그대로 임베딩하면 문서가 길수록 주제가 희석되므로, ③ 앞에서 키워드 3개와 요약문으로
압축한 짧은 질의문을 만들어 넘깁니다. 그 질의문은 `debug.embed_query_text` 에서 볼 수 있습니다.

KeyBERT 와 TextRank 는 ③ 의 bge-m3 인스턴스를 그대로 공유하므로 **추가로 받는 모델이 없습니다.**
kiwipiepy / keybert 가 없으면 정규식·빈도 기반 폴백으로 내려가며 `debug.warnings` 에 사유가 남습니다.

### 5.2 ⑤ 의 판정 방식

의도 판정과 카테고리 확정은 자유 생성이 아니라 **번호 토큰 1회 계산**입니다.

```
... 어시스턴트 발화가 "의도: " 까지 쓰인 상태로 forward 1회
    → 마지막 위치 로짓에서 "1"~"6" 토큰만 골라 softmax
    → 가장 높은 번호 = 판정, 그 확률 = 확신 점수
```

덕분에 (1) 형식이 깨질 수 없고 (2) 토큰 하나만 계산해 빠르며 (3) 확신 점수를 그대로 얻습니다.
카테고리도 같은 방식으로 ④ 가 추린 **후보 3개 중에서만** 고릅니다. 의도에 따라 선택지가 다릅니다.

| 의도 | 카테고리 판정 |
|---|---|
| 접수 | 후보 3개 중 하나 (없음 불가) |
| 조회 · 수정 · 삭제 | 후보 3개 + **`4. 없음`** — "어제 넣은 거 취소해 줘"처럼 주제가 안 드러나면 없음 |
| 문의 · 해당없음 | 판정하지 않음 (카테고리를 쓰지 않음) |

후보는 프롬프트에 `1. 국토교통` 처럼 번호와 이름만 적습니다. 각 카테고리의 설명은 이미 고정 프리픽스의
[카테고리 정의] 에 있으므로 반복하지 않습니다.

도구 호출 JSON 만 일반 생성이며, 의도가 `문의` 면 도구를 부르지 않고 안내 지식으로 즉답합니다.
도구의 `category` 인자는 번호 토큰으로 확정한 카테고리로 항상 맞춰집니다. (없음이면 `""`)

- 의도 5종(+해당없음) : `문의` `접수` `조회` `수정` `삭제` (+ `해당없음`) — 한 요청에 하나만 판정됩니다.
- 카테고리 7종 : `행정안전` `국토교통` `주택건축` `환경·위생` `보건복지` `소방` `기타`

카테고리를 바꾸려면 `app/services/categories.py` 한 곳만 고치면 됩니다. (도구 정의·프롬프트가 자동으로 따라갑니다)
정의문 임베딩은 프로세스당 한 번 계산해 캐시하므로, 고친 뒤에는 서버/커널을 재시작하세요.

### 5.3 KV 캐시 (고정 프리픽스 재사용)

매 요청마다 똑같이 들어가는 system 메시지이며 `prompts.build_fixed_prefix()` 한 곳에서 만듭니다.
⑤ 는 이 구간을 **한 번만 모델에 통과시켜(prefill) `past_key_values` 로 저장**해 두고,
요청마다 그 복사본에 요청 구간(민원 문장 + 단계별 지시)만 이어 붙여 계산합니다.
한 요청에서 ⑤ 가 모델을 2~3번 부르므로(의도 → 카테고리 → 도구 JSON) 매번 프리픽스를 다시 계산하지 않는 만큼 빨라집니다.

```
최초 1회   [고정 프리픽스 ~수천 토큰] ──prefill──▶ KV 캐시 (원본, 계속 보관)
요청마다   KV 캐시 복사본 + [요청 구간 수십~수백 토큰] ──▶ 번호 토큰 / 도구 JSON
```

| 순서 | 내용 | 코드 |
|---|---|---|
| 1 | 역할 지시문 | `prompts.ROLE_INSTRUCTION` |
| 2 | 도구 목록 (접수·조회·수정·취소 4종의 이름·용도·인자 이름) | `prompts.tool_block()` |
| 3 | 카테고리 정의 7종 (이름·담당 부서·한 줄 설명) | `categories.prompt_block()` |
| 4 | 의도 정의 5종 + 해당없음 | `prompts.intent_block()` |

- **프리픽스에는 모델의 판단에 필요한 내용만 둡니다.** INSERT/UPDATE, 소유권·상태 검사처럼 서버 코드가
  처리하는 동작 설명은 넣지 않습니다. 도구는 이름·용도·인자 이름만 한 줄씩 적고, 인자별 설명이 담긴
  전체 스키마(`TOOL_DEFINITIONS`)는 도구 호출 JSON 단계 프롬프트에 그 도구 하나만 넣습니다. (중복 제거)
  프리픽스가 짧을수록 서버 prefill 과 QLoRA 학습(샘플마다 프리픽스를 다시 계산)이 빨라집니다.
- **안내 지식(`GUIDE_KNOWLEDGE`, 자주 묻는 질문 QnA)은 프리픽스에 없습니다.** 의도·카테고리·도구 판정에는
  쓰이지 않으므로 '문의' 즉답 프롬프트(`build_answer_prompt`)에만 넣습니다. 문의 요청만 이 부분을 매번 계산합니다.
- 프리픽스는 요청마다 토큰이 한 글자도 달라지면 안 되므로 날짜·사용자 정보 같은 동적인 값은 넣지 마세요.
- **prefill 시점** — `PRELOAD_MODELS=true` 면 기동할 때, 아니면 첫 분류 요청 때 한 번 합니다.
  프리픽스(카테고리·도구·의도 정의)를 고쳤다면 서버를 재시작해야 새 캐시가 만들어집니다.
- **요청마다 복사본을 씁니다.** 캐시는 계산할 때마다 그 자리에서 늘어나는데, Qwen3.5 는 일부 층이
  선형 어텐션이라 늘어난 부분을 잘라내 되돌릴 수 없기 때문입니다. (복사 비용은 프리픽스 재계산보다 훨씬 작습니다)
- **안전장치** — 처음 만들 때 같은 입력을 캐시 사용/미사용으로 한 번씩 계산해 번호 판정이 같은지
  검증합니다(`LLM_KV_CACHE_VERIFY`). 다르거나, 사용 중 오류가 나면 **스스로 꺼지고** 기존 방식(전체 입력 계산)으로
  같은 결과를 냅니다. 상태는 `GET /api/v1/analyze/status` 의 `llm.kv_cache` 에서 봅니다.

| `llm.kv_cache` 필드 | 의미 |
|---|---|
| `ready` | 캐시가 만들어져 사용 중인지 |
| `prefix_tokens`, `prefill_ms` | 캐시된 프리픽스 토큰 수와 prefill 에 걸린 시간 |
| `verified` | 검증 결과 (`same_choice`, `max_prob_diff`) |
| `hits`, `fallbacks` | 캐시를 쓴 호출 수 / 오류로 기존 방식으로 넘어간 수 |
| `disabled_reason` | 꺼졌다면 그 이유 |

| 설정 | 기본값 | 의미 |
|---|---|---|
| `LLM_KV_CACHE` | `true` | `false` 면 매 요청 전체 입력을 계산 (기존 방식) |
| `LLM_KV_CACHE_VERIFY` | `true` | 처음 만들 때 캐시 사용/미사용 결과 비교 |

### 5.4 벡터DB (라벨링된 사례)

저장 방식은 `CASE_STORE_BACKEND` 로 고릅니다 — `numpy`(기본, Colab 테스트용) 또는 `pgvector`(PostgreSQL, 1-6 참고).
두 방식의 검색 결과는 같고, 아래 동작도 같습니다.

사용자와 무관한 **분류 보조 전용**이며, 비슷한 과거 사례의 카테고리 라벨로 top-3 후보를 다시 정렬한 뒤
⑤ 에 넘깁니다. 언제·어떻게 쓸지는 `CASE_MODE` 로 고릅니다.

| `CASE_MODE` | 동작 |
|---|---|
| `low_confidence` (기본) | ④ 1위 유사도가 `CANDIDATE_MIN_SCORE`(기본 0.65) 미만일 때만 섞기 |
| `always` | 매번 섞기 |
| `union` | 매번 섞고, **사례 점수 1위 카테고리가 top-3 밖이면 3위 자리에 넣기** (④ 와 사례 중 하나만 맞아도 정답이 후보에 들어감) |

④ 가 '확신한 채로 틀리면'(1위 점수가 높은데 오답) `low_confidence` 는 벡터DB 를 열지 않아 사례가 도움이
되지 못합니다. 어느 방식이 나은지는 학습 노트북 **6-1 측정 셀**로 records 에서 직접 비교해 고르세요.

④ 의 카테고리 점수 계산도 `CANDIDATE_SCORING` 으로 고를 수 있습니다.
`single` 은 카테고리 정의문 벡터 1개와 비교하고, `multi`(기본, 6-1 측정으로 고름)는 정의문과 키워드 하나하나의 벡터 중
가장 가까운 것의 점수를 씁니다. (주제가 많은 카테고리가 '평균 벡터' 때문에 흐려지는 문제를 줄임)
⑤ 의 프롬프트는 바뀌지 않으므로 KV 캐시와 충돌하지 않습니다.

```
사례 점수(c) = CASE_MIN_SCORE 이상인 유사 사례(최대 CASE_TOP_K개) 중 라벨이 c 인 것들의 유사도 합 / CASE_TOP_K
최종 점수(c) = (1 - CASE_BLEND_WEIGHT) × ④ 점수(c) + CASE_BLEND_WEIGHT × 사례 점수(c)
union 이면 : 최종 점수로 정렬한 뒤 사례 점수 1위 카테고리가 top-3 밖이면 3위 자리에 넣음
```

| 파일 | 내용 |
|---|---|
| `data/cases.csv` | 원본. 컬럼 `text`, `category` (카테고리는 이름 또는 코드). **사람이 편집하는 파일** |
| `data/cases.npy` | 사례 질의문 벡터 (N, 1024) — 자동 생성 |
| `data/cases.meta.json` | 사례 목록 + 지문 — 자동 생성 |

- 사례는 ④ 와 똑같이 **"키워드 + 요약" 질의문**으로 만들어 bge-m3 로 임베딩합니다.
- **서버 기동 시** CSV 를 읽습니다. CSV 내용·임베딩 모델·키워드/요약 설정이 그대로면 `.npy` 만 읽고,
  하나라도 바뀌었으면 다시 임베딩합니다. (그때만 bge-m3 를 올리므로 기동이 느려집니다)
- 7종에 없는 카테고리가 적힌 행은 건너뛰고 로그에 경고를 남깁니다.
- 동봉된 `data/cases.csv` 는 총 542건(헷갈리는 경계 사례는 `memo` 컬럼에 설명)인 **예시 데이터**입니다. 실제 민원 사례로 바꾸거나
  추가하세요. 줄임말·오타 섞인 짧은 문장(예: `골목 가로등 불 안 들어옴`)도 일부 넣어 두었습니다.
- 끄려면 `CASE_STORE_ENABLED=false`. 그러면 ④ 결과를 그대로 ⑤ 에 넘깁니다.

| 설정 | 기본값 | 의미 |
|---|---|---|
| `CASE_STORE_ENABLED` | `true` | 벡터DB 사용 여부 |
| `CASE_CSV_PATH` | `./data/cases.csv` | 사례 CSV 경로 |
| `CASE_STORE_BACKEND` | `numpy` | `numpy` 또는 `pgvector` |
| `CASE_PG_DSN` | (비어 있음) | pgvector 접속 정보 `postgresql://사용자:비밀번호@호스트:5432/DB` |
| `CASE_PG_TABLE` | `complaint_cases` | pgvector 테이블 이름 |
| `CASE_TOP_K` | `5` | 가져올 유사 사례 수 |
| `CASE_MIN_SCORE` | `0.5` | 이보다 덜 비슷한 사례는 반영하지 않음 |
| `CASE_BLEND_WEIGHT` | `0.2` | 0 이면 ④ 그대로, 1 이면 사례 투표만 |
| `CASE_MODE` | `low_confidence` | `low_confidence` / `always` / `union` (위 표) |
| `CANDIDATE_SCORING` | `single` | ④ 점수 방식 `single` / `multi` |

조회 결과는 `debug.case_lookup` 과 `debug.debug_text` 의 "④ 보조 : 벡터DB" 구간에서 볼 수 있습니다.

### 5.5 ⑥ 민원 DB (SQLite - 접수/조회/수정/취소 실제 실행)

⑤ 가 만드는 건 "무엇을 어떤 값으로" 실행할지 담은 **도구 호출 JSON** 뿐입니다. 그 JSON 을 실제로
DB 에 쓰거나 읽는 코드는 Qwen 과 무관한, 일반 파이썬 함수(`app/services/tool_executor.py` +
`app/services/complaint_store.py`)입니다. Qwen 이 만든 값은 여기서 스키마·소유권·상태를
다시 검증한 뒤에만 실행되므로, 모델이 없는 id 나 잘못된 카테고리를 적어도 그대로 실행되지 않습니다.

**왜 PostgreSQL(pgvector처럼)이 아니라 SQLite인가** — 이 DB 를 실제로 써야 하는 곳이 Colab(모델)과
PC(추출) 양쪽 다인데, Colab 은 PC 의 PostgreSQL 에 네트워크로 닿을 수 없습니다. (벡터DB 와 같은 제약,
1-6 참고) 민원 등록·조회·수정·취소는 트래픽이 낮고 스키마가 단순해 파일 하나(`data/complaints.db`,
표준 라이브러리 `sqlite3`, 설치 불필요)로 충분합니다. 나중에 운영 규모가 커지면
`complaint_store.py` 의 함수 시그니처를 유지한 채 PostgreSQL 구현으로 바꿔 끼울 수 있습니다.

**표 구조**

| 테이블 | 내용 |
|---|---|
| `complaints` | 민원 1건 = 1행. id·user_id·category·content·location·status·생성/수정 시각 |
| `complaint_history` | 상태가 바뀔 때마다(접수/수정/취소) 한 행씩 쌓이는 이력 |

**로그인 사용자 가정** — 아직 실제 인증이 없어, API 로 넘기는 `user_id` 문자열을 그대로
"로그인한 사용자"로 취급합니다. 비우면 `.env` 의 `COMPLAINT_DEFAULT_USER`(기본 `demo-user`)를
씁니다. 조회·수정·취소는 항상 "그 user_id 가 등록한 민원인지"를 같이 확인하므로, 다른 사용자의
민원 id 를 적어도 "찾을 수 없거나 본인 민원이 아닙니다" 로 막힙니다.

**의도별 실행 방식**

| 의도 | 실행 시점 | 비고 |
|---|---|---|
| 문의 | (해당 없음) | ⑥ 을 거치지 않고 ⑤ 의 즉답을 그대로 씁니다 |
| 접수 | 검증 통과 시 즉시 실행 | 중복을 허용합니다 |
| 조회 | 즉시 실행 | 읽기 전용. 찾은 민원이 1건이면 상태/이력, 여러 건이면 목록 |
| 수정 | 대상이 **1건**으로 정해지면 즉시 실행 | 여러 건이거나 조건이 없으면 실행하지 않고 후보를 돌려줌 |
| 삭제 | 대상이 **1건**으로 정해지면 즉시 실행 | 상태를 '취소'로 바꾸는 soft delete. 여러 건·조건 없음은 수정과 같음 |

**번호 없이 민원 찾기 (조회·수정·삭제)** — 시민은 민원 번호를 기억하지 못하는 경우가 대부분이라,
"어제 넣은 가로등 민원", "방금 신고한 거"처럼 말해도 찾습니다. ⑤ 가 도구 인자에 찾기 조건을 채웁니다.

| 인자 | 채우는 쪽 | 내용 |
|---|---|---|
| `complaint_id` | Qwen | 번호를 말했을 때만. 있으면 다른 조건보다 우선 |
| `category` | 코드 | ⑤ 가 확정한 카테고리. 조회·수정·삭제는 후보 3개 + **없음** 중에서 고르며, 주제가 안 드러나면 없음(`""`) |
| `keyword` | Qwen | 대상 표현을 원문 그대로 (예: `가로등`). 내용·위치에 글자로 있으면 일치, 없으면 bge-m3 의미 유사도 ≥ `SEARCH_SEMANTIC_MIN_SCORE` |
| `period` | Qwen | 시점 표현을 원문 그대로 (예: `어제`, `지난주`, `방금`). 날짜 계산은 `app/services/period.py` 가 `TIMEZONE` 기준으로 함 |

- 못 찾으면 조건을 조금 풀어 한 번 더 찾습니다. (카테고리 조건 빼기 → 기간 앞뒤 하루 넓히기) 푼 경우 `tool_result.warnings` 에 남습니다.
- 수정·삭제는 이미 취소된 민원을 대상에서 뺍니다.
- 수정·삭제 대상이 **여러 건**이거나 **조건이 하나도 없으면**(예: "민원 취소할게요") 실행하지 않고
  `tool_result.needs_selection=true` 와 후보 목록(`data`)을 돌려줍니다. 화면에서 하나를 고르게 한 뒤
  번호를 넣어 다시 요청하면 됩니다. (예: "37번 민원 취소해 주세요")
- 어떤 조건으로 몇 건을 찾았는지는 `tool_result.search` 에 있습니다.

알아듣는 시점 표현: 오늘·어제·그저께·엊그제, N일 전·사흘 전·며칠 전·일주일 전·N주 전·한 달 전,
이번 주·지난주·지지난주·주말, 이번 달·지난달, M월 D일·M월, 올해·작년, 최근·요즘(30일),
방금·아까·마지막·가장 최근(가장 최근 1건), 지난번·저번(기간 제한 없음). 모르는 표현은 기간 조건 없이 찾습니다.

검증에 실패하면(예: 없는 id, 남의 민원, 조건에 맞는 진행 중 민원 없음) `tool_result.ok=false` 와
`error` 로 이유가 담기고 DB 는 바뀌지 않습니다.

`POST /analyze/text` 요청 본문에 `user_id` 를 추가로 넣을 수 있습니다.

```json
{ "text": "어제 넣은 가로등 민원 위치를 행복로 15길로 바꿔주세요.", "user_id": "kim01" }
```

응답의 `tool_result` 에 실행 결과가 담깁니다. (`executed`=DB 반영 여부, `ok`=검증 통과 여부,
`message`=사람이 읽는 문장, `data`=민원 1건 또는 목록, `error`=검증 실패 사유,
`needs_selection`=대상을 하나로 정하지 못해 실행하지 않음, `search`=찾은 조건)

```json
{
  "tool_result": {
    "executed": true, "ok": true,
    "message": "수정 완료 - 37번 민원: 위치 '행복로 23길' → '행복로 15길'",
    "data": { "id": 37, "category": "국토교통", "location": "행복로 15길", "status": "접수", "department": "국토교통과" },
    "error": null,
    "needs_selection": false,
    "search": { "complaint_id": 0, "category": "국토교통", "keyword": "가로등", "period_text": "어제",
                "period": "'어제' → 2026-09-29", "condition": "국토교통 · '가로등' · 어제", "notes": [], "matched": 1 }
  }
}
```

Colab 노트북 6-A 의 "⑥ 민원 DB 실제 동작 확인" 셀에서 접수 → 조회 → 수정 → 취소 → 이력 조회를
한 번에 실행해 볼 수 있습니다.

| 설정 | 기본값 | 의미 |
|---|---|---|
| `COMPLAINT_DB_PATH` | `./data/complaints.db` | SQLite 파일 경로. 없으면 자동 생성 |
| `COMPLAINT_DEFAULT_USER` | `demo-user` | API 가 `user_id` 를 비워 보냈을 때 쓸 이름 |
| `TIMEZONE` | `Asia/Seoul` | 시점 표현("어제")을 날짜로 바꿀 기준 시간대 |
| `SEARCH_SEMANTIC_MIN_SCORE` | `0.55` | keyword 가 글자 그대로 없을 때 같은 민원으로 볼 의미 유사도 |

**알려진 한계** — 대화 상태(세션)가 없어, 후보 목록에서 고른 민원을 서버가 기억하지 못합니다.
`needs_selection=true` 를 받은 화면은 사용자가 고른 민원의 번호를 넣어 다시 요청해야 합니다.

### 5.6 모델 로드

- 가중치는 HuggingFace 에서 자동으로 내려받습니다. **첫 요청만 수 분 걸립니다.** (1-7 참고)
- `.env` 의 `DEVICE=auto` 가 기본값입니다. CUDA 가 보이면 GPU, 없으면 CPU 로 자동 선택합니다.
- CUDA + bitsandbytes 가 있으면 **4bit NF4(QLoRA 와 같은 설정)** 로, CPU 면 float32 로 올립니다.
- 4bit 연산 dtype 은 GPU 에 맞춰 자동으로 고릅니다. **T4(Turing)는 bfloat16 이 없으므로 float16**,
  Ampere(RTX30/A100) 이상이면 bfloat16 입니다. (`LLM_4BIT_COMPUTE_DTYPE=auto`)
- **Qwen3.5 는 기본이 사고(thinking) 모드입니다.** 의도/카테고리 판정은 다음 토큰 1개만 보므로
  켜 두면 그 1개가 `<think>` 가 되어 판정이 깨집니다. `LLM_ENABLE_THINKING=false` 로 두세요.
  (채팅 템플릿에 `enable_thinking=False` 를 넘기며, 혹시 섞여 나오는 `<think>` 블록은 잘라냅니다)
- Qwen3.5 체크포인트는 비전 인코더가 붙은 구조라, 설치된 transformers 가 `AutoModelForCausalLM` 으로
  못 올리면 자동으로 `AutoModelForImageTextToText` 로 다시 올립니다. 텍스트만 넣으므로 판정 방식은 같습니다.

로드 상태는 `GET /api/v1/analyze/status` 의 `loaded` 로 확인합니다.

### 5.7 QLoRA 학습과 어댑터 붙이기

학습은 **`colab_train.ipynb`** 하나에서 합니다. (데이터 검증 → ④ 후보 계산 → 학습 샘플 → 학습 전 평가 →
학습(실시간 loss 그래프·중간 출력) → 학습 후 평가·전후 비교·판정 → 저장 → 운영 경로로 재확인)

6-1 · 6-2 셀은 학습 전에 ④ 설정(`CANDIDATE_SCORING`, `CASE_MODE`, `CANDIDATE_MIN_SCORE`, `CASE_BLEND_WEIGHT`)을
records 로 비교해 고르고 `.env` 에 반영합니다. ④ 가 정답을 후보 3개에 못 올리면 ⑤ 를 학습해도 맞힐 수 없으므로,
`④ 놓침` 이 많으면 학습보다 먼저 여기서 줄이세요. (`training/calibrate.py`)

**무엇을 학습하나** — 어댑터 하나에 도구 호출 JSON 의 인자 추출(content / location / complaint_id / keyword / period / field /
reason)을 주력으로, 의도·카테고리 판정을 소량(기본 각 15%) 섞어 기존 능력을 붙잡아 둡니다.
'문의' 즉답은 학습하지 않습니다. (안내 지식이 바뀌면 가중치에 굳은 옛 지식이 남기 때문)

**학습 입력 = 추론 입력** — 학습 샘플은 운영 코드의 함수로 조립하므로 토큰 단위로 같습니다.

| 조각 | 학습에서 쓰는 운영 함수 |
|---|---|
| 고정 프리픽스·채팅 템플릿 | `llm_engine.template_parts(tokenizer)` |
| 단계별 프롬프트 | `prompts.build_intent_prompt` / `build_category_prompt` / `build_tool_prompt` |
| 번호 정답 토큰, 어시스턴트 앞머리 | `llm_engine.number_token_ids`, `INTENT_ASSISTANT_PREFIX` / `CATEGORY_ASSISTANT_PREFIX` |
| 도구 JSON 정답 형식 | `prompts.format_tool_call` (한 줄, 키 순서 고정) |
| ④ 후보 3개 | `pipeline.run_candidates` (벡터DB 재정렬 포함) |
| 베이스 모델 로드 | `llm_engine.load_base_model()` (서버와 같은 4bit 설정) |

평가도 `llm_engine.judge_intent` / `judge_category` / `generate_tool_call` 을 그대로 부르므로 노트북 점수가 곧 서버 동작입니다.

**정답 데이터** — `data/train/records.jsonl`, 한 줄에 한 건. 형식과 라벨링 규칙은 `training/records.py` 상단과
노트북 5단계에 있습니다. 예시 90건이 `training/sample_data/` 에 있습니다. (형식 확인용이며 실제 개선에는 부족)
학습/검증/평가 분할은 `group` 해시로 정해 **데이터를 늘려도 평가셋이 바뀌지 않습니다.**

**적용** — 학습이 끝나면 `adapters/<이름>/` 에 저장됩니다. `.env` 에 그 경로를 적으면 서버가 얹습니다.

```bash
LLM_ADAPTER_PATH=adapters/v1_0929     # 비우면 베이스 모델 (되돌리기)
```

어댑터 폴더의 `run_info.json` 에는 학습 당시의 베이스 모델·프롬프트 지문(`prompts.prompt_fingerprint()`)·
사고 모드 설정이 남습니다. 서버는 어댑터를 올릴 때 지금 설정과 대조해서 다르면 로그에 경고를 남기고,
`/analyze/status` 의 `llm.adapter_check.mismatches` 에도 보여 줍니다. **학습 뒤 `prompts.py` 를 고쳤다면 재학습이 필요합니다.**

### 5.8 DEBUG 출력

`.env` 의 `DEBUG=true` 이면 분류 응답에 `debug` 블록이 붙습니다.

| 필드 | 내용 |
|---|---|
| `source_text` | ② 추출 원문 |
| `keywords`, `summary` | ②→③ 키워드 3개 + 요약문 |
| `embed_query_text` | ③ 임베딩 모델에 실제로 넘긴 질의문 |
| `embedding_dim`, `embedding_preview` | ③ 벡터 차원과 앞부분 미리보기 |
| `candidates_top`, `candidates_all` | ⑤ 에 넘긴 상위 3개와 7종 전체 점수 (벡터DB 로 재정렬했으면 재정렬 후 점수) |
| `low_confidence` | ④ 1위 유사도가 `CANDIDATE_MIN_SCORE` 미만인지 (`CASE_MODE=low_confidence` 면 true 일 때 벡터DB 조회) |
| `case_lookup` | 벡터DB 조회 결과 — 유사 사례, 카테고리별 사례 점수, 재정렬 전후 top-3 (조회했을 때만) |
| `intent_choice`, `category_choice` | ⑤ 후보별 확신 점수 전체 |
| `llm_raw_output` | ⑤ 모델 원시 출력 |
| `timings_ms` | 단계별 소요 시간 |
| `warnings` | ② 추출 경고(`[②추출]`) + ②~⑤ 경고 |
| `debug_text` | 위 내용을 한 덩어리 텍스트로 정리한 것 |

Swagger 에서 눈으로 볼 때는 **`debug.debug_text` 하나만 펼쳐 보면** 전 단계가 다 보입니다.
`DEBUG=false` 면 `debug` 는 `null` 이 되고 최종 판정 결과만 나갑니다.

**`tool_result`(⑥ 실행 결과)는 `debug` 와 달리 `DEBUG` 설정과 무관하게 항상 응답에 실립니다.**
도구를 호출한 의도(접수/조회/수정/삭제)에서만 채워지며, 자세한 필드는 5.5 를 보세요.

### 5.9 게이트 (⑤ 의도 판정 기준 반려)

⑤ 의 의도 판정은 **6지선다**입니다: 1~5(문의/접수/조회/수정/삭제) + **6(해당없음)**.
"해당없음"은 잡담·광고·욕설·의미 없는 문자열·무엇을 원하는지 특정할 수 없는 모호한 말을 위해 둔
여섯 번째 선택지입니다. Qwen 이 이 번호를 고르면 카테고리 확정과 도구 호출을 생략하고 곧바로 반려합니다.
의도 판정(forward 1회)만 쓰고 멈추므로, 반려되는 요청이 오히려 더 빠릅니다.

**게이트는 카테고리(④)와 무관합니다.** ④ 의 코사인 유사도는 후보 3개를 좁히고 벡터DB 를 열지 정하는 데만 쓰입니다.

- `"제가 어제 문의한 내용 보여줘"` — 7종 어디와도 뚜렷이 겹치지 않지만, 의도(조회)가 명확하므로 **정상 통과**합니다.
- `"오늘 점심 뭐 먹지 ㅋㅋ"` — 5가지 의도 중 어디에도 해당하지 않으므로 **반려**됩니다.

반려도 정상 처리 결과이므로 HTTP 200 으로 돌려줍니다.

```json
{
  "status": "rejected",
  "gate": { "passed": false, "intent_score": 0.91, "enabled": true,
            "reason": "의도 판정 결과 '해당없음' (확신 0.9100) - 문의·접수·조회·수정·삭제 중 어디에도 해당하지 않는다고 판단" },
  "intent": "해당없음", "intent_code": "out_of_scope", "intent_score": 0.91,
  "category": null, "tool_call": null,
  "answer": "죄송합니다. 문의·접수·조회·수정·삭제 중 무엇을 원하시는지 확인하지 못해 처리하지 못했습니다. 다시 한번 말씀해 주시겠어요? ..."
}
```

| 설정 | 의미 |
|---|---|
| `GATE_ENABLED=true` | `false` 면 차단하지 않고 반려 사유만 `gate.reason` 에 남깁니다 (관찰 모드) |
| `CANDIDATE_MIN_SCORE=0.65` | ④ `low_confidence` 기준값. `CASE_MODE=low_confidence` 일 때 벡터DB 조회 기준. 반려와 무관합니다 |

`GATE_ENABLED=false` 로 두면 "해당없음"으로 판정된 요청도 통과는 시키되, 카테고리·도구는 여전히 생략되고
`debug.debug_text` 의 "게이트" 구간에 반려됐을 사유가 남습니다. 정상 민원과 잡담·광고를 몇십 개씩 돌려
의도 확신도(`gate.intent_score`)가 실제로 잘 갈리는지 확인할 때 씁니다.

**이 게이트가 못 막는 것** — 민원 어휘가 있고 의도 자체는 명확한 이상한 입력은 통과합니다.
`"이 자동차 제가 접수 했나요?"` 는 대상을 특정할 수 없는 게 문제인데, 의도(조회)는 여전히 명확하므로
이 게이트로는 걸러지지 않습니다. 이런 경우는 뒤쪽의 슬롯 검사(되묻기)나 ⑥ DB 조회 결과로 처리해야 합니다.

### 5.10 아직 구현하지 않은 것

| 그림의 요소 | 상태 |
|---|---|
| ⑥ 민원 DB 의 실제 인증 | `user_id` 를 API 호출자가 그대로 넘겨줘야 합니다. 로그인 세션에서 자동으로 채워주는 계층은 아직 없습니다. |
| ⑥ 수정/삭제의 대상 지정 | 번호·주제·시점으로 찾고, 여러 건이면 후보를 돌려줍니다. 화면에서 고른 id 를 서버가 기억하는 대화 상태(세션)는 아직 없어 번호를 넣어 다시 요청해야 합니다. |
| ⑦ 사용자 응답 | ⑤⑥ 결과를 `result_text` 텍스트로 대신합니다. 사용자에게 보여줄 문구를 다듬는 별도 단계는 없습니다. |

## 6. 오류 응답

모든 오류는 아래 형식으로 통일해서 응답하며, 콘솔과 `logs/app.log` 에 함께 기록됩니다.

```json
{
  "success": false,
  "error_code": "OCR_UNAVAILABLE",
  "message": "이미지 처리에 필요한 PP-OCRv5(PaddleOCR)가 설치되어 있지 않습니다.",
  "detail": "README.md 의 1-3. PP-OCRv5 설치를 참고하세요.",
  "created_at": "2026-09-21T07:20:11.004Z"
}
```

| 상태코드 | error_code | 의미 |
|---|---|---|
| 400 | `EMPTY_FILE` | 빈 파일 |
| 413 | `FILE_TOO_LARGE` | `.env` 의 `MAX_UPLOAD_MB` 초과 |
| 415 | `UNSUPPORTED_FORMAT` | 지원하지 않는 확장자 |
| 422 | `EXTRACTION_FAILED` | 파일 손상, 암호 설정 등으로 파싱 실패 |
| 422 | `NO_TEXT` | 추출된 텍스트가 비어 분류할 수 없음 (`/analyze/*`) |
| 422 | `VALIDATION_ERROR` | 요청 형식 오류 |
| 503 | `OCR_UNAVAILABLE` | PP-OCRv5 미설치 (이미지 파일 요청 시) |
| 503 | `MODEL_UNAVAILABLE` | bge-m3 / Qwen 미설치·로드 실패 (`/analyze/*`) |
| 500 | `INTERNAL_ERROR` | 처리되지 않은 예외 (`DEBUG=true` 면 `detail` 에 원인) |

## 7. 성능 참고

- **첫 OCR 요청과 첫 분류 요청은 느립니다.** 가중치를 내려받고 메모리에 올리기 때문이며, 이후 요청은 재사용합니다.
- OCR 속도가 급하면 `.env` 에서 `OCR_DET_MODEL=PP-OCRv5_mobile_det`, `OCR_DPI=150`,
  `PDF_OCR_EMBEDDED_IMAGES=false` 를 조합해 보세요.
- OCR 정확도가 급하면 `OCR_DPI=300` 으로 올리세요. 작은 글씨 인식률이 올라갑니다.

## 8. 폴더 구조

```
backend/
├── .env.example              # 설정 템플릿 (모든 항목 설명 포함)
├── .gitignore
├── requirements.txt          # 웹 + ①② 추출 (torch 없음)
├── requirements-model.txt    # ③④⑤ 모델 스택 (Colab / GPU 서버 전용)
├── colab_backend.ipynb       # Colab T4 실행 노트북 (서버·파이프라인 확인)
├── colab_train.ipynb         # Colab T4 QLoRA 학습 노트북 (5.7)
├── data/
│   ├── cases.csv             # 벡터DB 원본 (라벨링된 사례). .npy/.meta.json 은 기동 시 자동 생성
│   └── train/
│       ├── records.jsonl     # 학습 정답 데이터 (직접 관리, 없으면 노트북이 예시를 복사)
│       └── cache/            # ④ 후보·기준선 평가 캐시 (지워도 다시 생성)
├── adapters/                 # 학습된 어댑터 버전들 (adapters/<이름>/ + run_info.json + 평가 결과)
├── training/
│   ├── records.py            # 정답 레코드 형식·검증·분할 (group 해시로 고정)
│   ├── samples.py            # 정답 -> 학습 샘플 (추론과 같은 함수로 조립, 정답에만 loss)
│   ├── evaluate.py           # 운영 코드로 채점, 학습 전후 비교·판정
│   ├── trainer.py            # LoRA 부착·학습·실시간 그래프·저장·버전 비교
│   ├── calibrate.py          # ④ 방식 측정 (벡터DB 사용 방식·문턱·가중치 비교, 노트북 6-1)
│   ├── build_notebook.py     # colab_train.ipynb 생성 스크립트
│   └── sample_data/sample_records.jsonl   # 예시 정답 90건
├── scripts/
│   ├── setup_pgvector.sql    # PostgreSQL DB 생성 + vector 확장 켜기 (1-6)
│   └── sync_cases.py         # 벡터DB 적재 + 자기 벡터 검색으로 동작 확인 (1-6)
├── README.md
└── app/
    ├── __init__.py
    ├── main.py               # FastAPI 앱, 예외 처리, /health
    ├── config.py             # .env 로딩
    ├── schemas.py            # pydantic 요청/응답 모델
    ├── exceptions.py         # 공통 예외
    ├── logging_config.py     # 콘솔 + 일자별 파일 로깅
    ├── routers/
    │   ├── extract.py        # POST /extract/text, GET /formats
    │   └── analyze.py        # POST /analyze/text·file, GET /analyze/categories·status
    └── services/
        ├── base.py           # Segment / ExtractedText 공통 자료구조
        ├── file_types.py     # 확장자 판별 및 담당 모듈 결정
        ├── extraction.py     # ② 추출 공통 진입점 (/extract, /analyze/file 공유)
        ├── doc_extractor.py  # Office / 한글 / 텍스트
        ├── pdf_extractor.py  # PyMuPDF + 페이지별 OCR 분기
        ├── image_extractor.py# 이미지 전용 (PP-OCRv5)
        ├── ocr_engine.py     # PP-OCRv5 호출 및 결과 정렬
        ├── keyphrase.py      # ②→③ 키워드 3개 + 요약문
        ├── embedder.py       # ③ bge-m3
        ├── categories.py     # 카테고리 7종 정의
        ├── candidates.py     # ④ 코사인 유사도 top-3
        ├── case_store.py     # ④ 보조 벡터DB (라벨링된 사례로 후보 재정렬, numpy/pgvector 선택)
        ├── pg_store.py       # 벡터DB 의 PostgreSQL + pgvector 저장소
        ├── prompts.py        # ⑤ 고정 프리픽스(CAG 대상) + 단계별 프롬프트
        ├── llm_engine.py     # ⑤ Qwen 로드 / KV 캐시 재사용 / 번호 토큰 판정 / 도구 JSON
        ├── complaint_store.py# ⑥ 민원 DB (SQLite - 접수/조회/수정/취소)
        ├── tool_executor.py  # ⑤ 의 도구 호출 JSON을 ⑥ 에서 실제로 검증·실행 (번호 없이 민원 찾기 포함)
        ├── period.py         # 시점 표현("어제", "지난주") -> 날짜 범위
        ├── runtime.py        # 디바이스·4bit·dtype 판단
        └── pipeline.py       # ②~⑥ 오케스트레이션 + 게이트 + 텍스트 리포트
```
