"""
애플리케이션 설정.

모든 설정값은 backend/.env 파일에서 읽어옵니다.
(.env 가 없으면 아래 기본값 또는 OS 환경변수를 사용합니다)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ 디렉터리 (app/config.py -> app -> backend)
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- 애플리케이션 ---
    APP_NAME: str = "Document AI Backend"
    APP_ENV: str = "local"
    DEBUG: bool = True
    API_PREFIX: str = "/api/v1"

    # --- 업로드 ---
    MAX_UPLOAD_MB: int = 20

    # --- PP-OCRv5 (PaddleOCR) ---
    # 모델 이름만 바꾸면 코드 수정 없이 다른 PP-OCRv5 모델을 쓸 수 있습니다.
    OCR_DET_MODEL: str = "PP-OCRv5_server_det"
    OCR_REC_MODEL: str = "korean_PP-OCRv5_mobile_rec"
    OCR_DEVICE: str = ""              # "" = 자동, "cpu", "gpu:0"
    OCR_MIN_SCORE: float = 0.5        # 이 값 미만의 인식 결과는 버립니다
    OCR_DPI: int = 200                # PDF 페이지/이미지 영역 렌더링 해상도
    OCR_MIN_IMAGE_PX: int = 80        # 이보다 작은 이미지는 아이콘으로 보고 건너뜀
    OCR_USE_TEXTLINE_ORIENTATION: bool = False  # 세로쓰기/회전 문서면 true

    # --- PDF 처리 ---
    # 페이지 텍스트가 이 글자 수 미만이면 "텍스트 없음"으로 보고 페이지 전체를 OCR 합니다.
    PDF_MIN_TEXT_CHARS: int = 10
    # 텍스트가 있는 페이지에 박힌 이미지도 잘라서 OCR 할지 여부
    PDF_OCR_EMBEDDED_IMAGES: bool = True

    # =========================================================================
    # 분류 파이프라인 (③ 임베딩 / ④ 후보 추림 / ⑤ Qwen)
    # =========================================================================

    # --- 공통 실행 환경 ---
    # auto = CUDA 가 보이면 GPU, 없으면 CPU. ("cpu" / "cuda" / "cuda:0" 도 가능)
    DEVICE: str = "auto"
    # HuggingFace 모델 캐시 경로. 비우면 HF 기본 경로(~/.cache/huggingface)를 씁니다.
    MODEL_CACHE_DIR: str = ""
    # true 면 서버 기동 시 모델을 미리 로드합니다. (첫 요청이 빨라지지만 기동이 느립니다)
    PRELOAD_MODELS: bool = False
    # ②→⑤ 로 넘길 원문 최대 길이. 이보다 길면 앞부분만 사용합니다.
    PIPELINE_MAX_INPUT_CHARS: int = 4000

    # --- ②→③ 키워드 3개 + 요약문 ---
    KEYWORD_TOP_K: int = 3            # 그림의 "키워드 3개"
    KEYWORD_USE_MMR: bool = True      # 비슷한 키워드가 겹치지 않게 분산시킵니다
    KEYWORD_MMR_DIVERSITY: float = 0.5
    SUMMARY_MAX_SENTENCES: int = 3    # TextRank 로 고를 문장 수
    SUMMARY_MAX_CHARS: int = 400
    SUMMARY_DAMPING: float = 0.85     # PageRank 감쇠 계수
    SUMMARY_ITERATIONS: int = 30
    # ④ 질의문에서 상투 문구("여러 번 말씀드렸어요", "신속한 처리 부탁드립니다" …)를 뺍니다. ⑤ 입력은 그대로.
    QUERY_DROP_BOILERPLATE: bool = True
    # 상투 문구를 뺀 본문이 이 글자 수 이하면 요약하지 않고 전부 씁니다.
    SUMMARY_FULL_TEXT_CHARS: int = 200
    # TextRank 로 고를 때도 첫 문장은 항상 넣습니다. (핵심이 대개 첫 문장)
    SUMMARY_KEEP_FIRST: bool = True

    # --- ③ 임베딩 (bge-m3) ---
    EMBED_MODEL: str = "BAAI/bge-m3"  # 1024차원. 학습하지 않고 그대로 사용합니다.
    EMBED_NORMALIZE: bool = True      # L2 정규화 (내적 = 코사인 유사도)
    EMBED_BATCH_SIZE: int = 16
    EMBED_MAX_SEQ_LENGTH: int = 1024  # 8192 까지 가능하지만 길수록 느립니다

    # --- ④ 후보 추림 ---
    CANDIDATE_TOP_K: int = 3          # 그림의 "top-3"
    # 1위 유사도가 이보다 낮으면 candidates.low_confidence=true 로 표시하고 벡터DB(사례)를
    # 열어 후보를 재정렬합니다. (아래 게이트의 반려 기준은 아닙니다)
    CANDIDATE_MIN_SCORE: float = 0.65  # 학습 노트북 6-1 측정으로 고른 값 (예전 기본 0.45)
    # 카테고리 점수 계산 방식
    #   single : 카테고리 정의문(설명+키워드 전체) 벡터 1개와 비교 (예전 기본)
    #   multi  : 정의문 벡터 + 키워드 하나하나의 벡터 중 가장 가까운 것의 점수
    #            (주제가 많은 카테고리가 '평균 벡터' 때문에 흐려지는 문제를 줄임)
    CANDIDATE_SCORING: str = "multi"   # 학습 노트북 6-1 측정으로 고른 값

    # --- ④ 보조 : 벡터DB (라벨링된 사례) ---
    # ④ 가 low_confidence 일 때만 비슷한 사례의 라벨로 top-3 후보를 재정렬합니다.
    CASE_STORE_ENABLED: bool = True
    # 사례 CSV (컬럼 text, category). 같은 폴더에 .npy / .meta.json 이 자동 생성됩니다.
    CASE_CSV_PATH: str = "./data/cases.csv"
    CASE_TOP_K: int = 5               # 가져올 유사 사례 수
    CASE_MIN_SCORE: float = 0.5       # 이보다 덜 비슷한 사례는 반영하지 않습니다
    CASE_BLEND_WEIGHT: float = 0.2    # 최종 = (1-w)*④점수 + w*사례점수 (6-1 측정으로 고른 값, 예전 0.5)
    # 벡터DB 를 쓰는 방식 (학습 노트북 6-1 측정 셀로 고르세요)
    #   low_confidence : ④ 1위 < CANDIDATE_MIN_SCORE 일 때만 섞기 (기존 방식)
    #   always         : 매번 섞기
    #   union          : 매번 섞고, 사례 점수 1위 카테고리를 후보에 반드시 포함
    CASE_MODE: str = "low_confidence"
    # 저장 방식: numpy (메모리, Colab 테스트용) | pgvector (PostgreSQL, PC·운영 서버용)
    CASE_STORE_BACKEND: str = "numpy"
    # pgvector 접속 정보. 예) postgresql://postgres:비밀번호@localhost:5432/minwon
    CASE_PG_DSN: str = ""
    CASE_PG_TABLE: str = "complaint_cases"

    # --- ⑥ 민원 DB (SQLite - 등록/조회/수정/취소 실제 실행) ---
    # PostgreSQL 이 아니라 SQLite 파일인 이유는 README 의 "5.5 ⑥ 민원 DB" 참고.
    COMPLAINT_DB_PATH: str = "./data/complaints.db"
    # 아직 실제 로그인이 없어, 이 이름을 "로그인 사용자"로 취급합니다. (API 로 덮어쓸 수 있음)
    COMPLAINT_DEFAULT_USER: str = "demo-user"
    # 번호 없이 "어제 넣은 가로등 민원"처럼 찾을 때 (조회·수정·삭제)
    #   시점 표현("어제", "지난주")을 날짜로 바꿀 기준 시간대
    TIMEZONE: str = "Asia/Seoul"
    #   keyword 가 민원 내용·위치에 글자 그대로 없을 때, bge-m3 의미 유사도가 이 값 이상이면 같은 민원으로 봅니다.
    #   (예: "불 꺼진 거" -> "가로등 꺼짐")
    SEARCH_SEMANTIC_MIN_SCORE: float = 0.55

    # --- 게이트 ---
    # ⑤ 의 의도 판정은 6지선다입니다: 1~5(문의/접수/조회/수정/삭제) + 6(해당없음).
    # "해당없음"으로 판정되면 카테고리 확정·도구 호출을 생략하고 요청을 반려합니다.
    # 카테고리(④)와는 무관합니다 - "제가 어제 문의한 내용 보여줘"처럼 특정 카테고리와
    # 뚜렷이 겹치지 않는 문장도, 의도(조회)만 명확하면 통과합니다.
    # false 로 두면 차단하지 않고 반려 사유만 gate.reason 에 남깁니다. (관찰용)
    GATE_ENABLED: bool = True

    # --- ⑤ Qwen3.5 4B (+ QLoRA) ---
    # Colab T4(16GB)에서 4bit 로 올리는 것을 기준으로 잡았습니다.
    LLM_MODEL: str = "Qwen/Qwen3.5-4B"

    # Qwen3.5 는 기본이 '사고(thinking) 모드'라 답 앞에 <think>...</think> 를 먼저 씁니다.
    # 의도/카테고리는 번호 토큰 1개만 보므로 사고 모드가 켜져 있으면 판정이 깨집니다.
    # 반드시 false 로 두세요. (true 로 두면 <think> 블록을 잘라내고 처리하지만 훨씬 느립니다)
    LLM_ENABLE_THINKING: bool = False
    # 모델 저장소의 자체 코드(modeling_*.py)를 실행해야 올라가는 모델만 true. (예: 일부 EXAONE 판)
    # Qwen·Llama·Gemma 처럼 transformers 가 기본 지원하는 모델은 false 로 둡니다.
    LLM_TRUST_REMOTE_CODE: bool = False
    # QLoRA 어댑터 경로(로컬 폴더 또는 HF repo id). 비우면 베이스 모델만 씁니다.
    LLM_ADAPTER_PATH: str = ""
    # 4bit NF4 양자화 (QLoRA 와 동일 설정). CUDA + bitsandbytes 일 때만 적용됩니다.
    # Colab T4 에서 4B 모델을 올리려면 반드시 true 여야 합니다. (fp16 로는 8GB 를 먹습니다)
    LLM_LOAD_4BIT: bool = True

    # 4bit 연산 dtype. auto = GPU 가 bfloat16 을 지원하면 bfloat16, 아니면 float16.
    # T4(Turing)는 bfloat16 을 지원하지 않으므로 auto 면 자동으로 float16 이 됩니다.
    LLM_4BIT_COMPUTE_DTYPE: str = "auto"   # auto | bfloat16 | float16

    # KV 캐시(CAG) - 고정 프리픽스를 한 번만 prefill 해 두고 요청마다 재사용합니다.
    LLM_KV_CACHE: bool = True
    # 처음 만들 때 캐시 사용/미사용 결과가 같은지 확인합니다. 다르면 자동으로 끕니다.
    LLM_KV_CACHE_VERIFY: bool = True
    LLM_MAX_INPUT_CHARS: int = 2000
    LLM_MAX_NEW_TOKENS_TOOL: int = 200    # 도구 호출 JSON 생성
    LLM_MAX_NEW_TOKENS_ANSWER: int = 160  # '문의' 즉답 생성
    LLM_TEMPERATURE: float = 0.0          # 0 이면 greedy (판정은 재현성이 중요)
    LLM_TOP_P: float = 0.9

    # --- DEBUG 응답 ---
    # DEBUG=true 일 때 응답에 실어 보낼 임베딩 벡터 미리보기 개수
    DEBUG_VECTOR_PREVIEW: int = 8

    # --- 로깅 ---
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "./logs"
    LOG_TO_FILE: bool = True

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024

    @property
    def log_dir(self) -> Path:
        path = Path(self.LOG_DIR)
        if not path.is_absolute():
            path = BASE_DIR / path
        return path


@lru_cache
def get_settings() -> Settings:
    """설정 싱글턴. FastAPI 의존성으로도 사용할 수 있습니다."""
    return Settings()


settings = get_settings()
