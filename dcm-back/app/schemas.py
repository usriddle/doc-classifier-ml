"""
API 요청/응답 스키마 모음.

FastAPI 응답 모델이므로 Swagger 문서에 그대로 노출됩니다.

POST /extract/text 의 성공 응답은 청킹/임베딩 파이프라인에 그대로 넘기는 걸
전제로, 본문(text)과 그걸 식별하는 데 꼭 필요한 최소 메타데이터만 남겼습니다.
처리 방식·소요 시간·OCR 횟수 같은 운영 정보는 응답 대신 서버 로그
(logs/app.log)에 남습니다.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedPage(BaseModel):
    """페이지 / 시트 / 슬라이드 / 구역 단위 텍스트. 청킹의 기본 단위입니다."""

    page: int
    label: str = Field(
        description="예: document, sheet:매출, slide:1, section:2, page:3 (ocr)"
    )
    text: str


class FileMeta(BaseModel):
    """청크에 함께 붙일 소스 문서 식별 정보."""

    filename: str
    extension: str
    category: str = Field(description="pdf | image | office | hangul | text")


class TextExtractionResponse(BaseModel):
    """POST /extract/text 의 성공 응답."""

    file: FileMeta
    pages: list[ExtractedPage] = []


class ErrorResponse(BaseModel):
    """실패 응답 공통 포맷."""

    success: bool = False
    request_id: str | None = None
    error_code: str
    message: str
    detail: str | None = None
    created_at: str | None = None


class FormatInfo(BaseModel):
    category: str
    extensions: list[str]
    handler: str = Field(description="doc | pdf | image")
    parser: str
    requires_ocr: bool = Field(
        description="true 면 PP-OCRv5 가 설치되어 있어야 처리할 수 있습니다"
    )
    description: str


class SupportedFormatsResponse(BaseModel):
    max_upload_mb: int
    ocr_available: bool = Field(description="PP-OCRv5(PaddleOCR) 설치 여부")
    formats: list[FormatInfo]


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    env: str
    ocr_available: bool = Field(description="PP-OCRv5(PaddleOCR) 설치 여부")
    ocr_det_model: str
    ocr_rec_model: str
    ocr_error: str | None = Field(
        default=None,
        description="ocr_available 이 false 인 이유. 미설치면 ModuleNotFoundError, "
                    "DLL 충돌이면 OSError 가 나옵니다",
    )

    # --- 분류 파이프라인 ---
    debug: bool = Field(default=False, description="true 면 분류 응답에 단계별 중간 결과가 포함됩니다")
    device: str = Field(default="cpu", description="모델이 올라가는 디바이스 (cuda / cpu)")
    embed_model: str = Field(default="", description="③ 임베딩 모델")
    embed_loaded: bool = Field(default=False, description="③ 모델이 메모리에 올라와 있는지")
    llm_model: str = Field(default="", description="⑤ 판정 모델")
    llm_loaded: bool = Field(default=False, description="⑤ 모델이 메모리에 올라와 있는지")
    llm_adapter: str | None = Field(default=None, description="적용된 QLoRA 어댑터 경로")

# =============================================================================
# 분류 파이프라인 (③ 임베딩 -> ④ 후보 추림 -> ⑤ Qwen 판정)
# =============================================================================
class AnalyzeTextRequest(BaseModel):
    """POST /analyze/text 요청 본문. 파일 없이 문장만으로 ②~⑥ 을 테스트합니다."""

    text: str = Field(
        min_length=1,
        description="분류할 민원 문장 또는 문서 원문",
        examples=["우리 동네 가로등이 꺼졌어요. 밤에 골목이 너무 어두워서 위험합니다."],
    )
    user_id: str = Field(
        default="",
        description="로그인한 사용자 취급할 식별자. 비우면 서버 기본값(COMPLAINT_DEFAULT_USER)을 씁니다. "
                    "조회·수정·취소는 같은 user_id 로 등록한 민원만 볼 수 있습니다.",
    )


class KeywordItem(BaseModel):
    keyword: str
    score: float = Field(description="키워드 점수 (KeyBERT 코사인 유사도)")


class CandidateItem(BaseModel):
    rank: int
    code: str
    name: str
    score: float = Field(description="질의 벡터와 카테고리 정의문 벡터의 코사인 유사도")


class CaseHitItem(BaseModel):
    rank: int
    text: str = Field(description="라벨링된 사례 원문")
    category: str = Field(description="사례의 카테고리 라벨")
    score: float = Field(description="질의 벡터와 사례 벡터의 코사인 유사도")


class CaseLookupDebug(BaseModel):
    """④ 보조 벡터DB 조회 결과. ④ 가 low_confidence 일 때만 채워집니다."""

    used: bool = Field(description="사례 점수를 후보 순서에 실제로 반영했는지")
    reason: str = Field(description="조회한 이유 / 반영하지 않은 이유")
    hits: list[CaseHitItem] = Field(description="CASE_MIN_SCORE 이상인 유사 사례 (최대 CASE_TOP_K)")
    case_scores: dict[str, float] = Field(description="카테고리별 사례 점수 (유사도 합 / 사례 수)")
    candidates_before: list[CandidateItem] = Field(description="재정렬 전 ④ 상위 3개")
    candidates_after: list[CandidateItem] = Field(description="재정렬 후 상위 3개 (⑤ 에 전달)")


class ChoiceDetail(BaseModel):
    """⑤ 의 '번호 토큰 1회 계산' 결과."""

    number: int = Field(description="모델이 고른 번호 (1-based)")
    label: str
    score: float = Field(description="softmax 확신도 0~1")
    scores: dict[str, float] = Field(description="후보 전체 점수")


class ToolCallResult(BaseModel):
    called: bool = Field(description="도구를 호출했는지 여부. '문의' 의도면 false")
    name: str | None = None
    arguments: dict | None = None
    parse_error: str | None = None


class ToolExecutionResultSchema(BaseModel):
    """⑥ 민원 DB 실행 결과. tool_call.called 가 true 일 때만 채워집니다."""

    executed: bool = Field(description="DB 에 실제로 반영됐는지 (검증 실패면 false)")
    ok: bool = Field(description="검증을 통과했는지. false 면 error 를 보세요")
    message: str = Field(description="사람이 읽는 결과 문장 (그림의 ⑦ 사용자 응답)")
    data: dict | list | None = Field(default=None, description="구조화된 결과 (민원 1건 또는 목록)")
    error: str | None = None
    needs_selection: bool = Field(
        default=False,
        description=(
            "수정·삭제 대상이 여러 건이거나 특정되지 않아 실행하지 않았을 때 true. "
            "data 의 후보 목록을 보여 주고 사용자가 하나를 고르게 하세요"
        ),
    )
    search: dict | None = Field(
        default=None,
        description="조회·수정·삭제에서 대상 민원을 찾은 조건 (번호·카테고리·키워드·기간, 조건을 풀었는지)",
    )
    warnings: list[str] = Field(default_factory=list)


class PipelineDebug(BaseModel):
    """DEBUG=true 일 때만 채워집니다."""

    source_text: str = Field(description="② 추출 원문")
    sentence_count: int
    keywords: list[KeywordItem] = Field(description="②→③ 키워드 3개")
    summary: str = Field(description="②→③ 요약문")
    keyphrase_method: str = Field(description="keybert / frequency / none")
    embed_query_text: str = Field(description="③ 임베딩 모델에 실제로 넘긴 질의문")
    embedding_dim: int = Field(description="bge-m3 벡터 차원 (1024)")
    embedding_preview: list[float] = Field(description="벡터 앞부분 미리보기")
    candidates_top: list[CandidateItem] = Field(
        description="⑤ 에 전달한 상위 3개 (벡터DB 로 재정렬했으면 재정렬 후 점수)"
    )
    candidates_all: list[CandidateItem] = Field(
        description="7종 전체 점수 (벡터DB 로 재정렬했으면 재정렬 후 점수)"
    )
    candidate_margin: float = Field(description="1위와 2위 점수 차이")
    low_confidence: bool = Field(
        description="④ 1위 카테고리 유사도가 CANDIDATE_MIN_SCORE 미만인지. 참고용이며 반려 기준이 아닙니다"
    )
    case_lookup: CaseLookupDebug | None = Field(
        default=None, description="벡터DB 를 조회했을 때만 채워집니다 (CASE_MODE 에 따라 애매할 때만 또는 매번)"
    )
    intent_choice: ChoiceDetail = Field(description="⑤ 의도 판정(6지선다) 점수. 반려 시에도 채워집니다")
    category_choice: ChoiceDetail | None = Field(
        default=None, description="반려되었거나(GATE_ENABLED=false 관찰 모드 포함) 생략되면 null"
    )
    llm_raw_output: str = Field(default="", description="⑤ 모델 원시 출력 (반려 시 빈 문자열)")
    timings_ms: dict[str, int]
    warnings: list[str] = []
    debug_text: str = Field(description="위 내용을 한 덩어리 텍스트로 정리한 것")


class GateResult(BaseModel):
    """
    게이트 판정. ⑤ 의 의도 판정(6지선다) 기준입니다.

    문의·접수·조회·수정·삭제 중 하나로 판정되면 통과, "해당없음"으로 판정되면 반려입니다.
    카테고리(주제)와는 무관합니다.
    """

    passed: bool = Field(description="true 면 정상 처리, false 면 반려")
    intent_score: float = Field(description="선택된 의도(통과든 해당없음이든)의 확신도")
    enabled: bool = Field(description="GATE_ENABLED. false 면 관찰 모드(차단하지 않음)")
    reason: str = Field(default="", description="반려(또는 관찰 모드 통과) 사유")


class AnalyzeResponse(BaseModel):
    """POST /analyze/* 의 성공 응답. 게이트 반려도 200 으로 돌려줍니다."""

    status: str = Field(
        description="accepted = ⑤ 판정 완료 / rejected = ⑤ 의도 판정이 '해당없음'이라 반려"
    )
    file: FileMeta | None = Field(
        default=None, description="파일 업로드로 들어온 경우의 소스 문서 정보"
    )
    result_text: str = Field(
        description="판정(또는 반려) 결과를 사람이 읽는 텍스트로 정리한 것 (Swagger 확인용)"
    )
    gate: GateResult
    intent: str | None = Field(
        default=None,
        description="판정된 의도. 문의/접수/조회/수정/삭제 중 하나, 또는 반려된 경우 '해당없음'",
    )
    intent_code: str | None = None
    intent_score: float | None = None
    category: str | None = Field(
        default=None, description=(
            "확정된 카테고리 (7종 중 1개). 반려·해당없음·문의이거나, 조회·수정·삭제에서 "
            "주제를 특정할 수 없어 '없음'으로 판정되면 null"
        )
    )
    category_code: str | None = None
    category_score: float | None = None
    department: str | None = Field(default=None, description="확정 카테고리의 담당 부서")
    tool_call: ToolCallResult | None = Field(
        default=None, description="반려되었거나 의도가 '해당없음'이면 null"
    )
    tool_result: ToolExecutionResultSchema | None = Field(
        default=None, description="⑥ 민원 DB 실행 결과. tool_call.called 가 false 면 null"
    )
    answer: str = Field(
        default="", description="'문의' 의도일 때의 즉답, 또는 반려 시 사용자 안내 문구"
    )
    debug: PipelineDebug | None = Field(
        default=None, description="DEBUG=true 일 때만 채워집니다"
    )


class CategoryInfo(BaseModel):
    code: str
    name: str
    department: str
    summary: str
    keywords: list[str]


class CategoriesResponse(BaseModel):
    count: int
    categories: list[CategoryInfo]


class PipelineStatusResponse(BaseModel):
    """모델 설치·로드 상태. 첫 요청 전에 환경을 확인할 때 사용합니다."""

    debug: bool
    device: str
    runtime: dict
    keyphrase: dict
    embedder: dict
    case_store: dict
    complaint_store: dict
    llm: dict
