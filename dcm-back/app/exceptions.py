"""애플리케이션 공통 예외."""

from __future__ import annotations


class AppError(Exception):
    """모든 커스텀 예외의 부모. HTTP 상태코드와 에러코드를 함께 가집니다."""

    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class UnsupportedFormatError(AppError):
    status_code = 415
    error_code = "UNSUPPORTED_FORMAT"


class FileTooLargeError(AppError):
    status_code = 413
    error_code = "FILE_TOO_LARGE"


class EmptyFileError(AppError):
    status_code = 400
    error_code = "EMPTY_FILE"


class ExtractionError(AppError):
    """파일 손상, 암호 설정 등으로 파싱에 실패한 경우."""

    status_code = 422
    error_code = "EXTRACTION_FAILED"


class OcrUnavailableError(AppError):
    """PP-OCRv5(PaddleOCR)가 설치되어 있지 않거나 초기화에 실패한 경우."""

    status_code = 503
    error_code = "OCR_UNAVAILABLE"


class ModelUnavailableError(AppError):
    """
    분류 파이프라인 모델(bge-m3 / Qwen)을 사용할 수 없는 경우.

    라이브러리 미설치, 가중치 다운로드 실패, GPU 메모리 부족 등이 해당합니다.
    """

    status_code = 503
    error_code = "MODEL_UNAVAILABLE"


class NoTextError(AppError):
    """추출된 텍스트가 비어 있어 분류를 진행할 수 없는 경우."""

    status_code = 422
    error_code = "NO_TEXT"
