"""
로깅 설정.

계획서의 '작업로그관리' 항목(시간기록 / 작업상태 / 주요이벤트)에 맞춰
콘솔 + 파일(일자별 로테이션) 두 곳에 동일한 포맷으로 남깁니다.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler

from app.config import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL.upper())

    # uvicorn --reload 시 핸들러가 중복 등록되는 것을 방지
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if settings.LOG_TO_FILE:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            filename=settings.log_dir / "app.log",
            when="midnight",
            backupCount=14,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # 외부 라이브러리 로그 소음 줄이기
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("pdfminer").setLevel(logging.ERROR)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
