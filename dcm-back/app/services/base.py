"""추출기들이 공통으로 사용하는 결과 자료구조."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Segment:
    """페이지 / 시트 / 슬라이드 / 구역 단위 텍스트 조각."""

    index: int
    label: str
    text: str


@dataclass
class ExtractedText:
    segments: list[Segment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # 추출 과정 통계 (예: {"ocr_pages": 2, "ocr_images": 5})
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """전체 세그먼트를 이어붙인 텍스트."""
        return "\n\n".join(seg.text for seg in self.segments if seg.text).strip()

    @property
    def char_count(self) -> int:
        return sum(len(seg.text) for seg in self.segments)
