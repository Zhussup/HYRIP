"""Where articles come from and what they carry."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ArticleSource(StrEnum):
    PASTE = "paste"
    FILE = "file"
    URL = "url"


def now_utc() -> datetime:
    return datetime.now(UTC)


class Article(BaseModel):
    """A normalized passage: paragraph list with IELTS-style A, B, C… labels."""

    id: int | None = None
    title: str = "Untitled article"
    source: ArticleSource
    source_url: str | None = None
    author: str | None = None
    published: str | None = None
    text: str
    paragraphs: list[str] = Field(default_factory=list)
    paragraph_labels: list[str] = Field(default_factory=list)
    word_count: int = 0
    fetched_at: datetime = Field(default_factory=now_utc)

    def paragraph_for(self, label: str) -> str | None:
        if label in self.paragraph_labels:
            return self.paragraphs[self.paragraph_labels.index(label)]
        return None

    def preview(self, limit: int = 160) -> str:
        return self.text[:limit].replace("\n", " ") + ("…" if len(self.text) > limit else "")
