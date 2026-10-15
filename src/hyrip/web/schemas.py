"""Pydantic request DTOs for the web API. Responses reuse engine models directly."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ArticleIn(BaseModel):
    source: str  # "text" | "url"
    text: str | None = None
    url: str | None = None
    title: str | None = None

    @model_validator(mode="after")
    def _payload_matches_source(self) -> ArticleIn:
        if self.source == "text" and not (self.text or "").strip():
            raise ValueError("paste some article text for source=text")
        if self.source == "url" and not (self.url or "").strip():
            raise ValueError("provide url for source=url")
        if self.source not in ("text", "url"):
            raise ValueError("source must be 'text' or 'url'")
        return self


class SessionIn(BaseModel):
    article_id: int
    mode: str = "exam"
    model: str | None = None
    n_questions: int | None = Field(default=None, ge=8, le=16)
    time_budget_min: int | None = Field(default=None, ge=1, le=90)
    claims: bool = False


class AnswerIn(BaseModel):
    number: int = Field(ge=1)
    raw: str | None = None
    skipped: bool = False


class FollowupIn(BaseModel):
    number: int = Field(ge=1)
    text: str = Field(min_length=1)
