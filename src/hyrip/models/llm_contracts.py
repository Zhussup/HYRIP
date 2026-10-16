"""Pydantic contracts for JSON the LLM must return. extra="forbid" so drift fails loudly."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .questions import QuestionType


class RawEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paragraph: str = ""
    quote: str = Field(min_length=1)


class RawOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    trap_kind: Literal["narrow", "broad", "unsupported", None] = None


class RawItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: int = Field(ge=1)
    prompt: str = Field(min_length=1)
    paragraph_label: str | None = (
        None  # matching_headings / match_paragraphs: the paragraph in play
    )
    answer_key: str = Field(min_length=1)
    evidence: list[RawEvidence] = Field(default_factory=list)
    explanation: str = ""
    acceptable: list[str] | None = None  # completion / short_answer only
    word_limit: int | None = None
    allows_numbers: bool | None = None
    options: list[RawOption] | None = None  # summary_choice only


class RawGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_no: int | None = None  # accepted and ignored; build_paper numbers groups by position
    question_type: QuestionType
    instructions: str = Field(min_length=1)
    headings: list[str] | None = None  # required for matching_headings
    options: list[str] | None = None  # required for multiple_choice (shared 4-option bank)
    items: list[RawItem] = Field(min_length=1)


class RawGeneratedPaper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passage_title: str = ""  # optional; falls back to the article title
    # n=14/16 recipes carry 7 groups (short_answer joins at 14), so the cap is 7, not 6.
    groups: list[RawGroup] = Field(min_length=2, max_length=7)


class MarkResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: int
    verdict: Literal["correct", "partial", "incorrect"]
    issues: list[str] = Field(default_factory=list, max_length=2)
    why: str = ""
    evidence: RawEvidence | None = None
    teaching_note: str = ""


class VerdictBatch(BaseModel):
    """Marker output for completion/short-answer items."""

    model_config = ConfigDict(extra="forbid")

    results: list[MarkResult] = Field(default_factory=list)


class CoachObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    title: str
    detail: str
    question_numbers: list[int] = Field(default_factory=list)


class CoachReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observations: list[CoachObservation] = Field(default_factory=list)
