"""Answer and verdict models shared by every frontend."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .questions import OPEN_TYPES, Evidence

Verdict = Literal["correct", "partial", "incorrect", "skipped"]
MarkedBy = Literal["deterministic", "llm", "fallback"]

MODES = ("exam", "tutor")
Mode = Literal["exam", "tutor"]

SESSION_STATUSES = (
    "generating",
    "reading",
    "active",
    "grading",
    "report",
    "abandoned",
    "failed",
)
SessionStatus = Literal[
    "generating",
    "reading",
    "active",
    "grading",
    "report",
    "abandoned",
    "failed",
]


def is_open_type(question_type: str) -> bool:
    return question_type in OPEN_TYPES


class UserAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: int
    raw: str | None = None  # None or empty ⇒ skipped (recorded explicitly)
    skipped: bool = False


class GradedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: int
    question_type: str
    verdict: Verdict
    user_raw: str | None = None
    word_count_ok: bool | None = None  # open answers only; None = not applicable
    issues: list[str] = Field(default_factory=list)
    why: str = ""
    teaching_note: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    graded_by: MarkedBy = "deterministic"


class QuestionReview(GradedItem):
    """GradedItem joined with question content, for report rendering."""

    prompt: str = ""
    answer_key: str = ""
    explanation: str = ""


class ReadingObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    title: str
    detail: str
    question_numbers: list[int] = Field(default_factory=list)


class BandEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_correct: int
    total: int
    pct: float
    band: float
    disclaimer: str = (
        "Unofficial estimate from a single passage; the real raw-to-band"
        " conversion varies by test date."
    )


class SessionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: BandEstimate
    elapsed_seconds: int
    reviews: list[QuestionReview] = Field(default_factory=list)
    observations: list[ReadingObservation] = Field(default_factory=list)
    next_focus: list[str] = Field(default_factory=list)


class Tally(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correct: int
    answered: int
    skipped: int
    total: int
