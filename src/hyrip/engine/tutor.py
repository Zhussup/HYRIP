"""Tutor mode: immediate per-item grading and streamed, pointed feedback."""

from __future__ import annotations

from collections.abc import Callable, Iterator

from ..llm.client import LLMClient
from ..llm.prompts import build_tutor_feedback_messages, build_tutor_followup_messages
from ..models.grading import GradedItem, Tally, UserAnswer
from ..models.questions import OPEN_TYPES, CompletionItem, QuestionPaper
from .grader import grade_closed, grade_one_open, heading_bank_of

TUTOR_MAX_TOKENS = 600

SingleMarker = Callable[..., object]


def grade_immediately(
    paper: QuestionPaper,
    article,
    question,
    answer: UserAnswer,
    *,
    mode: str,
    single_marker: SingleMarker | None = None,
) -> GradedItem:
    """Grading at submit time (tutor mode): deterministic for closed, single LLM mark for open."""
    if question.question_type in OPEN_TYPES:
        assert isinstance(question, CompletionItem)
        raw = answer.raw
        skipped = answer.skipped or raw is None or not raw.strip()
        mark = None
        if single_marker is not None and not skipped:
            try:
                mark = single_marker(question, raw or "")
            except Exception:
                mark = None
        return grade_one_open(question, raw, skipped=skipped, mode=mode, mark=mark, article=article)
    return grade_closed(
        question,
        answer.raw,
        skipped=answer.skipped,
        headings_bank=heading_bank_of(paper, question.group_no),
    )


def feedback_stream(
    client: LLMClient,
    article,
    question,
    graded: GradedItem,
    tally: Tally,
    history: list[dict] | None = None,
) -> Iterator[str]:
    """First tutor comment on an answer, streamed."""
    messages = build_tutor_feedback_messages(article, question, graded, tally, history)
    yield from client.chat_stream(messages, max_tokens=TUTOR_MAX_TOKENS)


def followup_stream(client: LLMClient, article, question, history: list[dict]) -> Iterator[str]:
    """A user follow-up ("why is C wrong?") within the current question's thread."""
    messages = build_tutor_followup_messages(article, question, history)
    yield from client.chat_stream(messages, max_tokens=TUTOR_MAX_TOKENS)
