"""Report assembly: score, band estimate, per-question review, observations, next focus."""

from __future__ import annotations

from ..models.grading import (
    BandEstimate,
    GradedItem,
    QuestionReview,
    SessionReport,
    Tally,
)
from ..models.questions import QuestionPaper
from . import insights

# Threshold at or above each band (raw % on one passage). Below the last threshold: 3.5.
BAND_TABLE: tuple[tuple[float, float], ...] = (
    (97.5, 9.0),
    (87.5, 8.0),
    (82.5, 7.5),
    (75.0, 7.0),
    (67.5, 6.5),
    (57.5, 6.0),
    (47.5, 5.5),
    (37.5, 5.0),
    (32.5, 4.5),
    (25.0, 4.0),
)
FLOOR_BAND = 3.5


def band_for_pct(pct: float) -> float:
    for threshold, band in BAND_TABLE:
        if pct >= threshold:
            return band
    return FLOOR_BAND


def tally_of(graded: dict[int, GradedItem], total: int) -> Tally:
    items = list(graded.values())
    answered = sum(1 for g in items if g.verdict != "skipped")
    skipped = sum(1 for g in items if g.verdict == "skipped")
    correct = sum(1 for g in items if g.verdict == "correct")
    return Tally(correct=correct, answered=answered, skipped=skipped, total=total)


def build_report(
    paper: QuestionPaper,
    graded: dict[int, GradedItem],
    *,
    elapsed_seconds: int,
) -> SessionReport:
    """Assemble the end-of-session report. No LLM here — the coach merges in before render
    if the service chose to run it."""
    total = len(paper.questions)
    score = tally_of(graded, total)
    pct = 100.0 * score.correct / total if total else 0.0
    band = band_for_pct(pct)
    reviews = [
        _review(paper, question.number, graded.get(question.number)) for question in paper.questions
    ]
    deterministic = insights.build_observations(paper, graded)
    return SessionReport(
        score=BandEstimate(
            raw_correct=score.correct,
            total=total,
            pct=round(pct, 1),
            band=band,
        ),
        elapsed_seconds=elapsed_seconds,
        reviews=reviews,
        observations=deterministic,
        next_focus=insights.next_focus(deterministic),
    )


def _review(paper: QuestionPaper, number: int, g: GradedItem | None) -> QuestionReview:
    question = insights.paper_item(paper, number)
    return QuestionReview(
        number=question.number,
        question_type=question.question_type,
        verdict=g.verdict if g else "skipped",
        user_raw=g.user_raw if g else None,
        word_count_ok=g.word_count_ok if g else None,
        issues=(g.issues if g else []),
        why=(g.why or question.explanation) if g else question.explanation,
        teaching_note=(g.teaching_note if g else "") or "",
        evidence=(list(g.evidence) or list(question.evidence)) if g else list(question.evidence),
        graded_by=(g.graded_by if g else "deterministic"),
        prompt=question.prompt,
        answer_key=question.answer_key,
        explanation=question.explanation,
    )


__all__ = ["BAND_TABLE", "band_for_pct", "build_report", "tally_of"]
