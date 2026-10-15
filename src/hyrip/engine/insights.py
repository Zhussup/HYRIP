"""Deterministic reading observations from graded items, plus the optional coach call.

Every observation cites the question numbers it concerns; coach output is appended AFTER
the deterministic ones and is allowed to fail silently (a report must never depend on it).
"""

from __future__ import annotations

from ..llm.client import LLMClient
from ..llm.prompts import build_coach_messages
from ..models.grading import GradedItem, QuestionReview, ReadingObservation
from ..models.llm_contracts import CoachReport
from ..models.questions import QuestionPaper, SummaryChoiceQuestion
from .generator import parse_json_response

DETAIL_TYPES = {"completion", "short_answer", "multiple_choice"}
SCAN_TYPES = {"true_false_not_given", "yes_no_not_given", "matching_headings", "match_paragraphs"}


def build_observations(
    paper: QuestionPaper, graded: dict[int, GradedItem]
) -> list[ReadingObservation]:
    """The always-present rules; cap 4, strongest signal first."""
    observations: list[ReadingObservation] = []
    {q.number for q in paper.questions}

    # NOT GIVEN: "the passage doesn't say" read as "the passage says no"
    ng_numbers = sorted(
        g.number
        for g in graded.values()
        if g.question_type in ("true_false_not_given", "yes_no_not_given")
        and g.verdict == "incorrect"
        and "GIVEN" in (paper_item(paper, g.number).answer_key.upper())
    )
    if len(ng_numbers) >= 2:
        observations.append(
            ReadingObservation(
                kind="not_given_confusion",
                title="NOT GIVEN confusion",
                detail=(
                    "You treated statements the passage is silent on as false or true."
                    " Before rejecting a claim, look for BOTH a confirmation and a contradiction"
                    " — neither means Not Given, however obvious the real world answer is."
                ),
                question_numbers=ng_numbers,
            )
        )

    # paraphrase traps: wrong MC picks usually mean the stem's paraphrase was missed
    mc_numbers = sorted(
        g.number
        for g in graded.values()
        if g.question_type == "multiple_choice" and g.verdict == "incorrect"
    )
    if len(mc_numbers) >= 2:
        observations.append(
            ReadingObservation(
                kind="paraphrase_trap",
                title="Paraphrase traps",
                detail=(
                    "Wrong choices answered a different detail than the stem asked."
                    " Underline the stem's exact question, then locate the sentence whose words"
                    " paraphrase it — every distractor below it is bait."
                ),
                question_numbers=mc_numbers,
            )
        )

    # main idea
    summary_number = paper.main_idea_number
    summary_graded = graded.get(summary_number)
    if summary_graded is not None and summary_graded.verdict != "skipped":
        kind, title, detail = _main_idea_note(paper, summary_graded)
        observations.append(
            ReadingObservation(
                kind=kind,
                title=title,
                detail=detail,
                question_numbers=[summary_number],
            )
        )

    # skimming vs detail
    skim = _split_note(paper, graded)
    if skim is not None:
        observations.append(skim)

    # word-limit discipline
    limit_numbers = sorted(
        g.number for g in graded.values() if g.verdict != "skipped" and g.word_count_ok is False
    )
    if limit_numbers:
        observations.append(
            ReadingObservation(
                kind="word_limit_slips",
                title="Word-limit slips",
                detail=(
                    "Answers over the stated word limit are marked wrong in the real test "
                    "regardless of meaning. Count the words against the instruction once before "
                    "writing."
                ),
                question_numbers=limit_numbers,
            )
        )

    return observations[:4]


def _main_idea_note(paper: QuestionPaper, graded: GradedItem) -> tuple[str, str, str]:
    question = paper_item(paper, graded.number)
    assert isinstance(question, SummaryChoiceQuestion)
    if graded.verdict == "correct":
        return (
            "main_idea",
            "Main idea captured",
            "You picked the summary that covers the whole passage, not one paragraph's point"
            " — that is the conscious-reading this app exists to check. Keep doing it.",
        )
    chosen = _chosen_trap(question, graded.user_raw or "")
    if chosen == "narrow":
        return (
            "main_idea",
            "Main idea missed: too narrow",
            "Your pick was one paragraph's point wearing a whole-article costume. Ask of each"
            " summary: could I delete three paragraphs and this still hold? Could = too narrow.",
        )
    if chosen in ("broad", "unsupported"):
        return (
            "main_idea",
            "Main idea missed: too broad",
            "Your pick went beyond what the passage supports or swallowed more than its topic."
            " The best summary is the one only this passage could produce.",
        )
    return (
        "main_idea",
        "Main idea missed",
        "The best summary covers every paragraph without adding anything. Compare options"
        " against the passage's first and last paragraph together: the frame, not a detail.",
    )


def _chosen_trap(question: SummaryChoiceQuestion, raw: str) -> str | None:
    from .grader import canonical_letter

    letter = canonical_letter([option.text for option in question.options], raw)
    if letter is None:
        return None
    index = ord(letter) - ord("A")
    if 0 <= index < len(question.options):
        return question.options[index].trap_kind
    return None


def _split_note(paper: QuestionPaper, graded: dict[int, GradedItem]) -> ReadingObservation | None:
    detail = [g for g in graded.values() if g.question_type in DETAIL_TYPES]
    scan = [g for g in graded.values() if g.question_type in SCAN_TYPES]
    detail_pct = _accuracy(detail)
    scan_pct = _accuracy(scan)
    if None in (detail_pct, scan_pct) or len(detail) < 2 or len(scan) < 2:
        return None
    gap = detail_pct - scan_pct
    if abs(gap) < 30.0:
        return None
    if gap > 0:
        return ReadingObservation(
            kind="skimming_vs_detail",
            title="Detail better than quick checking",
            detail=(
                f"Careful reading served you ({_fmt(detail_pct)} on detail questions) more than "
                f"fast scanning ({_fmt(scan_pct)}). Practise locating question keywords in the "
                "passage fast rather than re-reading."
            ),
            question_numbers=sorted(g.number for g in detail if g.verdict == "incorrect"),
        )
    return ReadingObservation(
        kind="skimming_vs_detail",
        title="Quick checking better than detail",
        detail=(
            f"Fast items went better ({_fmt(scan_pct)}) than detail work ({_fmt(detail_pct)}). "
            "Slow down on completion and multiple choice: read the whole sentence, not just "
            "the gap."
        ),
        question_numbers=sorted(g.number for g in detail if g.verdict == "incorrect"),
    )


def paper_item(paper: QuestionPaper, number: int):
    for question in paper.questions:
        if question.number == number:
            return question
    raise KeyError(number)


def _accuracy(items: list[GradedItem]) -> float | None:
    marked = [g for g in items if g.verdict != "skipped"]
    if not marked:
        return None
    return 100.0 * sum(1 for g in marked if g.verdict == "correct") / len(marked)


def _fmt(pct: float) -> str:
    return f"{pct:.0f}%"


_FOCUS_ACTIONS = {
    "not_given_confusion": (
        "Practise the two-question test: is it stated? and is it contradicted? — silence means"
        " NOT GIVEN."
    ),
    "paraphrase_trap": (
        "Before reading options, rewrite the stem in your own words and hunt its paraphrase in"
        " the passage."
    ),
    "main_idea": (
        "After the first read, state the passage's main idea in one sentence of five words, then"
        " compare it with each summary option."
    ),
    "skimming_vs_detail": (
        "Time-box question types: scan for which-paragraph and headings, slow-read for completion"
        " and multiple choice."
    ),
    "word_limit_slips": (
        "Write the answer, then count words against the limit; trim articles before exceeding it."
    ),
}


def next_focus(observations: list[ReadingObservation], limit: int = 3) -> list[str]:
    return [
        _FOCUS_ACTIONS[observation.kind]
        for observation in observations
        if observation.kind in _FOCUS_ACTIONS
    ][:limit]


def coach_observations(
    client: LLMClient,
    article,
    reviews: list[QuestionReview],
    deterministic: list[ReadingObservation],
) -> list[ReadingObservation]:
    """One optional coach call; returns [] on any failure — never let it break a report."""
    try:
        messages = build_coach_messages(article, reviews, deterministic)
        result = client.chat(messages, json_mode=True, temperature=0.5)
        report = CoachReport.model_validate_json(parse_json_response(result.content))
        return [
            ReadingObservation(
                kind=observation.kind,
                title=observation.title,
                detail=observation.detail,
                question_numbers=[n for n in observation.question_numbers if isinstance(n, int)],
            )
            for observation in report.observations
        ][:4]
    except Exception:
        return []
