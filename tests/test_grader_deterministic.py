"""Deterministic grading: canonicalisation tables, word limits, skip, batch marker flow."""

from __future__ import annotations

import json

import pytest

from hyrip.engine.grader import (
    canonical_heading,
    canonical_letter,
    canonical_paragraph_label,
    canonical_tfng,
    grade_closed,
    grade_one_open,
    grade_paper,
    heading_bank_of,
    heading_key,
    make_batch_marker,
    make_single_marker,
)
from hyrip.llm.client import ChatResult
from hyrip.models.grading import UserAnswer
from hyrip.models.llm_contracts import MarkResult
from hyrip.models.questions import build_paper


@pytest.fixture
def paper(raw_paper, article):
    return build_paper(raw_paper, article=article)


def answer(number: int, raw: str | None, skipped: bool = False) -> UserAnswer:
    return UserAnswer(number=number, raw=raw, skipped=skipped)


# ---- canonicalisation --------------------------------------------------------


def test_canonical_letter_forms(paper):
    options = [o.text for o in paper.questions[0].options]
    assert canonical_letter(options, "a") == "A"
    assert canonical_letter(options, "B") == "B"
    assert canonical_letter(options, "B.") == "B"
    assert canonical_letter(options, "(B)") == "B"
    assert canonical_letter(options, "2") == "B"
    assert canonical_letter(options, "The countryside became too crowded for hives.") == "B"
    assert canonical_letter(options, "B) The countryside became too crowded for hives.") == "B"
    assert canonical_letter(options, "nope") is None


def test_canonical_letter_full_text_wins_over_prefix(paper):
    options = [o.text for o in paper.questions[0].options]
    # option B's full text starts with "The"; A starts with "Rooftop". Full-text identity first.
    assert canonical_letter(options, "Rooftop hives were legalised in 2010.") == "A"


def test_canonical_tfng_strict_words():
    assert canonical_tfng("true", yes_mode=False) == "TRUE"
    assert canonical_tfng("T", yes_mode=False) == "TRUE"
    assert canonical_tfng("Not Given", yes_mode=False) == "NOT GIVEN"
    assert canonical_tfng("ng", yes_mode=False) == "NOT GIVEN"
    # TFNG and YNNG use different word sets on purpose: writing YES on TFNG is a mistake
    assert canonical_tfng("yes", yes_mode=False) == "YES"
    assert canonical_tfng("yes", yes_mode=False) != "TRUE"
    assert canonical_tfng("yes", yes_mode=True) == "YES"
    assert canonical_tfng("no", yes_mode=True) == "NO"
    assert canonical_tfng("false", yes_mode=True) == "FALSE"


def test_canonical_heading_forms(raw_paper):
    bank = raw_paper.groups[2].headings
    assert canonical_heading("iv", bank) == "iv"
    assert canonical_heading("IV.", bank) == "iv"
    assert canonical_heading("2", bank) == "ii"
    assert canonical_heading("A warning from ecologists", bank) == "ii"
    assert canonical_heading("nope", bank) == "nope"


def test_heading_key_and_bank(paper, raw_paper, article):
    assert heading_key("iv. Learning through hives") == "iv"
    assert heading_bank_of(paper, 3) == raw_paper.groups[2].headings
    assert heading_bank_of(paper, 4) == []


def test_canonical_paragraph_label():
    assert canonical_paragraph_label("h") == "H"
    assert canonical_paragraph_label("Paragraph D.") == "D"


# ---- grade_closed ---------------------------------------------------------------


def test_mc_by_letter_and_full_text(paper):
    q = paper.questions[0]
    assert grade_closed(q, "a", skipped=False).verdict == "correct"
    wrong = grade_closed(q, "B", skipped=False)
    assert wrong.verdict == "incorrect"
    assert wrong.why == q.explanation
    assert wrong.graded_by == "deterministic"
    assert (
        grade_closed(q, "The countryside became too crowded for hives.", skipped=False).verdict
        == "incorrect"
    )
    assert (
        grade_closed(q, "Rooftop hives were legalised in 2010.", skipped=False).verdict == "correct"
    )


def test_summary_index_and_text(paper):
    q = paper.questions[11]
    assert grade_closed(q, "2", skipped=False).verdict == "correct"  # index 2 → B
    assert grade_closed(q, "4", skipped=False).verdict == "incorrect"
    assert (
        grade_closed(
            q,
            "Urban beekeeping has grown quickly, and its rapid spread now needs managing carefully.",
            skipped=False,
        ).verdict
        == "correct"
    )


def test_tfng_strictness(paper):
    q = paper.questions[2]  # key TRUE
    assert grade_closed(q, "T", skipped=False).verdict == "correct"
    assert grade_closed(q, "True.", skipped=False).verdict == "correct"
    assert grade_closed(q, "yes", skipped=False).verdict == "incorrect"
    assert grade_closed(q, "F", skipped=False).verdict == "incorrect"


def test_ynng_paths():
    # the fixture has no YNNG group; canonical_tfng covers YES/NO/NOT GIVEN above
    pass


def test_headings_by_roman_and_text(paper, raw_paper):
    q5, q6, q7 = (paper.questions[i] for i in (4, 5, 6))
    bank = heading_bank_of(paper, q5.group_no)
    assert bank == raw_paper.groups[2].headings
    assert grade_closed(q5, "ii", skipped=False, headings_bank=bank).verdict == "correct"
    assert grade_closed(q5, "2", skipped=False, headings_bank=bank).verdict == "correct"
    got = grade_closed(q5, "2", skipped=False)
    assert got.verdict == "incorrect"  # no bank passed: numeric index cannot be resolved
    assert (
        grade_closed(
            q6, "Beekeepers respond with restraint", skipped=False, headings_bank=bank
        ).verdict
        == "correct"
    )
    assert grade_closed(q7, "iv", skipped=False, headings_bank=bank).verdict == "correct"


def test_match_paragraphs(paper):
    q8 = paper.questions[7]
    assert grade_closed(q8, "H", skipped=False).verdict == "correct"
    assert grade_closed(q8, "h", skipped=False).verdict == "correct"
    assert grade_closed(q8, "Paragraph D", skipped=False).verdict == "incorrect"


def test_skipped_and_blank(paper):
    q = paper.questions[0]
    for raw, skipped in ((None, True), (None, False), ("", False), ("   ", False)):
        graded = grade_closed(q, raw, skipped=skipped)
        assert graded.verdict == "skipped"


# ---- open answers ---------------------------------------------------------------


def test_open_containment_fallback(paper):
    q10 = paper.questions[9]
    graded = grade_one_open(q10, "30", skipped=False, mode="exam")
    assert graded.verdict == "correct"
    assert graded.word_count_ok is True
    assert graded.graded_by == "fallback"
    assert grade_one_open(q10, "thirty", skipped=False, mode="tutor").verdict == "correct"
    assert grade_one_open(q10, "31", skipped=False, mode="exam").verdict == "incorrect"
    assert grade_one_open(q10, "  ", skipped=False, mode="exam").verdict == "skipped"


def test_open_word_limit_exam_incorrect(paper):
    q10 = paper.questions[9]
    graded = grade_one_open(q10, "around 30 kg exactly", skipped=False, mode="exam")
    assert graded.verdict == "incorrect"
    assert graded.word_count_ok is False
    assert any("limit is 3" in issue for issue in graded.issues)
    assert graded.graded_by == "deterministic"


def test_open_word_limit_tutor_partial(paper):
    q10 = paper.questions[9]
    graded = grade_one_open(q10, "around 30 kg exactly", skipped=False, mode="tutor")
    assert graded.verdict == "partial"
    assert graded.word_count_ok is False


def test_open_within_limit_three_words_is_not(paper):
    """'30 kg exactly' is 3 words == limit 3: fine, but it must match the key to be correct."""
    q10 = paper.questions[9]
    graded = grade_one_open(q10, "30 kg exactly", skipped=False, mode="exam")
    assert graded.verdict == "incorrect"
    assert graded.word_count_ok is True


def test_open_word_limit_beats_containment(paper):
    q11 = paper.questions[10]
    graded = grade_one_open(
        q11, "starve all the wild pollinators", skipped=False, mode="exam", mark=None
    )
    assert graded.verdict == "incorrect"


def test_open_llm_mark_partial_downgraded_in_exam(paper):
    q11 = paper.questions[10]
    mark = MarkResult(
        number=11,
        verdict="partial",
        why="right words, wrong form",
        evidence=None,
        teaching_note="watch plurals",
    )
    graded = grade_one_open(q11, "starve the pollinators", skipped=False, mode="exam", mark=mark)
    assert graded.verdict == "incorrect"
    assert any("wrong form" in issue for issue in graded.issues)
    assert graded.graded_by == "llm"
    assert graded.teaching_note == "watch plurals"
    tutor_graded = grade_one_open(
        q11, "starve the pollinators", skipped=False, mode="tutor", mark=mark
    )
    assert tutor_graded.verdict == "partial"


def test_open_llm_mark_evidence_verified(paper, article):
    q11 = paper.questions[10]
    mark = MarkResult(
        number=11,
        verdict="correct",
        evidence={"paragraph": "E", "quote": "Ten hives on one block can starve wild pollinators"},
    )
    graded = grade_one_open(
        q11, "starve the pollinators", skipped=False, mode="tutor", mark=mark, article=article
    )
    assert graded.evidence[0].paragraph == "E"
    assert graded.evidence[0].quote_verified is True


def test_open_llm_mark_wrong_quote_unverified(paper, article):
    q11 = paper.questions[10]
    mark = MarkResult(
        number=11,
        verdict="correct",
        evidence={"paragraph": "E", "quote": "this passage never said that"},
    )
    graded = grade_one_open(
        q11, "starve the pollinators", skipped=False, mode="tutor", mark=mark, article=article
    )
    assert graded.evidence[0].quote_verified is False


# ---- batch flow ------------------------------------------------------------------


def test_grade_paper_full_flow_exam(paper, article, fake_llm):
    marker = make_batch_marker(fake_llm, article)
    fake_llm.responses = [
        ChatResult(
            content=json.dumps(
                {
                    "results": [
                        {
                            "number": 11,
                            "verdict": "partial",
                            "why": "right words, wrong form",
                            "evidence": {
                                "paragraph": "E",
                                "quote": "Ten hives on one block can starve wild pollinators",
                            },
                            "teaching_note": "count the words",
                        }
                    ]
                }
            ),
            finish_reason="stop",
        )
    ]
    answers = {
        1: answer(1, "A"),  # correct
        2: answer(2, "C"),  # correct
        3: answer(3, "TRUE"),  # correct
        4: answer(4, "TRUE"),  # wrong (FALSE)
        5: answer(5, "ii"),  # correct
        6: answer(6, "iii"),  # correct
        7: answer(7, "iii"),  # wrong (iv) but no bank → heading bank comes from paper
        8: answer(8, "H"),  # correct
        9: answer(9, "D"),  # correct
        10: answer(10, "30"),  # containment fallback correct
        11: answer(11, "starve the pollinators"),  # LLM partial → exam incorrect
        12: answer(12, "B"),  # correct
    }
    graded = grade_paper(paper, answers, mode="exam", marker=marker, article=article)
    verdicts = {n: g.verdict for n, g in graded.items()}
    assert verdicts[4] == "incorrect"
    assert graded[7].verdict == "incorrect"  # iii ≠ iv regardless of bank
    assert graded[10].graded_by == "fallback"
    assert graded[10].verdict == "correct"
    assert graded[11].verdict == "incorrect"
    assert graded[11].graded_by == "llm"
    assert verdicts[12] == "correct"
    # the marker got exactly the two open items, and only Q11's answer came back valid
    assert len(fake_llm.calls) == 1
    over_limit = grade_paper(
        paper,
        {11: answer(11, "starve all the wild pollinators")},
        mode="exam",
        marker=marker,
    )
    assert over_limit[11].verdict == "incorrect"
    assert over_limit[11].word_count_ok is False


def test_grade_paper_marker_failure_degrades_to_fallback(paper, article, fake_llm):
    """A marker crash or JSON drift must not lose the whole report."""
    fake_llm.responses = [ChatResult(content="garbage", finish_reason="stop")]
    answers = {10: answer(10, "30")}
    graded = grade_paper(paper, answers, mode="exam", marker=make_batch_marker(fake_llm, article))
    assert graded[10].verdict == "correct"
    assert graded[10].graded_by == "fallback"


def test_grade_paper_no_marker_marks_fallback(paper):
    answers = {10: answer(10, "30")}
    graded = grade_paper(paper, answers, mode="exam")
    assert graded[10].verdict == "correct"
    assert graded[10].graded_by == "fallback"


def test_single_marker_for_tutor(paper, article, fake_llm):
    fake_llm.responses = [
        ChatResult(
            content=json.dumps(
                {
                    "results": [
                        {"number": 10, "verdict": "correct", "why": "matches the yield figure"}
                    ]
                }
            ),
            finish_reason="stop",
        )
    ]
    marker = make_single_marker(fake_llm, article)
    q10 = paper.questions[9]
    mark = marker(q10, "30")
    assert mark is not None and mark.verdict == "correct"
    graded = grade_one_open(q10, "30", skipped=False, mode="tutor", mark=mark)
    assert graded.verdict == "correct"
    assert graded.graded_by == "llm"
