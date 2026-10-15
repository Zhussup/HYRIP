"""Paper validation: hard invariants break, soft issues only warn, quotes are verified."""

from __future__ import annotations

from hyrip.engine.validation import (
    check_converted_paper,
    check_raw_paper,
    heading_token,
    parse_roman,
)
from hyrip.models.questions import build_paper


def test_valid_fixture_paper_is_clean(raw_paper, article):
    hard, soft = check_raw_paper(raw_paper, article, expected_n=12)
    assert hard == [], hard
    # every quote in the fixture is verbatim; distractors are tagged
    assert soft == [], soft


def test_hard_numbering_gap(raw_paper, article):
    raw_paper.groups[0].items[1].number = 20
    hard, _ = check_raw_paper(raw_paper, article, expected_n=12)
    assert any("consecutively" in e for e in hard)


def test_hard_mc_key_outside_bank(raw_paper, article):
    raw_paper.groups[0].items[0].answer_key = "E"
    hard, _ = check_raw_paper(raw_paper, article, expected_n=12)
    assert any("A/B/C/D" in e for e in hard)


def test_hard_headings_key_outside_bank(raw_paper, article):
    raw_paper.groups[2].items[0].answer_key = "ix"
    hard, _ = check_raw_paper(raw_paper, article, expected_n=12)
    assert any("heading bank" in e for e in hard)


def test_hard_match_paragraphs_unknown_label(raw_paper, article):
    raw_paper.groups[3].items[0].answer_key = "Z"
    hard, _ = check_raw_paper(raw_paper, article, expected_n=12)
    assert any("passage label" in e for e in hard)


def test_hard_not_given_with_evidence(raw_paper, article):
    raw_paper.groups[1].items[0].answer_key = "NOT GIVEN"
    hard, _ = check_raw_paper(raw_paper, article, expected_n=12)
    assert any("NOT GIVEN" in e for e in hard)


def test_hard_tfng_key_literal(raw_paper, article):
    raw_paper.groups[1].items[0].answer_key = "true"
    hard, _ = check_raw_paper(raw_paper, article, expected_n=12)
    assert any("must be one of" in e for e in hard)


def test_summary_group_missing_trap_tags_is_soft(raw_paper, article):
    for option in raw_paper.groups[-1].items[0].options:
        option.trap_kind = None
    hard, soft = check_raw_paper(raw_paper, article, expected_n=12)
    assert hard == []
    assert any("trap_kind" in s for s in soft)


def test_summary_correct_option_trap_tag_is_stripped(raw_paper, article):
    """Models tag the correct option too; the key lives in answer_key, so strip the noise —
    no hard failure, no wasted repair round-trip."""
    options = raw_paper.groups[-1].items[0].options
    key = raw_paper.groups[-1].items[0].answer_key
    letters = ["A", "B", "C"]
    correct = options[letters.index(key)]
    correct.trap_kind = "narrow"
    hard, _ = check_raw_paper(raw_paper, article, expected_n=12)
    assert hard == []
    assert correct.trap_kind is None


def test_soft_fewer_items_than_expected(raw_paper, article):
    """±1 is tolerated; a bigger gap is a soft warning only."""
    hard, soft = check_raw_paper(raw_paper, article, expected_n=10)
    assert hard == []
    assert any("returned 12" in s for s in soft)


def test_soft_completion_without_gap(raw_paper, article):
    raw_paper.groups[4].items[0].prompt = raw_paper.groups[4].items[0].prompt.replace("____", "___")
    _, soft = check_raw_paper(raw_paper, article, expected_n=12)
    assert any("no ____ gap" in s for s in soft)


def test_soft_quote_not_verbatim(raw_paper, article):
    raw_paper.groups[0].items[0].evidence[0].quote = "Bees lived happily on every rooftop."
    _, soft = check_raw_paper(raw_paper, article, expected_n=12)
    assert any("not verbatim" in s for s in soft)


def test_converted_paper_marks_quotes(raw_paper, article):
    paper = build_paper(raw_paper, article=article)
    warnings = check_converted_paper(paper, article)
    assert warnings == []
    assert all(e.quote_verified for q in paper.questions for e in q.evidence)
    paper.questions[0].evidence[0].quote = "invented words only"
    warnings = check_converted_paper(paper, article)
    assert warnings and not paper.questions[0].evidence[0].quote_verified


def test_parse_roman_and_heading_token():
    assert parse_roman("iv.") == 4
    assert parse_roman("IV") == 4
    assert parse_roman("x") == 10
    assert parse_roman("eleven") is None
    assert heading_token("i. Urban legends") == "i"
    assert heading_token("Urban legends") == "Urban legends"
