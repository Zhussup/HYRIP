"""Question model behaviour: union parsing, invariants of shape, helpers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hyrip.models.questions import (
    MatchParagraphsItem,
    MCQuestion,
    QuestionPaper,
    build_paper,
    paragraph_label,
)


def test_paragraph_labels_beyond_26():
    assert paragraph_label(0) == "A"
    assert paragraph_label(25) == "Z"
    assert paragraph_label(26) == "AA"


def test_build_paper_flattens_and_anchors(raw_paper, article):
    paper = build_paper(raw_paper, article=article, article_id=None)
    assert isinstance(paper, QuestionPaper)
    assert len(paper.questions) == 12
    assert [q.number for q in paper.questions] == list(range(1, 13))
    assert paper.main_idea_number == 12
    assert paper.title == raw_paper.passage_title
    # groups keep their items; flat numbering lines up with groups
    headings_group = paper.groups[2]
    assert headings_group.question_type == "matching_headings"
    assert [q.number for q in headings_group.items] == [5, 6, 7]
    assert headings_group.headings and headings_group.items[0].paragraph_label == "E"


def test_build_paper_duplicate_number_rejected(raw_paper):
    raw_paper.groups[0].items[1].number = 1
    with pytest.raises(ValueError, match="duplicate question number"):
        build_paper(raw_paper, article=None)


def test_build_paper_gap_in_numbering_rejected(raw_paper):
    raw_paper.groups[0].items[1].number = 20
    with pytest.raises(ValueError, match="consecutively"):
        build_paper(raw_paper, article=None)


def test_build_paper_mc_group_needs_four_options(raw_paper):
    raw_paper.groups[0].options = raw_paper.groups[0].options[:3]
    with pytest.raises(ValueError, match="exactly 4"):
        build_paper(raw_paper, article=None)


def test_tfng_answer_key_is_checked(raw_paper):
    raw_paper.groups[1].items[0].answer_key = "MAYBE"
    with pytest.raises(ValidationError):
        build_paper(raw_paper, article=None)


def test_mc_requires_four_options(raw_paper, article):
    paper = build_paper(raw_paper, article=article)
    mc = paper.questions[0]
    assert isinstance(mc, MCQuestion)
    assert len(mc.options) == 4
    with pytest.raises(ValidationError):
        MCQuestion.model_validate({**mc.model_dump(), "options": mc.options[:3]})


def test_union_dispatches_by_type(raw_paper, article):
    paper = build_paper(raw_paper, article=article)
    types = [q.question_type for q in paper.questions]
    assert types[:2] == ["multiple_choice", "multiple_choice"]
    assert types[4] == "matching_headings"
    assert types[7] == "match_paragraphs"
    assert types[-1] == "summary_choice"
    match_item = paper.questions[8]
    assert isinstance(match_item, MatchParagraphsItem)
    assert match_item.paragraph_label == "D"


def test_extra_fields_forbidden(raw_paper, article):
    paper = build_paper(raw_paper, article=article)
    with pytest.raises(ValidationError):
        MCQuestion.model_validate({**paper.questions[0].model_dump(), "trap_extra": True})


def test_paper_requires_main_idea(raw_paper):
    raw_paper.groups[-1].question_type = "match_paragraphs"
    with pytest.raises(ValueError, match="summary_choice"):
        build_paper(raw_paper, article=None)
