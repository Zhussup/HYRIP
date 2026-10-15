"""Key-stripped question views for the frontends: never leak answers before grading."""

from __future__ import annotations

from ..models.questions import (
    CompletionItem,
    MatchHeadingsItem,
    MatchParagraphsItem,
    MCQuestion,
    QuestionGroup,
    QuestionPaper,
    SummaryChoiceQuestion,
)


def group_meta(group: QuestionGroup) -> dict:
    return {
        "group_no": group.group_no,
        "question_type": group.question_type,
        "instructions": group.instructions,
        "headings": list(group.headings) if group.headings else None,
    }


def public_question(question) -> dict:
    """One question with every answer-revealing field removed."""
    view: dict = {
        "number": question.number,
        "group_no": question.group_no,
        "question_type": question.question_type,
        "prompt": question.prompt,
    }
    match question:
        case MCQuestion() | SummaryChoiceQuestion():
            view["options"] = [o.text for o in question.options]
        case MatchHeadingsItem() | MatchParagraphsItem():
            view["paragraph_label"] = question.paragraph_label
        case CompletionItem():
            view["word_limit"] = question.word_limit
    return view


def public_view(paper: QuestionPaper, answered: set[int]) -> dict:
    """The whole paper, safe to send: groups + stripped questions + what's answered."""
    return {
        "title": paper.title,
        "n_questions": len(paper.questions),
        "main_idea_number": paper.main_idea_number,
        "groups": [group_meta(g) for g in paper.groups],
        "questions": [public_question(q) for q in paper.questions],
        "answered": sorted(answered),
    }
