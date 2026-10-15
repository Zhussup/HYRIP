"""Question models: the IELTS reading repertoire, one discriminated union."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .article import Article, now_utc

QuestionType = Literal[
    "multiple_choice",
    "true_false_not_given",
    "yes_no_not_given",
    "matching_headings",
    "match_paragraphs",
    "completion",
    "short_answer",
    "summary_choice",
]

QUESTION_TYPES: tuple[str, ...] = (
    "multiple_choice",
    "true_false_not_given",
    "yes_no_not_given",
    "matching_headings",
    "match_paragraphs",
    "completion",
    "short_answer",
    "summary_choice",
)

OPEN_TYPES: frozenset[str] = frozenset({"completion", "short_answer"})

MC_KIND = Literal["scope", "extent", "time", "wrong_detail", None]
SUMMARY_KIND = Literal["narrow", "broad", "unsupported", None]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paragraph: str = ""
    quote: str
    quote_verified: bool = False


class McOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    trap_kind: MC_KIND = None


class SummaryOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    trap_kind: SUMMARY_KIND = None


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: int
    group_no: int = 0
    question_type: str  # narrowed by each subclass
    prompt: str
    answer_key: str
    evidence: list[Evidence] = Field(default_factory=list)
    explanation: str = ""


class MCQuestion(Question):
    question_type: Literal["multiple_choice"] = "multiple_choice"
    options: list[McOption] = Field(min_length=4, max_length=4)


class TFNGQuestion(Question):
    question_type: Literal["true_false_not_given"] = "true_false_not_given"
    answer_key: Literal["TRUE", "FALSE", "NOT GIVEN"]


class YNNGQuestion(Question):
    question_type: Literal["yes_no_not_given"] = "yes_no_not_given"
    answer_key: Literal["YES", "NO", "NOT GIVEN"]


class MatchHeadingsItem(Question):
    question_type: Literal["matching_headings"] = "matching_headings"
    paragraph_label: str


class MatchParagraphsItem(Question):
    question_type: Literal["match_paragraphs"] = "match_paragraphs"
    paragraph_label: str


class CompletionItem(Question):
    question_type: Literal["completion"] = "completion"
    acceptable: list[str] = Field(default_factory=list)
    word_limit: int = Field(default=1, ge=1, le=5)
    allows_numbers: bool = False


class ShortAnswerItem(CompletionItem):
    question_type: Literal["short_answer"] = "short_answer"


class SummaryChoiceQuestion(Question):
    question_type: Literal["summary_choice"] = "summary_choice"
    options: list[SummaryOption] = Field(min_length=3, max_length=4)


MCQuestion.model_rebuild()
Answerable = (
    MCQuestion
    | TFNGQuestion
    | YNNGQuestion
    | MatchHeadingsItem
    | MatchParagraphsItem
    | CompletionItem
    | ShortAnswerItem
    | SummaryChoiceQuestion
)
Question = Annotated[Answerable, Field(discriminator="type")]


class QuestionGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_no: int
    question_type: QuestionType
    instructions: str
    headings: list[str] | None = None
    items: list[Answerable]


class GenerationMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    created_at: datetime = Field(default_factory=now_utc)
    latency_s: float = 0.0
    repair_used: bool = False
    evidence_unverified: int = 0
    finish_reason: str | None = None


class QuestionPaper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    article_id: int | None = None
    title: str = "Untitled passage"
    groups: list[QuestionGroup] = Field(min_length=1)
    questions: list[Answerable] = Field(min_length=1)
    meta: GenerationMeta | None = None
    main_idea_number: int


def paragraph_label(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA, …"""
    if index < 26:
        return chr(ord("A") + index)
    return "A" + chr(ord("A") + index - 26)


def _make_question(qtype: str, item, group_options: list[str] | None) -> Answerable:
    """One raw item → one typed Question."""
    evidence = [Evidence(**e.model_dump()) for e in item.evidence]
    common = dict(
        number=item.number,
        prompt=item.prompt,
        answer_key=item.answer_key,
        evidence=evidence,
        explanation=item.explanation,
    )
    match qtype:
        case "multiple_choice":
            assert group_options is not None, "multiple_choice group needs a shared option bank"
            options = [McOption(text=t) for t in group_options]
            return MCQuestion(options=options, **common)
        case "true_false_not_given":
            return TFNGQuestion(**common)
        case "yes_no_not_given":
            return YNNGQuestion(**common)
        case "matching_headings":
            label = item.paragraph_label or item.prompt
            return MatchHeadingsItem(paragraph_label=label.strip(), **common)
        case "match_paragraphs":
            return MatchParagraphsItem(paragraph_label=item.answer_key.strip().upper(), **common)
        case "completion" | "short_answer":
            kwargs: dict = {}
            if item.word_limit is not None:
                kwargs["word_limit"] = item.word_limit
            if item.allows_numbers is not None:
                kwargs["allows_numbers"] = item.allows_numbers
            cls = CompletionItem if qtype == "completion" else ShortAnswerItem
            return cls(acceptable=item.acceptable or [], **kwargs, **common)
        case "summary_choice":
            assert item.options is not None and len(item.options) >= 3, (
                f"question {item.number}: summary_choice needs 3-4 options"
            )
            return SummaryChoiceQuestion(
                options=[SummaryOption(**o.model_dump()) for o in item.options], **common
            )
        case _:
            raise ValueError(f"unknown question type {qtype!r}")


def build_paper(
    raw, article: Article | None = None, article_id: int | None = None
) -> QuestionPaper:
    """Flatten RawGeneratedPaper into a QuestionPaper, validating numbering invariants.

    Raises ValueError for shape/numbering problems; content invariants (banks, NOT GIVEN ⇔
    no evidence, quotes) are checked separately by engine.validation.
    """
    groups: list[QuestionGroup] = []
    flat: dict[int, Answerable] = {}
    for g_no, raw_group in enumerate(raw.groups, start=1):
        qtype = raw_group.question_type
        if qtype == "multiple_choice" and (not raw_group.options or len(raw_group.options) != 4):
            raise ValueError("multiple_choice group needs exactly 4 shared options")
        items = [_make_question(qtype, item, raw_group.options) for item in raw_group.items]
        for q in items:
            q.group_no = g_no
            if q.number in flat:
                raise ValueError(f"duplicate question number {q.number}")
            flat[q.number] = q
        groups.append(
            QuestionGroup(
                group_no=g_no,
                question_type=qtype,
                instructions=raw_group.instructions,
                headings=raw_group.headings,
                items=items,
            )
        )
    numbers = sorted(flat)
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError(f"question numbers must be 1..{len(numbers)} consecutively; got {numbers}")
    summary_numbers = [
        q.number for g in groups if g.question_type == "summary_choice" for q in g.items
    ]
    if len(summary_numbers) != 1:
        raise ValueError("exactly one summary_choice group with exactly one item is required")
    title = raw.passage_title or (article.title if article is not None else "Untitled passage")
    if article_id is None and article is not None:
        article_id = article.id
    return QuestionPaper(
        article_id=article_id,
        title=title,
        groups=groups,
        questions=[flat[n] for n in numbers],
        main_idea_number=summary_numbers[0],
    )
