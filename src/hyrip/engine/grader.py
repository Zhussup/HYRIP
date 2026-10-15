"""Grading: deterministic for closed questions, LLM-assisted for open answers.

Word limits are enforced here, never by the LLM: over-limit in exam mode is incorrect
whatever the meaning; in tutor mode it is partial ("right meaning, but the limit...").
A `partial` from the marker (right meaning, wrong form) is also incorrect in exam mode,
matching real IELTS marking, and stays "almost" in tutor mode.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from ..ingest.normalize import quote_finds, quote_normalized
from ..llm.client import LLMClient
from ..llm.prompts import build_marker_messages
from ..models.grading import GradedItem, Mode, UserAnswer
from ..models.llm_contracts import MarkResult, VerdictBatch
from ..models.questions import (
    OPEN_TYPES,
    CompletionItem,
    MatchHeadingsItem,
    MatchParagraphsItem,
    MCQuestion,
    QuestionPaper,
    SummaryChoiceQuestion,
    TFNGQuestion,
    YNNGQuestion,
)

CLOSED_TYPES = frozenset(
    {
        "multiple_choice",
        "true_false_not_given",
        "yes_no_not_given",
        "matching_headings",
        "match_paragraphs",
        "summary_choice",
    }
)

_TRUE = {"true", "t"}
_FALSE = {"false", "f"}
_YES = {"yes", "y"}
_NO = {"no", "n"}
_NOT_GIVEN = {"not given", "notgiven", "ng", "n g"}

_ROMAN = ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x")


def canonical_letter(options: list[str], raw: str) -> str | None:
    """User letter / 1-based index / full option text -> the target letter (A first)."""
    token = raw.strip().rstrip(".").strip()
    if len(token) == 1 and token.upper() in "ABCD":
        return token.upper()
    if re.fullmatch(r"[1-4]", token) and len(options) >= int(token):
        return "ABCD"[int(token) - 1]
    if len(token) > 2:
        needle = quote_normalized(token)
        for index, option in enumerate(options):
            if quote_normalized(option) == needle:
                return "ABCD"[index]
        if prefix := _letter_prefix(token):  # "B) The roofs…" style, after full text failed
            return prefix
    return None


def _letter_prefix(token: str) -> str | None:
    match = re.match(r"^\(?([A-Da-d])[\).,:\s]", token)
    return match.group(1).upper() if match else None


def canonical_tfng(raw: str, *, yes_mode: bool) -> str:
    token = quote_normalized(raw).strip()
    if token in _NOT_GIVEN:
        return "NOT GIVEN"
    if yes_mode:
        if token in _YES:
            return "YES"
        if token in _NO:
            return "NO"
    else:
        if token in _TRUE:
            return "TRUE"
        if token in _FALSE:
            return "FALSE"
    return token.upper()


def heading_key(text: str) -> str:
    """ "iv. Urban hives" -> "iv"; a bank entry without the numeral passes through."""
    prefix, _, _rest = text.strip().partition(".")
    return prefix.lower() if prefix.lower() in _ROMAN else text.strip().lower()


def canonical_heading(raw: str, bank: list[str]) -> str:
    token = raw.strip().rstrip(".").strip()
    if token.lower() in _ROMAN:
        return token.lower()
    if token.isdigit():
        index = int(token) - 1
        if 0 <= index < len(bank):
            return heading_key(bank[index])
    needle = quote_normalized(token)
    for heading in bank:
        for variant in (quote_normalized(heading), quote_normalized(_after_numeral(heading))):
            if needle == variant:
                return heading_key(heading)
    return token.lower()


def _after_numeral(heading: str) -> str:
    """'iv. Learning through hives' -> 'Learning through hives'; bare text passes through."""
    _prefix, _dot, rest = heading.strip().partition(".")
    return rest or heading


def canonical_paragraph_label(raw: str) -> str:
    token = quote_normalized(raw).upper().strip()
    if token.startswith("PARAGRAPH "):
        token = token[len("PARAGRAPH ") :].strip()
    return token


def heading_bank_of(paper: QuestionPaper, group_no: int) -> list[str]:
    for group in paper.groups:
        if group.group_no == group_no and group.headings:
            return group.headings
    return []


# ---- closed grading ----------------------------------------------------------


def grade_closed(
    question, raw: str | None, *, skipped: bool, headings_bank: list[str] | None = None
) -> GradedItem:
    """Deterministic marking for the six closed types. Blank/None/skipped -> skipped."""
    base = dict(number=question.number, question_type=question.question_type, user_raw=raw)
    if skipped or raw is None or not raw.strip():
        return GradedItem(**base, verdict="skipped")
    match question:
        case MCQuestion() | SummaryChoiceQuestion():
            options = [option.text for option in question.options]
            got = canonical_letter(options, raw)
            equal = got == question.answer_key.upper().strip()
        case TFNGQuestion():
            equal = canonical_tfng(raw, yes_mode=False) == question.answer_key
        case YNNGQuestion():
            equal = canonical_tfng(raw, yes_mode=True) == question.answer_key
        case MatchHeadingsItem():
            bank = headings_bank or []
            equal = canonical_heading(raw, bank) == heading_key(question.answer_key)
        case MatchParagraphsItem():
            equal = canonical_paragraph_label(raw) == question.answer_key.strip().upper()
        case _:
            raise ValueError(f"{question.question_type} is not a closed type")
    verdict = "correct" if equal else "incorrect"
    return GradedItem(
        **base,
        verdict=verdict,
        why=question.explanation,
        evidence=list(question.evidence),
        graded_by="deterministic",
    )


# ---- open grading --------------------------------------------------------------

WORD_RE = re.compile(r"\S+")


def words_in(raw: str) -> int:
    return len(WORD_RE.findall(raw))


def word_count_ok(question: CompletionItem, raw: str | None) -> bool | None:
    if raw is None or not raw.strip():
        return None
    return words_in(raw) <= question.word_limit


def containment_match(question: CompletionItem, raw: str) -> bool:
    """No-LLM fallback: the candidate's words, normalized, must equal the key or a variant."""
    candidate = quote_normalized(raw)
    accepted = [
        quote_normalized(question.answer_key),
        *(quote_normalized(a) for a in question.acceptable),
    ]
    return bool(candidate) and candidate in [a for a in accepted if a]


@dataclass
class OpenPending:
    question: CompletionItem
    raw: str
    within_limit: bool


def grade_one_open(
    question: CompletionItem,
    raw: str | None,
    *,
    skipped: bool,
    mode: Mode,
    mark: MarkResult | None = None,
    article=None,
) -> GradedItem:
    """Mark one completion/short_answer item deterministically, upgrading with a marker
    verdict when one is at hand."""
    base = dict(number=question.number, question_type=question.question_type, user_raw=raw)
    if skipped or raw is None or not raw.strip():
        return GradedItem(**base, verdict="skipped")
    within = word_count_ok(question, raw)
    issues: list[str] = []
    if within is False:
        issues.append(f"{words_in(raw)} words, limit is {question.word_limit}")
        return GradedItem(
            **base,
            verdict="incorrect" if mode == "exam" else "partial",
            word_count_ok=False,
            issues=issues,
            why=question.explanation,
            evidence=list(question.evidence),
            graded_by="deterministic",
        )
    if mark is not None:
        verdict = mark.verdict
        if mode == "exam" and verdict == "partial":
            verdict = "incorrect"
            issues.append("right meaning, wrong form")
        return GradedItem(
            **base,
            verdict=verdict,
            word_count_ok=True,
            issues=[*mark.issues, *issues],
            why=mark.why,
            teaching_note=mark.teaching_note,
            evidence=_mark_evidence(mark, question, article),
            graded_by="llm",
        )
    verdict = "correct" if containment_match(question, raw) else "incorrect"
    return GradedItem(
        **base,
        verdict=verdict,
        word_count_ok=True,
        issues=issues,
        why=question.explanation,
        evidence=list(question.evidence),
        graded_by="fallback",
    )


def _mark_evidence(mark: MarkResult, question: CompletionItem, article) -> list:
    from ..models.questions import Evidence

    if mark.evidence is None:
        return list(question.evidence)
    verified = False
    if article is not None and mark.evidence.paragraph:
        try:
            paragraph_text = article.paragraph_for(mark.evidence.paragraph)
        except (ValueError, IndexError):
            paragraph_text = None
        verified = bool(paragraph_text and quote_finds(mark.evidence.quote, paragraph_text))
    return [
        Evidence(
            paragraph=mark.evidence.paragraph,
            quote=mark.evidence.quote,
            quote_verified=verified,
        )
    ]


# ---- paper-level facade ---------------------------------------------------------


def grade_paper(
    paper: QuestionPaper,
    answers: dict[int, UserAnswer],
    *,
    mode: Mode,
    marker: BatchMarker | None = None,
    article=None,
) -> dict[int, GradedItem]:
    """Grade every question in the paper (exam-mode finish).

    Open items within their word limit go through `marker` as one batch; over-limit
    items, unanswered items, and items missing from the marker output are marked
    deterministically (override / skipped / containment fallback).
    """
    graded: dict[int, GradedItem] = {}
    pending: list[OpenPending] = []
    user_answers: dict[int, str] = {}
    for question in paper.questions:
        answer = answers.get(question.number) or UserAnswer(number=question.number, skipped=True)
        if question.question_type in OPEN_TYPES:
            assert isinstance(question, CompletionItem)
            raw = answer.raw if not (answer.skipped or not answer.raw) else None
            if raw is None:
                graded[question.number] = grade_one_open(question, None, skipped=True, mode=mode)
            else:
                within = word_count_ok(question, raw) is not False
                user_answers[question.number] = raw
                if within:
                    pending.append(OpenPending(question=question, raw=raw, within_limit=True))
                else:  # exam rule: the limit is decisive, no meaning check needed
                    graded[question.number] = grade_one_open(
                        question, raw, skipped=False, mode=mode, mark=None
                    )
        else:
            graded[question.number] = grade_closed(
                question,
                answer.raw,
                skipped=answer.skipped,
                headings_bank=heading_bank_of(paper, question.group_no),
            )
    if pending:
        marks = marker or _noop_marker
        items = [(p.question, True) for p in pending]
        try:
            by_number = marks(items, user_answers)
        except Exception:
            by_number = {}
        for pending_item in pending:
            mark = by_number.get(pending_item.question.number)
            graded[pending_item.question.number] = grade_one_open(
                pending_item.question,
                pending_item.raw,
                skipped=False,
                mode=mode,
                mark=mark,
                article=article,
            )
    return graded


def _noop_marker(items: list[tuple], user_answers: dict[int, str]) -> dict[int, MarkResult]:
    return {}


# ---- marker factories (service owns when/whether to call) -----------------------

BatchMarker = Callable[[list[tuple], dict[int, str]], dict[int, MarkResult]]


def make_batch_marker(client: LLMClient, article, temperature: float = 0.2) -> BatchMarker:
    """One marker call for a whole batch of open items (exam-mode finish)."""

    def marker(items: list[tuple], user_answers: dict[int, str]) -> dict[int, MarkResult]:
        messages = build_marker_messages(article, items, user_answers)
        result = client.chat(messages, json_mode=True, temperature=temperature)
        batch = VerdictBatch.model_validate_json(_strip_fence(result.content))
        return {r.number: r for r in batch.results}

    return marker


def make_single_marker(client: LLMClient, article, temperature: float = 0.2):
    """One marker call for a single open item (tutor mode marks on submit)."""

    def marker(question: CompletionItem, raw: str) -> MarkResult | None:
        messages = build_marker_messages(
            article,
            [(question, word_count_ok(question, raw) is not False)],
            {question.number: raw},
        )
        result = client.chat(messages, json_mode=True, temperature=temperature)
        batch = VerdictBatch.model_validate_json(_strip_fence(result.content))
        return batch.results[0] if batch.results else None

    return marker


def _strip_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = cleaned.strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    first, last = cleaned.find("{"), cleaned.rfind("}")
    if first != -1 and last > first:
        return cleaned[first : last + 1]
    return cleaned or "{}"
