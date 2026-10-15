"""Invariance checks for a generated paper, before it is shown to a user.

Hard errors make the paper unanswerable (bad numbering, keys outside their banks, a NOT GIVEN
item with an evidence quote) and trigger the repair round-trip. Soft problems are surfaced as
warnings — especially unverified evidence quotes; one paraphrased quote must not cost the user
eleven good questions.
"""

from __future__ import annotations

from ..ingest.normalize import quote_finds
from ..models.article import Article
from ..models.llm_contracts import RawGeneratedPaper, RawItem
from ..models.questions import OPEN_TYPES

ROMAN_HEADINGS: tuple[str, ...] = ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x")
_ROMAN_TO_N = {n: i + 1 for i, n in enumerate(ROMAN_HEADINGS)}

MC_LETTERS = "ABCD"


def parse_roman(key: str) -> int | None:
    """'iv.' / 'IV' -> 4; None unless the token is one of the bank numerals i..x."""
    return _ROMAN_TO_N.get(key.strip().rstrip(".").lower())


def heading_token(heading: str) -> str:
    """'i. Urban legends' -> 'i'; 'Urban legends' -> 'Urban legends'."""
    token = heading.strip()
    prefix, _, rest = token.partition(".")
    if prefix.lower() in _ROMAN_TO_N and rest:
        return prefix.lower()
    return token


def check_raw_paper(
    raw: RawGeneratedPaper,
    article: Article,
    *,
    expected_n: int,
) -> tuple[list[str], list[str]]:
    """Validate raw LLM output. Returns (hard_errors, warnings)."""
    hard: list[str] = []
    soft: list[str] = []

    flat = [item for group in raw.groups for item in group.items]
    numbers = [item.number for item in flat]
    if len(set(numbers)) != len(numbers):
        hard.append(f"item numbers repeat: {sorted(numbers)}")
    elif numbers != list(range(1, len(numbers) + 1)):
        hard.append(f"item numbers must be 1..{len(numbers)} consecutively; got {numbers}")

    summary_groups = [group for group in raw.groups if group.question_type == "summary_choice"]
    if len(summary_groups) != 1 or raw.groups[-1].question_type != "summary_choice":
        hard.append(
            f"expected exactly one summary_choice group, placed last (found {len(summary_groups)})"
        )
    else:
        summary = raw.groups[-1]
        if len(summary.items) != 1:
            hard.append(f"summary_choice group must hold exactly 1 item (has {len(summary.items)})")
        elif summary.items[0].options is None:
            hard.append("summary_choice item is missing options")
        else:
            opts = summary.items[0].options
            letters = MC_LETTERS[: len(opts)]
            key = summary.items[0].answer_key
            if key not in letters:
                hard.append(f"summary_choice key {key!r} is not one of {list(letters)}")
            else:
                correct = opts[letters.index(key)]
                if correct.trap_kind is not None:
                    # Models persistently tag the correct option. The key comes from
                    # answer_key, so the tag is noise: strip it instead of forcing a
                    # repair round-trip the model has already shown it repeats.
                    correct.trap_kind = None
                wrong = [o for i, o in enumerate(opts) if letters[i] != key]
                if any(o.trap_kind is None for o in wrong):
                    soft.append("summary_choice: some distractors carry no trap_kind")

    labels = set(article.paragraph_labels)
    for group in raw.groups:
        if group.question_type == "matching_headings":
            if not group.headings:
                hard.append("matching_headings group is missing its heading bank")
                continue
            bank = {heading_token(h) for h in group.headings}
            for item in group.items:
                token = heading_token(item.answer_key)
                if token not in bank:
                    hard.append(
                        f"matching_headings key {item.answer_key!r} is not in the heading bank"
                    )
            keys = [heading_token(i.answer_key) for i in group.items]
            if len(set(keys)) != len(keys):
                soft.append("a heading was chosen for two paragraphs (bank says use each once)")
        elif group.question_type == "match_paragraphs":
            for item in group.items:
                letter = item.answer_key.strip().upper()
                if letter not in labels:
                    hard.append(f"match_paragraphs key {item.answer_key!r} is not a passage label")
        elif group.question_type == "multiple_choice":
            if len(group.options or []) != 4:
                hard.append("multiple_choice group needs exactly 4 shared options")
            for item in group.items:
                if item.answer_key not in MC_LETTERS:
                    hard.append(f"multiple_choice key {item.answer_key!r} is not one of A/B/C/D")
        elif group.question_type in ("true_false_not_given", "yes_no_not_given"):
            allowed = (
                {"TRUE", "FALSE", "NOT GIVEN"}
                if group.question_type == "true_false_not_given"
                else {
                    "YES",
                    "NO",
                    "NOT GIVEN",
                }
            )
            for item in group.items:
                if item.answer_key not in allowed:
                    hard.append(
                        f"question {item.number}: {group.question_type} key {item.answer_key!r}"
                        f" must be one of {sorted(allowed)}"
                    )

    for group in raw.groups:
        ng_allowed = group.question_type in ("true_false_not_given", "yes_no_not_given")
        for item in group.items:
            is_ng = ng_allowed and item.answer_key == "NOT GIVEN"
            if is_ng and item.evidence:
                hard.append(
                    f"question {item.number}: NOT GIVEN must ship no evidence"
                    f" (found {len(item.evidence)})"
                )
            elif not is_ng and not item.evidence:
                soft.append(f"question {item.number}: no evidence quotes given")
            if group.question_type in OPEN_TYPES:
                _check_open(item, soft)
            _verify_item_quotes(item, article, soft)

    if abs(len(flat) - expected_n) > 1:
        soft.append(f"requested {expected_n} items, the model returned {len(flat)}")

    return hard, soft


def _check_open(item: RawItem, warnings: list[str]) -> None:
    if "____" not in item.prompt:
        warnings.append(f"question {item.number}: completion prompt has no ____ gap")
    limit = item.word_limit or 3
    for cand in [item.answer_key, *(item.acceptable or [])]:
        if len(cand.split()) > limit:
            warnings.append(
                f"question {item.number}: answer {cand!r} exceeds the {limit}-word limit"
            )


def _verify_item_quotes(item: RawItem, article: Article, warnings: list[str]) -> list[bool]:
    verified: list[bool] = []
    for ev in item.evidence:
        label = ev.paragraph.strip().upper()
        paragraph = article.paragraph_for(label)
        if paragraph is None:
            warnings.append(
                f"question {item.number}: evidence paragraph {ev.paragraph!r} is unknown"
            )
            verified.append(False)
        elif quote_finds(ev.quote, paragraph):
            verified.append(True)
        else:
            warnings.append(
                f"question {item.number}: evidence quote is not verbatim in paragraph {label}"
            )
            verified.append(False)
    return verified


def check_converted_paper(paper, article: Article) -> list[str]:
    """Mark evidence quote_verified on the built paper; returns warnings (soft only)."""
    warnings: list[str] = []
    unverified = 0
    for question in paper.questions:
        for ev in question.evidence:
            paragraph = article.paragraph_for(ev.paragraph.strip().upper())
            if paragraph is not None and quote_finds(ev.quote, paragraph):
                ev.quote_verified = True
            else:
                ev.quote_verified = False
                unverified += 1
    if unverified:
        warnings.append(
            f"{unverified} evidence quote(s) not verbatim in the passage (shown unmarked)"
        )
    return warnings


__all__ = [
    "check_converted_paper",
    "check_raw_paper",
    "heading_token",
    "parse_roman",
]
