"""Article → QuestionPaper. One generation call, strict contracts, one repair round-trip."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import ValidationError

from ..config import Config
from ..errors import GenerationError
from ..llm.client import LLMClient
from ..llm.prompts import build_generation_messages, build_repair_messages
from ..models.article import Article
from ..models.llm_contracts import RawGeneratedPaper
from ..models.questions import GenerationMeta, QuestionPaper, build_paper
from .validation import check_converted_paper, check_raw_paper

# (type, count) profiles per total n. summary_choice is always exactly 1, always last.
_RECIPES: dict[int, list[tuple[str, int]]] = {
    8: [
        ("multiple_choice", 2),
        ("true_false_not_given", 1),
        ("matching_headings", 2),
        ("completion", 2),
        ("summary_choice", 1),
    ],
    10: [
        ("multiple_choice", 2),
        ("true_false_not_given", 2),
        ("matching_headings", 3),
        ("completion", 2),
        ("summary_choice", 1),
    ],
    12: [
        ("multiple_choice", 2),
        ("true_false_not_given", 2),
        ("matching_headings", 3),
        ("match_paragraphs", 2),
        ("completion", 2),
        ("summary_choice", 1),
    ],
    14: [
        ("multiple_choice", 2),
        ("true_false_not_given", 3),
        ("matching_headings", 3),
        ("match_paragraphs", 2),
        ("completion", 2),
        ("short_answer", 1),
        ("summary_choice", 1),
    ],
    16: [
        ("multiple_choice", 3),
        ("true_false_not_given", 3),
        ("matching_headings", 3),
        ("match_paragraphs", 2),
        ("completion", 2),
        ("short_answer", 2),
        ("summary_choice", 1),
    ],
}

MIN_QUESTIONS, MAX_QUESTIONS = 8, 16
_BUMP_ORDER = ["true_false_not_given", "multiple_choice", "match_paragraphs", "completion"]


@dataclass
class GenerationOutcome:
    paper: QuestionPaper
    raw_text: str
    warnings: list[str]


def mix_recipe(
    *, n: int, claims_mode: bool = False, n_paragraphs: int = 99
) -> list[tuple[str, int]]:
    """Deterministic default mix for n items (8..16), adapted to the passage shape."""
    if not (MIN_QUESTIONS <= n <= MAX_QUESTIONS):
        raise GenerationError(
            f"questions must be between {MIN_QUESTIONS} and {MAX_QUESTIONS}; got {n}"
        )
    base = 8 if n > 16 else n
    if base % 2 == 1:
        base -= 1
    profile = _RECIPES[base]
    recipe: list[tuple[str, int]] = list(profile)
    extra = n - sum(count for _, count in recipe)
    while extra > 0:  # spread odd remainders over the bump order
        for index, (qtype, count) in enumerate(recipe):
            if extra == 0:
                break
            if qtype in _BUMP_ORDER and not (qtype == "match_paragraphs" and n_paragraphs < 6):
                recipe[index] = (qtype, count + 1)
                extra -= 1
        else:
            break
    if claims_mode:  # writer's-claims flavour of the same trick
        recipe = [("yes_no_not_given" if t == "true_false_not_given" else t, c) for t, c in recipe]
    if n_paragraphs < 4:  # headings and which-paragraph need paragraphs to point at
        recipe = [
            ("completion" if t in ("matching_headings", "match_paragraphs") else t, c)
            for t, c in recipe
        ]
    elif n_paragraphs < 6:
        recipe = [("completion" if t == "match_paragraphs" else t, c) for t, c in recipe]
    merged: dict[str, int] = {}
    for qtype, count in recipe:
        merged[qtype] = merged.get(qtype, 0) + count
    order = [t for t, _ in recipe]
    out: list[tuple[str, int]] = []
    for qtype in dict.fromkeys(order):  # keep first-seen order, summary_choice last
        if qtype != "summary_choice":
            out.append((qtype, merged[qtype]))
    if "summary_choice" in merged:
        out.append(("summary_choice", merged["summary_choice"]))
    return out


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _fmt_schema_error(err: dict) -> str:
    loc = ".".join(str(p) for p in err["loc"]) or "<root>"
    return f"{loc}: {err['msg']}"


def parse_json_response(text: str) -> str:
    """JSON text from a completion: strips code fences and stray prose around the object."""
    cleaned = text.strip()
    cleaned = _FENCE.sub("", cleaned).strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first != -1 and last > first:
        return cleaned[first : last + 1]
    raise GenerationError("The model did not return a JSON object.")


def _validate_result(
    content: str, article: Article, expected_n: int
) -> tuple[RawGeneratedPaper | None, list[str], list[str], bool]:
    """Return (raw|None, hard_errors, soft_warnings, json_never_parsed) — never raise.

    json_never_parsed marks the unrepairable case: structurally broken or truncated JSON
    (pydantic's "json_invalid") gives the model nothing actionable to fix in place, so the
    paper must be rewritten from scratch. Parseable-but-schema-invalid output is the
    opposite: `extra="forbid"` violations are fed back verbatim to the repair round-trip,
    which can act on them. Hard errors are phrased for the repair round-trip: schema
    details and rule names, unlike a single unparseable-output shrug.
    """
    try:
        json_text = parse_json_response(content)
    except GenerationError:
        return None, ["the output was not a JSON object"], [], True
    try:
        raw = RawGeneratedPaper.model_validate_json(json_text)
    except ValidationError as exc:
        errors = exc.errors()
        hard = [f"schema: {_fmt_schema_error(e)}" for e in errors[:6]]
        return None, hard, [], any(err["type"] == "json_invalid" for err in errors)
    hard, soft = check_raw_paper(raw, article, expected_n=expected_n)
    return raw, hard, soft, False


def generate_paper(
    client: LLMClient,
    article: Article,
    *,
    config: Config,
    n_questions: int = 12,
    claims_mode: bool = False,
) -> GenerationOutcome:
    messages = build_generation_messages(
        article,
        mix_recipe(n=n_questions, claims_mode=claims_mode, n_paragraphs=len(article.paragraphs)),
    )
    result = client.chat(
        messages,
        json_mode=True,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    raw, hard, soft, unparseable = _validate_result(result.content, article, n_questions)
    repair_used = False
    if result.finish_reason == "length" or unparseable:
        # The output cannot be fixed by feedback: a length-cut JSON ends mid-object, so the
        # model has nothing actionable to repair and must rewrite from scratch anyway — do
        # exactly that, with token headroom, instead of a repair round-trip at the same
        # budget that just truncated.
        first = list(hard) or ["the output was cut at the token budget"]
        result = client.chat(
            messages,
            json_mode=True,
            temperature=config.temperature,
            max_tokens=int(config.max_tokens * 2),
        )
        raw, hard, soft, unparseable = _validate_result(result.content, article, n_questions)
        if unparseable or result.finish_reason == "length":
            second = list(hard)
            if result.finish_reason == "length":  # known cut, right at a parseable end
                second.append(f"cut again at {int(config.max_tokens * 2)} output tokens")
            lines = [
                "Question-paper generation failed validation twice.",
                f"- first: {'; '.join(first)}",
                f"- second: {'; '.join(second)}",
            ]
            if unparseable or result.finish_reason == "length":
                lines.append(
                    "Hint: the paper likely does not fit the output budget — raise"
                    " HYRIP_MAX_TOKENS or ask for fewer questions."
                )
            raise GenerationError("\n".join(lines))
    if hard:  # parseable but drifted: the normal repair round-trip
        repair_messages = build_repair_messages(messages, result.content, hard)
        second = client.chat(
            repair_messages,
            json_mode=True,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        raw, hard2, soft2, _ = _validate_result(second.content, article, n_questions)
        if hard2:
            raise GenerationError(
                "Question-paper generation failed validation twice.\n"
                f"- first: {'; '.join(hard)}\n- second: {'; '.join(hard2)}"
            )
        result, repair_used, soft = second, True, soft2

    try:
        paper = build_paper(raw, article=article)
    except ValueError as exc:
        raise GenerationError(f"paper construction failed: {exc}") from exc
    converted = check_converted_paper(paper, article)
    paper.meta = GenerationMeta(
        model=config.model,
        latency_s=result.latency_s,
        repair_used=repair_used,
        finish_reason=result.finish_reason,
        evidence_unverified=sum(
            1 for q in paper.questions for e in q.evidence if not e.quote_verified
        ),
    )
    return GenerationOutcome(paper=paper, raw_text=result.content, warnings=soft + converted)
