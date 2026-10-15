"""Every LLM prompt in one module: generation, marking, tutoring, coaching.

The system prompts encode IELTS item-writing conventions; keep them precise and short —
each rule earns its place by fixing an observed generation failure.
"""

from __future__ import annotations

from ..models.article import Article
from ..models.grading import GradedItem, ReadingObservation, Tally
from ..models.questions import (
    CompletionItem,
    MatchHeadingsItem,
    MatchParagraphsItem,
    MCQuestion,
    ShortAnswerItem,
    SummaryChoiceQuestion,
)

GENERATION_SYSTEM = """You are a senior IELTS Academic Reading item writer. Given one passage, \
you write a complete Cambridge-style question set ABOUT THAT PASSAGE ONLY.

Hard rules:
1. Use ONLY the passage. Invent nothing. Never test background knowledge.
2. Every answer must be provable by a verbatim quote copied EXACTLY from the passage. \
Put each quote's paragraph label in its "paragraph" field.
3. NOT GIVEN items: "evidence": [] and the statement must be neither confirmed nor contradicted \
(the trick is that the passage is silent or only partially about it).
4. Question stems MUST paraphrase the passage. Never reuse a long span of passage text in a stem.
5. Distractors must be plausible: supported by the passage but answering the wrong detail \
— different scope, extent, time, or the wrong actor.
6. matching_headings: use the roman-numeral heading bank provided; add 2 extra headings that are \
plausible but match no paragraph; use each heading at most once; not all headings are used.
7. summary_choice is ALWAYS the last group with EXACTLY ONE item. Its options are full-sentence \
candidate main ideas of the whole passage: one correct, one TOO NARROW (one paragraph's point), \
one TOO BROAD or UNSUPPORTED. Tag each distractor's trap_kind ("narrow"/"broad"/"unsupported"); \
the correct option has no trap_kind.
8. completion and short_answer: reference and acceptable answers must respect the word limit \
(each word_limit or fewer words).
9. Order groups so items follow the order in which their answers appear in the passage. \
summary_choice stays last.
10. Output ONE JSON object, no markdown, no commentary.
11. Emit EXACTLY ONE group per recipe entry, in recipe order, nothing more — no splitting a
   type across groups and no extra groups. The groups array holds as many entries as the
   recipe lists (typically 5-6)."""

_JSON_SKELETON = """{
  "passage_title": "<short title>",
  "groups": [
    {
      "question_type": "multiple_choice",
      "instructions": "CHOOSE THE CORRECT LETTER, A-D",
      "options": ["<full option text A>", "<B>", "<C>", "<D>"],
      "items": [
        {"number": 1, "prompt": "<paraphrased stem>", "answer_key": "A",
         "evidence": [{"paragraph": "C", "quote": "<verbatim from paragraph C>"}],
         "explanation": "<why the key is right and the tempting distractor is wrong>"}
      ]
    },
    {
      "question_type": "true_false_not_given",
      "instructions": "DO THE FOLLOWING STATEMENTS AGREE WITH THE INFORMATION GIVEN?",
      "items": [
        {"number": n, "prompt": "<statement>",
         "answer_key": "TRUE" | "FALSE" | "NOT GIVEN",
         "evidence": [...], "explanation": "..."}
      ]
    },
    {
      "question_type": "matching_headings",
      "instructions": "CHOOSE THE CORRECT HEADING FOR EACH PARAGRAPH",
      "headings": ["i. <heading>", "ii. ...", "iii. ...", ...],
      "items": [
        {"number": n, "prompt": "Choose the correct heading for paragraph <X>.",
         "paragraph_label": "<X>", "answer_key": "IV",
         "evidence": [...], "explanation": "..."}
      ]
    },
    {
      "question_type": "match_paragraphs",
      "instructions": "WHICH PARAGRAPH CONTAINS THE FOLLOWING INFORMATION?",
      "items": [
        {"number": n, "prompt": "<paraphrased information>", "answer_key": "D",
         "evidence": [{"paragraph": "D", "quote": "..."}], "explanation": "..."}
      ]
    },
    {
      "question_type": "completion",
      "instructions": "COMPLETE EACH SENTENCE WITH NO MORE THAN THREE WORDS",
      "items": [
        {
          "number": n,
          "prompt": "<sentence with ____ gap>",
          "answer_key": "<<=3 words>",
          "acceptable": ["<variant>"],
          "word_limit": 3,
          "evidence": [...],
          "explanation": "..."
        }
      ]
    },
    {
      "question_type": "short_answer",
      "instructions": "ANSWER USING NO MORE THAN TWO WORDS",
      "items": [
        {
          "number": n,
          "prompt": "<question>",
          "answer_key": "<<=2 words>",
          "acceptable": [...],
          "word_limit": 2,
          "evidence": [...],
          "explanation": "..."
        }
      ]
    },
    {
      "question_type": "summary_choice",
      "instructions": "WHICH STATEMENT BEST SUMMARISES THE MAIN IDEA OF THE PASSAGE?",
      "items": [
        {
          "number": n,
          "prompt": "Which statement best summarises the whole passage?",
          "answer_key": "B",
          "options": [
            {"text": "<full-sentence main idea>", "trap_kind": null},
            {"text": "<TOO NARROW: one paragraph only>", "trap_kind": "narrow"},
            {"text": "<TOO BROAD / UNSUPPORTED>", "trap_kind": "broad"}
          ],
          "evidence": [...],
          "explanation": "..."
        }
      ]
    }
  ]
}"""

_MARKDOWN_EXAMPLE = """{
  "prompt": "Beekeeping in towns grew after the ____ of hives on rooftops became legal.",
  "answer_key": "legalisation",
  "acceptable": ["legalization", "legalisation of hives"],
  "word_limit": 3,
  "evidence": [{"paragraph": "B", "quote": "since rooftop hives were legalised in 2010"}]
}"""


def render_passage(article: Article) -> str:
    lines = [
        f"{label}. {text}"
        for label, text in zip(article.paragraph_labels, article.paragraphs, strict=True)
    ]
    return "\n\n".join(lines)


def _type_label(qtype: str) -> str:
    names = {
        "multiple_choice": "multiple choice (choose letter A-D)",
        "true_false_not_given": "TRUE/FALSE/NOT GIVEN",
        "yes_no_not_given": "YES/NO/NOT GIVEN (writer's claims)",
        "matching_headings": "matching headings to paragraphs",
        "match_paragraphs": "which paragraph contains the information",
        "completion": "sentence completion with a ____ gap",
        "short_answer": "short answer questions",
        "summary_choice": "choose the best summary (main idea)",
    }
    return names.get(qtype, qtype)


def recipe_line(recipe: list[tuple[str, int]]) -> str:
    parts = [f"{k} x{k} ({_type_label(q)})" for q, k in recipe if k > 0]
    return "; ".join(parts)


def build_generation_messages(article: Article, recipe: list[tuple[str, int]]) -> list[dict]:
    user = (
        "PASSAGE (paragraph labels in the left margin; refer to these labels)\n\n"
        f"{render_passage(article)}\n\n"
        "Produce exactly this mix of question groups:\n"
        f"{recipe_line(recipe)}\n\n"
        "Numbering: number all items 1..N consecutively across groups in that order; "
        "the summary_choice item is the last number.\n\n"
        "Respond with ONE JSON object of this shape (types not listed in the recipe are omitted):\n"
        + _JSON_SKELETON
        + "\n\nA fully worked example of the completion item shape:\n"
        + _MARKDOWN_EXAMPLE
    )
    return [
        {"role": "system", "content": GENERATION_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_repair_messages(
    generation_messages: list[dict], assistant_json: str, errors: list[str]
) -> list[dict]:
    report = "\n".join(f"- {e}" for e in errors)
    return [
        *generation_messages,
        {"role": "assistant", "content": assistant_json},
        {
            "role": "user",
            "content": (
                "Your JSON failed validation:\n"
                f"{report}\n\n"
                "Return the corrected complete JSON object only. Keep every other part identical."
            ),
        },
    ]


MARKER_SYSTEM = """You are an IELTS Academic Reading marker. You receive completion/short-answer \
items: each with its sentence or question, the reference answer, acceptable variants, the word \
limit (already checked client-side), and what the candidate wrote.

Decide for each: correct (same meaning as the reference answer), partial (right meaning, wrong \
form — e.g. misspelling that obscures the word, wrong plural, stray article), or incorrect. \
Misspellings are partial, never correct. Quote the passage words that settle the decision. \
Do not credit information taken from outside the passage.

Output exactly {"results": [...]} with one entry per input, in input order, each \
{"number": <int>, "verdict": "correct|partial|incorrect", "issues": ["<short string>", ...], \
"why": "<=40 words quoting the passage", "evidence": {"paragraph": "<letter>", \
"quote": "<verbatim passage words>"}, "teaching_note": "<=25 words tip"}. No extra keys."""


def build_marker_messages(
    article: Article, items: list[tuple], user_answers: dict[int, str]
) -> list[dict]:
    """items: list of (question, word_count_ok: bool). user_answers maps number -> raw text."""
    payload = []
    for question, word_count_ok in items:
        assert isinstance(question, CompletionItem | ShortAnswerItem)
        payload.append(
            {
                "number": question.number,
                "prompt": question.prompt,
                "answer_key": question.answer_key,
                "acceptable": question.acceptable,
                "word_limit": question.word_limit,
                "word_count_ok": word_count_ok,  # over_limit already decided: judge meaning only
                "candidate_wrote": user_answers.get(question.number) or "",
            }
        )
    user = (
        f"PASSAGE\n{render_passage(article)}\n\nITEMS TO MARK\n"
        + _fmt_marker_items(payload)
        + "\n\nMark each. Output the results JSON only."
    )
    return [
        {"role": "system", "content": MARKER_SYSTEM},
        {"role": "user", "content": user},
    ]


def _fmt_marker_items(payload: list[dict]) -> str:
    import json

    return "\n".join(json.dumps(item, ensure_ascii=False) for item in payload)


TUTOR_SYSTEM = """You are a warm, precise IELTS reading tutor commenting on ONE question. You \
receive the question, the candidate's answer, the verdict, the key, the explanation, and the \
evidence quote. Speak in 3-5 sentences (<= 120 words) of plain text (no markdown headings). \
Point at the exact words in the passage that decide the answer. If the answer was wrong, say in \
one sentence what to check differently next time (scope words like only/all/first, paraphrase \
recognition, Not Given logic). Never reveal or hint at answers to other questions. Answer any \
follow-up directly."""


def build_tutor_feedback_messages(
    article: Article,
    question,
    graded: GradedItem,
    tally: Tally,
    history: list[dict] | None = None,
) -> list[dict]:
    qdict: dict = {
        "number": question.number,
        "type": question.question_type,
        "prompt": question.prompt,
        "answer_key": question.answer_key,
        "explanation": graded.why or question.explanation,
        "evidence": [e.model_dump() for e in (graded.evidence or question.evidence)],
    }
    if isinstance(question, (MCQuestion, SummaryChoiceQuestion)):
        qdict["options"] = [o.text for o in question.options]
    elif isinstance(question, MatchHeadingsItem | MatchParagraphsItem):
        qdict["paragraph"] = question.paragraph_label
    elif isinstance(question, CompletionItem | ShortAnswerItem):
        qdict["word_limit"] = question.word_limit
        qdict["acceptable"] = question.acceptable
    context = (
        f"Passage excerpt:\n{render_passage(article)}\n\n"
        f"Question: {qdict}\n"
        f"Candidate wrote: {graded.user_raw!r}\n"
        f"Verdict: {graded.verdict} (marked by {graded.graded_by})\n"
        f"Issues: {graded.issues}\n"
        f"Teaching note: {graded.teaching_note}\n"
        f"Tally so far: {tally.correct}/{tally.total} correct"
        f", {tally.skipped} skipped, {tally.answered} answered.\n"
    )
    return [
        {"role": "system", "content": TUTOR_SYSTEM},
        {"role": "user", "content": context + "Comment on this answer now."},
        *(history or []),
    ]


def build_tutor_followup_messages(article: Article, question, history: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": TUTOR_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Passage excerpt:\n{render_passage(article)}\n\n"
                f"Current question: number {question.number} ({question.question_type}):\n"
                f"{question.prompt}"
            ),
        },
        *history,
    ]


COACH_SYSTEM = """You are an IELTS reading coach summarising what went wrong and what to \
practise. You receive the passage, the item-by-item results (including trap tags), and \
deterministic observations already made. Write 2-4 observations, each 1-3 sentences, each \
citing the question numbers it concerns. Focus on reading strategy: paraphrase recognition, \
scope and extent words, Not Given logic, skimming vs detail, instruction adherence. Do not \
praise generically; be specific. Output the CoachReport JSON only: \
{"observations": [{"kind": "<slug>", "title": "<short>", "detail": "<1-3 sentences>", \
"question_numbers": [..]}]}"""


def build_coach_messages(
    article: Article, reviews: list, observations: list[ReadingObservation]
) -> list[dict]:
    import json

    results = [
        {
            "number": r.number,
            "type": r.question_type,
            "verdict": r.verdict,
            "key": r.answer_key,
            "wrote": r.user_raw,
            "teaching_note": r.teaching_note,
        }
        for r in reviews
    ]
    det = [{"kind": o.kind, "title": o.title, "detail": o.detail} for o in observations]
    user = (
        f"Passage:\n{render_passage(article)}\n\n"
        f"Item results:\n{json.dumps(results, ensure_ascii=False)}\n\n"
        f"Deterministic observations already made:\n{json.dumps(det, ensure_ascii=False)}\n\n"
        "Return the CoachReport JSON only."
    )
    return [
        {"role": "system", "content": COACH_SYSTEM},
        {"role": "user", "content": user},
    ]
