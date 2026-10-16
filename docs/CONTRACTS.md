# LLM contracts

Every JSON object the model produces is parsed by a pydantic v2 model with
`extra="forbid"` (`src/hyrip/models/llm_contracts.py`), so an extra/misspelled key fails
validation loudly instead of silently dropping data. This file documents the four active
contracts and the recovery paths; keep it in sync with the code when they change.

Call parameters at a glance:

| Call | Endpoint | Mode | Temperature | Max tokens |
|---|---|---|---|---|
| Paper generation | `POST /v1/chat/completions` | `response_format: json_object` | config (`HYRIP_TEMPERATURE`, default 0.4) | config (`HYRIP_MAX_TOKENS`, default 12000) |
| Recovery: rewrite (truncated/unparseable) | same | `json_object` | same | `HYRIP_MAX_TOKENS * 2` |
| Repair round-trip | same | same | same | same as config |
| Marker (batch / single) | same | `json_object` | 0.2 | same as config |
| Tutor feedback / follow-up | same (streamed) | plain text, **not** JSON | default | 600 (`TUTOR_MAX_TOKENS`) |
| Coach (optional, off by default) | same | `json_object` | default | same as config |

---

## 1. Paper generation

**Prompt**: `build_generation_messages(article, recipe)` — system prompt with 10 hard
item-writing rules + passage (`label. text` per paragraph) + recipe line ("multiple_choice x2 …")
+ the JSON skeleton below.

**Contract** — `RawGeneratedPaper` → `RawGroup` → `RawItem` / `RawOption` / `RawEvidence`:

```jsonc
{
  "passage_title": "<short title (optional, falls back to article title)>",
  "groups": [                                   // 2–7 groups
    {
      "question_type": "multiple_choice" | "true_false_not_given" | "yes_no_not_given"
                     | "matching_headings" | "match_paragraphs" | "completion"
                     | "short_answer" | "summary_choice",
      "instructions": "<IELTS rubric line>",
      "options": ["<full option A>", ...],     // multiple_choice only (shared bank)
      "headings": ["i. ...", ...],             // matching_headings only
      "items": [
        {
          "number": 1,                         // consecutive 1..N across all groups
          "prompt": "<paraphrased stem or statement>",
          "paragraph_label": "C",              // matching_headings / match_paragraphs only
          "answer_key": "A" | "TRUE|FALSE|NOT GIVEN" | "YES|NO|NOT GIVEN"
                      | "IV" | "<paragraph letter>" | "<<=word_limit words>",
          "evidence": [{"paragraph": "C", "quote": "<verbatim passage words>"}],
          "explanation": "<why the key is right>",
          "acceptable": ["<spelling variant>"], // completion / short_answer only
          "word_limit": 3,                      // completion / short_answer only (1–3)
          "options": [                          // summary_choice only
            {"text": "<full-sentence candidate main idea>", "trap_kind": null},
            {"text": "<too narrow>", "trap_kind": "narrow"},
            {"text": "<too broad / unsupported>", "trap_kind": "broad"}
          ]
        }
      ]
    }
  ]
}
```

**Post-parse validation** (`engine/validation.py`) — hard failures trigger **one** recovery
call, chosen by failure shape (`generate_paper`):

- **Output never parsed** (`parse_json_response` failed, or pydantic's `json_invalid` —
  structurally broken / token-cut JSON, the "Invalid JSON: EOF …" case): the model cannot
  see past its own truncation, so feedback has nothing actionable — the paper is rewritten
  **from scratch** with `HYRIP_MAX_TOKENS * 2`. If that also fails to parse (or is
  `finish_reason == "length"` again), `GenerationError` carries the hint to raise
  `HYRIP_MAX_TOKENS`.
- **Parsed but drifted** (domain hard errors, or schema `extra="forbid"`/type violations on
  a complete JSON): one repair round-trip (assistant JSON + validator errors fed back at the
  base budget); a second failure raises `GenerationError` — the article is kept.

Recovery hard errors:

- consecutive numbering 1..N;
- exactly one `summary_choice` group and it is last;
- letter / roman / paragraph-label keys exist in their banks;
- NOT GIVEN ⇔ `evidence: []`;
- `multiple_choice` options bank present; `matching_headings` headings present.

One drift shape is normalized instead of repaired: models persistently tag the *correct*
`summary_choice` option with a `trap_kind`. The key comes from `answer_key`, so
`check_raw_paper` strips `trap_kind` from the correct option client-side — no hard failure, no
wasted repair round-trip.

Soft checks (recorded as warnings, never discarded): evidence quotes verified verbatim
against paragraph text (`quote_verified=False` otherwise); item count within ±1 of the recipe.

If a response is `finish_reason == "length"`, or its JSON never parsed, the paper is
unrepairable by feedback — one from-scratch rewrite with `HYRIP_MAX_TOKENS * 2` instead.

**Recipe/mix**: default 12 items (2 MC, 2 TFNG or YNNG, 3 headings, 2 which-paragraph,
2 completion, 1 main-idea); `--claims` switches TFNG→YNNG; `match_paragraphs` is dropped
below 6 paragraphs; N is clamped to 8–16 (`HYRIP_QUESTIONS` `--questions`). The 14/16
recipes carry 7 groups (short_answer joins at 14), which is why the `groups` cap is 7.

## 2. Marker (open questions only)

Completion / short-answer items are the only ones the model judges — and only the *meaning*:
word limits are already enforced deterministically client-side. Closed types (letters,
TRUE/FALSE/NOT GIVEN, headings, paragraph matching, summary_choice) are graded by table-driven
normalization in `engine/grader.py`; the model never sees them.

Request items (one JSON object per line) carry `number, prompt, answer_key, acceptable,
word_limit, word_count_ok, candidate_wrote`. Response contract — `VerdictBatch`:

```jsonc
{ "results": [
    {
      "number": 4,
      "verdict": "correct" | "partial" | "incorrect",
      "issues": ["<short string>", ...],     // ≤ 2
      "why": "<=40 words, quoting the passage",
      "evidence": {"paragraph": "B", "quote": "<verbatim>"},
      "teaching_note": "<=25 words tip"
    } ] }
```

Semantics: `correct` = same meaning as reference; **`partial` = right meaning, wrong form**
(misspellings are always partial, never correct). `partial` ⇒ incorrect in exam mode,
"almost" in tutor wording.

Callers (`engine/grader.py`): `make_batch_marker` (exam mode — one call for all open items at
finish) and `make_single_marker` (tutor mode — one call on submit, needed for an instant
verdict). A missing/unparseable response falls back deterministically:
over-limit ⇒ incorrect, unanswered ⇒ skipped, missing batch entry ⇒
`GradedItem(graded_by="fallback")` so the report shows what did (and didn't) come from a model.

## 3. Tutor feedback & follow-up

One system prompt (`TUTOR_SYSTEM`): warm, precise, 3–5 sentences ≤ 120 words of **plain text**
(no markdown, no JSON), point at the exact decisive passage words, name the trap type
(scope / extent / time / paraphrase / Not Given logic), never reveal other questions' answers.
Context payload: the passage, the question dict (type/prompt/key/explanation/evidence — options
text for MC, word limit + acceptable for open items), `candidate wrote`, verdict + graded_by,
issues, teaching note, and a compact tally ("4/7 so far; Q4 was a scope trap"). Follow-up turns
append the last 6 question turns as chat history.

Transport: plain-token streaming through `client.chat_stream` (`max_tokens=600`); frontends
render tokens as they arrive. The Web SPA receives them as SSE frames:

```
data: {"token": "The"}         # one per token
data: {"error": "<message>"}   # terminal, on failure (HyripError.user_message())
data: {"done": true}           # terminal, on success
": heartbeat\n\n"              # comment frame every 15 s of idle
```

## 4. Coach report (optional extra call)

Enabled by default (`COACH_NOTES=1` in `config.py`; set `COACH_NOTES=0` to skip the call).
The deterministic observations (`engine/insights.py`) run first regardless; the coach may only
*add*. Contract — `CoachReport`:

```jsonc
{ "observations": [
    {"kind": "<slug>", "title": "<short>",
     "detail": "<1-3 sentences>", "question_numbers": [4, 7]} ] }
```

---

## Frontend-facing shapes (not LLM contracts, listed for completeness)

- `GradedItem` / `Tally` / `SessionReport` (`models/grading.py`, `engine/report.py`) — the
  persisted and re-rendered grader output.
- Web route payloads: `src/hyrip/web/schemas.py` (FastAPI request models) and the DTOs built in
  `routes.py`; keys are stripped from questions until a session is finished (`views.py`).
- Qt consumes the same engine objects directly — no serialization layer.