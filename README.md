# Hyrip

**H**ave **y**ou **r**ead **i**t **p**roperly? An IELTS Reading trainer that turns any article
(paste it, point at a file, or give it a BBC link) into an exam-style question paper, then marks
your answers and tells you *why* each one was right or wrong — quoting the words in the passage
that settled it, in the real IELTS question mix (including TRUE/FALSE/NOT GIVEN, heading
matching, and a final "which is the best main idea" check).

Powered by [Ollama Cloud](https://ollama.com) — one API key, OpenAI-compatible endpoint,
no local model install required.

## Three frontends over one engine

| Interface | Start it with | Needs |
|---|---|---|
| CLI | `hyrip read` · `hyrip url <link>` · `hyrip file article.txt` | base install |
| Web | `hyrip serve` → <http://127.0.0.1:8000> | `pip install -e '.[web]'` |
| Desktop | `hyrip gui` | `pip install -e '.[qt]'` |

All three share the same session engine, question bank rules, and SQLite history — pick the one
that suits the moment.

There are also ready-made launcher scripts (symlinked to `~/.local/bin` at setup, see
`scripts/`): `hyrip-cli` (bare = paste mode), `hyrip-web [= opens the browser]`,
`hyrip-qt` — each self-locates the venv and reads the API key from
`~/.config/hyrip/ollama_key` so they work from any shell.

## Setup

```bash
git clone <this repo> && cd hyrip
python3 -m venv .venv && .venv/bin/pip install -e '.[web,qt,dev]'   # or just -e .
export OLLAMA_API_KEY='…'        # from ollama.com → Settings → API keys
.venv/bin/hyrip doctor           # checks key, cloud reachability, database
```

Environment knobs (all optional — defaults in parentheses):

| Variable | Meaning |
|---|---|
| `OLLAMA_API_KEY` | Bearer key for `https://ollama.com/v1` |
| `HYRIP_MODEL` | default model (`gpt-oss:120b`) |
| `HYRIP_QUESTIONS` | default paper size (12) |
| `HYRIP_TEMPERATURE` / `HYRIP_MAX_TOKENS` / `HYRIP_TIMEOUT` | generation call tuning (0.4 / 6000 / 180) |
| `COACH_NOTES=0` | skip the extra one-call LLM "coach" summary in reports |
| `HYRIP_DATA_DIR` | move the SQLite database elsewhere |
| `HYRIP_BASE_URL` / `HYRIP_TAGS_URL` | point at a different OpenAI-compatible host |

Prefer [uv](https://docs.astral.sh/uv/)? `uv sync --all-extras` works the same way (the
project is plain hatchling; no code changes needed).

## Getting a question paper

```bash
hyrip file essay.txt --mode tutor            # tutor: feedback after every answer
hyrip url https://www.bbc.com/news/...       # exam: silent test, report at the end
cat essay.txt | hyrip read                   # paste mode; end input with a line saying EOF
```

Session options (all three commands):

| Flag | Meaning | Default |
|---|---|---|
| `--mode exam\|tutor` | exam marks at the end; tutor streams feedback + lets you chat per question | `exam` |
| `--model NAME` | any name from `hyrip models` | `gpt-oss:120b` |
| `--questions N` | 8–16 items | `12` |
| `--time MIN` | time budget shown in the footer/SPA | `20` |
| `--claims` | use YES/NO/NOT GIVEN phrasing instead of TRUE/FALSE/NOT GIVEN | off |
| `--save-raw PATH` | dump the raw LLM response for post-mortems | off |

If generation fails validation twice on the default model (some models are sloppy with
structural JSON), try `--model deepseek-v4-pro:0813` — it currently holds the best record
at producing clean papers.

The CLI flow: banner → generation (30–120 s; a repair round-trip sometimes runs) → the article
opens in a pager (q closes, v reopens later — reading time is part of the exam) → one question
at a time → report. Answer grammar inside the loop: `p`/`\` back, `s` skip, `f` finish, `v` view
the article again, `l` re-list headings, `q` quit (saved as abandoned), `?` help.

## The report

- **Raw score + band estimate** — from the Academic raw→band mapping, always shown with the
  disclaimer that one passage ≠ a real test.
- **Observations** — deterministic pattern rules over your answers: NOT-GIVEN confusion,
  paraphrase traps (scope/extent/time), skimming-vs-detail split, word-limit slips, and the
  main-idea verdict (too narrow / too broad). Cited by question number. A one-call model "coach"
  summary (on by default, `COACH_NOTES=0` to disable) can add observations of its own.
- **Next focus** — up to three concrete drills.
- **Per-question review** — your answer vs the key, why, and the verbatim evidence quote
  (an ⚠ mark means the model's quote wasn't found verbatim in the passage).

## What the engine enforces about the LLM

Generation and marking are contracts, not vibes — see [docs/CONTRACTS.md](docs/CONTRACTS.md)
for the exact JSON shapes. Highlights:

- closed-type questions (T/F/NG, headings, MC, paragraph matching) are marked **deterministically**
  against a normalization table — the model never judges your letter;
- word limits (NO MORE THAN THREE WORDS) are counted in code, never trusted to the model;
- open questions (completion/short answer) get one batched model-marking pass at finish;
  if no reliable mark comes back, a containment fallback grades it and says so;
- every answer must be provable by a quote from the passage; NOT GIVEN means the passage is silent.

## History & stats

Every session lands in a local SQLite database (`~/.local/share/hyrip/hyrip.db`,
`HYRIP_DATA_DIR` to move it). `hyrip history` lists and replays past reports (tutor transcripts
included), `hyrip stats` shows per-question-type accuracy and recent sessions.

Sessions never resume mid-flight in v1 — a quit (`q`) is *saved as abandoned* so it still shows
in history but can't be re-entered.

## Development

```bash
.venv/bin/ruff check src tests && .venv/bin/ruff format src tests
.venv/bin/pytest tests/ -q                  # Qt tests need a display, or QT_QPA_PLATFORM=offscreen
.venv/bin/pytest --cov=hyrip -q             # engine/models/store covered ≥85%
```

Layout: `src/hyrip/{models,llm,ingest,engine,store,service,cli,web,qt}` — models are
pydantic v2, the store is stdlib sqlite3 (per-thread connections, WAL, no ORM), the LLM client
is plain httpx. `docs/CONTRACTS.md` documents the JSON the model must produce.

## License

MIT