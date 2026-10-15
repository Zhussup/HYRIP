"""SessionService: the one engine API shared by CLI, web, and Qt.

State machine: generating → reading → active → grading → report (| abandoned | failed).
Answers persist at record time; exam verdicts at finish, tutor verdicts at submit.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

from ..config import Config
from ..engine import insights, tutor
from ..engine.generator import generate_paper
from ..engine.grader import (
    BatchMarker,
    grade_paper,
    make_batch_marker,
    make_single_marker,
)
from ..engine.report import build_report
from ..engine.views import public_view
from ..errors import ConfigError, GenerationError, LlmError, StateError
from ..ingest.file_source import article_from_file
from ..ingest.text_source import article_from_text
from ..ingest.url_source import article_from_url
from ..llm.client import LLMClient
from ..models.article import Article
from ..models.grading import GradedItem, SessionReport, Tally, UserAnswer
from ..models.questions import QuestionPaper
from ..store.db import Store
from ..store.repo import (
    SessionRow,
    annotate_session_started,
    count_session_answers,
    count_session_skipped,
    create_session,
    load_answers,
    load_article,
    load_paper,
    load_report,
    load_session,
    load_tutor_messages,
    question_id_for,
    save_article,
    save_paper,
    save_report,
    save_tutor_message,
    session_row_claims,
    set_session_paper,
    set_session_status,
    update_answer_verdict,
    upsert_answer,
)
from ..store.stats import Stats, compute_stats

TUTOR_HISTORY_TURNS = 6


@dataclass
class SessionSnapshot:
    id: int
    mode: str
    status: str
    model: str
    title: str | None
    n_questions: int | None
    n_answered: int
    n_skipped: int
    time_budget_min: int
    elapsed_seconds: int | None
    remaining_seconds: int | None


class SessionService:
    """Owns the store; frontends call this and render only."""

    def __init__(self, store: Store, client: LLMClient, config: Config):
        self.store = store
        self.client = client
        self.config = config

    # ---- article intake ------------------------------------------------------

    def create_article_from_text(self, text: str, *, title: str | None = None) -> int:
        return save_article(self.store, article_from_text(text, self.config, title=title))

    def create_article_from_file(self, path: str) -> int:
        return save_article(self.store, article_from_file(path, self.config))

    def create_article_from_url(self, url: str) -> int:
        return save_article(self.store, article_from_url(url, self.config))

    # ---- session lifecycle -----------------------------------------------------

    def create_session(
        self,
        article_id: int,
        *,
        mode: str = "exam",
        n_questions: int | None = None,
        model: str | None = None,
        claims: bool = False,
        time_budget_min: int | None = None,
    ) -> int:
        load_article(self.store, article_id)  # fail fast on a bad id
        return create_session(
            self.store,
            article_id,
            None,
            mode=mode,
            model=model or self.config.model,
            time_budget_min=time_budget_min or self.config.time_budget_min,
            claims=claims,
            n_questions=n_questions,
        )

    def generate_for_session(self, session_id: int) -> int:
        """Generate, store, and attach the paper; generating → reading."""
        row = load_session(self.store, session_id)
        if row.status != "generating":
            raise StateError(f"cannot generate in status {row.status!r}")
        article = self._article_for(row)  # sessions carry article_id before any paper
        try:
            outcome = generate_paper(
                self.client,
                article,
                config=self.config,
                n_questions=row.n_questions or self.config.n_questions_default,
                claims_mode=session_row_claims(self.store, session_id),
            )
        except (LlmError, GenerationError, ConfigError):
            set_session_status(self.store, session_id, "failed")
            raise
        paper_id = save_paper(self.store, outcome.paper, raw_text=outcome.raw_text)
        set_session_paper(self.store, session_id, paper_id)
        set_session_status(self.store, session_id, "reading")
        return paper_id

    def start_and_generate(self, article_id: int, **options) -> int:
        """Synchronous convenience for CLI/Qt: create session + generate in one call."""
        session_id = self.create_session(article_id, **options)
        self.generate_for_session(session_id)
        return session_id

    def start(self, session_id: int) -> None:
        """reading → active; the clock starts."""
        annotate_session_started(self.store, session_id)

    def abandon(self, session_id: int) -> None:
        """Mark active/reading sessions abandoned (the v1 no-resume rule)."""
        row = load_session(self.store, session_id)
        if row.status in ("reading", "active"):
            set_session_status(
                self.store, session_id, "abandoned", elapsed_seconds=self._elapsed(row)
            )

    def snapshot(self, session_id: int) -> SessionSnapshot:
        row = load_session(self.store, session_id)
        title, n = None, 0
        if row.paper_id is not None:
            paper = load_paper(self.store, row.paper_id)
            title, n = paper.title, len(paper.questions)
        elapsed = self._elapsed(row)
        remaining = max(0, row.time_budget_min * 60 - elapsed) if row.status == "active" else None
        return SessionSnapshot(
            id=row.id,
            mode=row.mode,
            status=row.status,
            model=row.model,
            title=title,
            n_questions=n or None,
            n_answered=count_session_answers(self.store, session_id),
            n_skipped=count_session_skipped(self.store, session_id),
            time_budget_min=row.time_budget_min,
            elapsed_seconds=elapsed,
            remaining_seconds=remaining,
        )

    # ---- answers ----------------------------------------------------------------

    def record_answer(self, session_id: int, answer: UserAnswer) -> GradedItem | None:
        """Persist one answer. Tutor sessions grade now and return the verdict."""
        self._require_active(session_id)
        paper = self._paper(session_id)
        if not (1 <= answer.number <= len(paper.questions)):
            raise StateError(f"question {answer.number} is outside this paper")
        question = paper.questions[answer.number - 1]
        question_id = question_id_for(self.store, paper.id, answer.number)
        upsert_answer(self.store, session_id, question_id, answer)
        if self._mode_of(session_id) != "tutor":
            return None
        graded = self._grade_tutor_now(session_id, paper, question, answer)
        save_tutor_message(
            self.store,
            session_id,
            question_id,
            "user",
            f"[answer {graded.verdict}] {answer.raw or ''}",
        )
        update_answer_verdict(
            self.store,
            session_id,
            question_id,
            verdict=graded.verdict,
            graded_by=graded.graded_by,
            verdict_json=graded.model_dump_json(),
        )
        return graded

    def view(self, session_id: int) -> dict:
        paper = self._paper(session_id)
        answered = {row.number for row in load_answers(self.store, session_id)}
        return public_view(paper, answered)

    def tally(self, session_id: int) -> Tally:
        paper = self._paper(session_id)
        rows = load_answers(self.store, session_id)
        return Tally(
            correct=sum(1 for r in rows if r.verdict == "correct"),
            answered=sum(1 for r in rows if not r.skipped),
            skipped=sum(1 for r in rows if r.skipped),
            total=len(paper.questions),
        )

    # ---- tutor streaming -----------------------------------------------------------

    def answer_verdict(self, session_id: int, number: int) -> GradedItem | None:
        """The stored verdict for one answer, if it has been graded (tutor sessions)."""
        row = next((r for r in load_answers(self.store, session_id) if r.number == number), None)
        if row is None or row.verdict_json is None:
            return None
        return GradedItem.model_validate_json(row.verdict_json)

    def tutor_feedback_stream(
        self, session_id: int, number: int, graded: GradedItem
    ) -> Iterator[str]:
        """Stream the tutor's comment; the assembled text is persisted when fully consumed."""
        self._require_active(session_id)
        paper = self._paper(session_id)
        question = paper.questions[number - 1]
        article = self._article_for(load_session(self.store, session_id))
        history = self._tutor_history(session_id, number)
        chunks: list[str] = []
        try:
            for chunk in tutor.feedback_stream(
                self.client,
                article,
                question,
                graded,
                self.tally(session_id),
                history,
            ):
                chunks.append(chunk)
                yield chunk
        finally:
            if chunks:
                save_tutor_message(
                    self.store,
                    session_id,
                    self._qid(paper, number),
                    "assistant",
                    "".join(chunks),
                )

    def tutor_followup_stream(self, session_id: int, number: int, text: str) -> Iterator[str]:
        self._require_active(session_id)
        paper = self._paper(session_id)
        question = paper.questions[number - 1]
        article = self._article_for(load_session(self.store, session_id))
        qid = self._qid(paper, number)
        save_tutor_message(self.store, session_id, qid, "user", text)
        history = self._tutor_history(session_id, number)
        chunks: list[str] = []
        try:
            for chunk in tutor.followup_stream(self.client, article, question, history):
                chunks.append(chunk)
                yield chunk
        finally:
            if chunks:
                save_tutor_message(self.store, session_id, qid, "assistant", "".join(chunks))

    # ---- finish ------------------------------------------------------------------

    def finish(self, session_id: int) -> SessionReport:
        row = load_session(self.store, session_id)
        if row.status == "report":
            return load_report(self.store, session_id)
        if row.status not in ("active", "grading"):
            raise StateError(f"cannot finish a session in status {row.status!r}")
        set_session_status(self.store, session_id, "grading")
        try:
            report = self._grade_all(session_id, row)
        except Exception:
            set_session_status(self.store, session_id, "active")  # resumable again
            raise
        save_report(self.store, session_id, report)
        set_session_status(self.store, session_id, "report", elapsed_seconds=report.elapsed_seconds)
        return report

    # ---- frontends ---------------------------------------------------------------

    def report_of(self, session_id: int) -> SessionReport:
        return load_report(self.store, session_id)

    def transcript(self, session_id: int) -> list[dict]:
        return load_tutor_messages(self.store, session_id)

    def list_history(self, limit: int = 20) -> list[dict]:
        from ..store.repo import list_sessions

        return list_sessions(self.store, limit=limit)

    def stats(self) -> Stats:
        return compute_stats(self.store)

    def list_models(self):
        return self.client.list_models()

    # ---- internals ------------------------------------------------------------------

    def _article_for(self, row: SessionRow) -> Article:
        return load_article(self.store, row.article_id)

    def _grade_tutor_now(
        self, session_id: int, paper: QuestionPaper, question, answer: UserAnswer
    ) -> GradedItem:
        row = load_session(self.store, session_id)
        article = self._article_for(row)
        marker = make_single_marker(self.client, article)
        return tutor.grade_immediately(
            paper,
            article,
            question,
            answer,
            mode="tutor",
            single_marker=marker,
        )

    def _mode_of(self, session_id: int) -> str:
        return load_session(self.store, session_id).mode

    def _require_active(self, session_id: int) -> None:
        status = load_session(self.store, session_id).status
        if status != "active":
            raise StateError(
                f"session {session_id} is {status!r}; answers are recorded only while active"
            )

    def _paper(self, session_id: int) -> QuestionPaper:
        row = load_session(self.store, session_id)
        if row.paper_id is None:
            raise StateError(f"session {session_id} has no paper yet")
        return load_paper(self.store, row.paper_id)

    def _tutor_history(self, session_id: int, number: int) -> list[dict]:
        paper = self._paper(session_id)
        qid = self._qid(paper, number)
        same = [m for m in load_tutor_messages(self.store, session_id) if m["question_id"] == qid]
        return [{"role": m["role"], "content": m["content"]} for m in same[-TUTOR_HISTORY_TURNS:]]

    def _qid(self, paper: QuestionPaper, number: int) -> int | None:
        if paper.id is None:
            return None
        try:
            return question_id_for(self.store, paper.id, number)
        except StateError:
            return None

    def _grade_all(self, session_id: int, row: SessionRow) -> SessionReport:
        paper = self._paper(session_id)
        article = self._article_for(row)
        rows = load_answers(self.store, session_id)
        answers: dict[int, UserAnswer] = {
            r.number: UserAnswer(number=r.number, raw=r.raw, skipped=r.skipped) for r in rows
        }
        marker = self._marker_for(article) if row.mode == "exam" else None
        graded = grade_paper(paper, answers, mode=row.mode, marker=marker, article=article)
        if row.mode == "tutor":  # submit-time marks are authoritative; restore them verbatim
            graded = self._merge_stored_tutor_marks(rows, graded)
        self._stamp_verdicts(session_id, paper, graded)
        elapsed = self._elapsed(row)
        report = build_report(paper, graded, elapsed_seconds=elapsed)
        if self.config.coach_notes:
            report.observations.extend(
                insights.coach_observations(
                    self.client, article, report.reviews, report.observations
                )
            )
        report.next_focus = insights.next_focus(report.observations)
        return report

    def _marker_for(self, article) -> BatchMarker:
        return make_batch_marker(self.client, article)

    def _merge_stored_tutor_marks(
        self, rows, graded: dict[int, GradedItem]
    ) -> dict[int, GradedItem]:
        for r in rows:
            if r.verdict_json:
                graded[r.number] = GradedItem.model_validate_json(r.verdict_json)
        return graded

    def _stamp_verdicts(
        self, session_id: int, paper: QuestionPaper, graded: dict[int, GradedItem]
    ) -> None:
        recorded = {r.number for r in load_answers(self.store, session_id)}
        for number in recorded & graded.keys():
            g = graded[number]
            question_id = question_id_for(self.store, paper.id, number)
            update_answer_verdict(
                self.store,
                session_id,
                question_id,
                verdict=g.verdict,
                graded_by=g.graded_by,
                verdict_json=g.model_dump_json(),
            )

    def _elapsed(self, row: SessionRow) -> int:
        if row.status in ("report", "abandoned", "failed") and row.elapsed_seconds is not None:
            return row.elapsed_seconds
        try:
            started = datetime.fromisoformat(row.started_at)
        except (ValueError, TypeError):
            return 0
        if started.tzinfo is None:
            started = started.replace(tzinfo=UTC)
        return max(0, int((datetime.now(UTC) - started).total_seconds()))


__all__ = ["SessionService", "SessionSnapshot"]
