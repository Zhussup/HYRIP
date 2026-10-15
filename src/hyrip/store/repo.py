"""Typed CRUD over the store; the ONLY module writing SQL for rows."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from ..errors import StateError
from ..models.article import Article
from ..models.grading import SessionStatus, UserAnswer
from ..models.questions import QuestionPaper, paragraph_label
from .db import Store, now_utc_iso


@dataclass
class SessionRow:
    id: int
    article_id: int
    paper_id: int | None
    mode: str
    status: SessionStatus
    model: str
    n_questions: int | None
    time_budget_min: int
    started_at: str
    finished_at: str | None
    elapsed_seconds: int | None


@dataclass
class AnswerRow:
    question_id: int
    number: int
    raw: str | None
    skipped: bool
    verdict: str | None
    graded_by: str | None
    verdict_json: str | None


# ---- articles ---------------------------------------------------------------


def save_article(store: Store, article: Article) -> int:
    article_id = (
        store.connection()
        .execute(
            "INSERT INTO articles(title, source, source_url, author, published,"
            " text, paragraphs_json, word_count, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (
                article.title,
                str(article.source),
                article.source_url,
                article.author,
                article.published,
                article.text,
                json.dumps(article.paragraphs),
                article.word_count,
                now_utc_iso(),
            ),
        )
        .lastrowid
    )
    store.connection().commit()
    article.id = int(article_id)
    return int(article_id)


def load_article(store: Store, article_id: int) -> Article:
    row = _fetch_one(store, "SELECT * FROM articles WHERE id = ?", (article_id,), "article")
    article = Article(
        id=row["id"],
        title=row["title"],
        source=row["source"],
        source_url=row["source_url"],
        author=row["author"],
        published=row["published"],
        text=row["text"],
        paragraphs=json.loads(row["paragraphs_json"]),
        word_count=row["word_count"],
    )
    article.paragraph_labels = [paragraph_label(i) for i in range(len(article.paragraphs))]
    return article


# ---- papers ------------------------------------------------------------------


def save_paper(store: Store, paper: QuestionPaper, *, raw_text: str | None = None) -> int:
    conn = store.connection()
    paper_id = conn.execute(
        "INSERT INTO papers(article_id, model, passage_title, payload_json, raw_json, meta_json,"
        " created_at)"
        " VALUES (?,?,?,?,?,?,?)",
        (
            paper.article_id,
            paper.meta.model if paper.meta else "",
            paper.title,
            paper.model_dump_json(),
            raw_text,
            paper.meta.model_dump_json() if paper.meta else None,
            now_utc_iso(),
        ),
    ).lastrowid
    conn.executemany(
        "INSERT INTO questions(paper_id, number, group_no, question_type, answer_key, payload_json)"
        " VALUES (?,?,?,?,?,?)",
        [
            (
                paper_id,
                q.number,
                q.group_no,
                q.question_type,
                q.answer_key,
                q.model_dump_json(),
            )
            for q in paper.questions
        ],
    )
    conn.commit()
    paper.id = int(paper_id)
    return int(paper_id)


def load_paper(store: Store, paper_id: int) -> QuestionPaper:
    row = _fetch_one(store, "SELECT * FROM papers WHERE id = ?", (paper_id,), "paper")
    paper = QuestionPaper.model_validate_json(row["payload_json"])
    paper.id = int(row["id"])
    return paper


def load_paper_raw(store: Store, paper_id: int) -> str | None:
    row = _fetch_one(store, "SELECT raw_json FROM papers WHERE id = ?", (paper_id,), "paper")
    return row["raw_json"]


# ---- sessions ----------------------------------------------------------------


def create_session(
    store: Store,
    article_id: int,
    paper_id: int | None = None,
    *,
    mode: str,
    model: str,
    time_budget_min: int,
    claims: bool = False,
    n_questions: int | None = None,
) -> int:
    session_id = (
        store.connection()
        .execute(
            "INSERT INTO sessions(article_id, paper_id, mode, status, model, n_questions,"
            " claims, time_budget_min, started_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (
                article_id,
                paper_id,
                mode,
                "generating",
                model,
                n_questions,
                int(claims),
                time_budget_min,
                now_utc_iso(),
            ),
        )
        .lastrowid
    )
    store.connection().commit()
    return int(session_id)


def set_session_paper(store: Store, session_id: int, paper_id: int) -> None:
    conn = store.connection()
    cursor = conn.execute("UPDATE sessions SET paper_id = ? WHERE id = ?", (paper_id, session_id))
    conn.commit()
    if cursor.rowcount == 0:
        _raise_if_missing_session(store, session_id)


def session_row_claims(store: Store, session_id: int) -> bool:
    row = _fetch_one(store, "SELECT claims FROM sessions WHERE id = ?", (session_id,), "session")
    return bool(row["claims"])


def update_answer_verdict(
    store: Store,
    session_id: int,
    question_id: int,
    *,
    verdict: str,
    graded_by: str,
    verdict_json: str,
) -> None:
    """Exam-mode finish: stamp grades over recorded raws without touching them."""
    conn = store.connection()
    cursor = conn.execute(
        "UPDATE answers SET verdict = ?, graded_by = ?, verdict_json = ?"
        " WHERE session_id = ? AND question_id = ?",
        (verdict, graded_by, verdict_json, session_id, question_id),
    )
    conn.commit()
    if cursor.rowcount == 0:
        from ..errors import StateError

        raise StateError(f"answer {question_id} of session {session_id} was never recorded")


def set_session_status(
    store: Store, session_id: int, status: SessionStatus, *, elapsed_seconds: int | None = None
) -> None:
    conn = store.connection()
    if status in ("report", "abandoned", "failed"):
        cursor = conn.execute(
            "UPDATE sessions SET status = ?, finished_at = ?,"
            " elapsed_seconds = COALESCE(?, elapsed_seconds)"
            " WHERE id = ?",
            (status, now_utc_iso(), elapsed_seconds, session_id),
        )
    else:
        cursor = conn.execute("UPDATE sessions SET status = ? WHERE id = ?", (status, session_id))
    conn.commit()
    if cursor.rowcount == 0:
        _raise_if_missing_session(store, session_id)


def load_session(store: Store, session_id: int) -> SessionRow:
    row = _fetch_one(store, "SELECT * FROM sessions WHERE id = ?", (session_id,), "session")
    return SessionRow(
        id=row["id"],
        article_id=int(row["article_id"]),
        paper_id=int(row["paper_id"]) if row["paper_id"] is not None else None,
        mode=row["mode"],
        status=row["status"],
        model=row["model"],
        n_questions=int(row["n_questions"]) if row["n_questions"] is not None else None,
        time_budget_min=row["time_budget_min"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        elapsed_seconds=row["elapsed_seconds"],
    )


def annotate_session_started(store: Store, session_id: int) -> None:
    """reading → active; the clock starts now (started_at reset so elapsed is clean)."""
    conn = store.connection()
    row = conn.execute("SELECT status FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if row is None:
        raise StateError(f"session {session_id} does not exist")
    if row["status"] not in ("reading",):
        raise StateError(f"cannot start a session in status {row['status']!r}")
    conn.execute(
        "UPDATE sessions SET status = 'active', started_at = ? WHERE id = ?",
        (now_utc_iso(), session_id),
    )
    conn.commit()


# ---- answers -------------------------------------------------------------------


def question_id_for(store: Store, paper_id: int, number: int) -> int:
    row = (
        store.connection()
        .execute("SELECT id FROM questions WHERE paper_id = ? AND number = ?", (paper_id, number))
        .fetchone()
    )
    if row is None:
        raise StateError(f"session paper has no question {number}")
    return int(row["id"])


def upsert_answer(
    store: Store,
    session_id: int,
    question_id: int,
    answer: UserAnswer,
    *,
    verdict: str | None = None,
    graded_by: str | None = None,
    verdict_json: str | None = None,
) -> None:
    normalized = (answer.raw or "").strip()
    conn = store.connection()
    conn.execute(
        "INSERT INTO answers(session_id, question_id, raw, normalized, skipped,"
        " verdict, graded_by, verdict_json, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?)"
        " ON CONFLICT(session_id, question_id) DO UPDATE SET"
        "   raw = excluded.raw, normalized = excluded.normalized, skipped = excluded.skipped,"
        "   verdict = COALESCE(excluded.verdict, verdict),"
        "   graded_by = COALESCE(excluded.graded_by, graded_by),"
        "   verdict_json = COALESCE(excluded.verdict_json, verdict_json)",
        (
            session_id,
            question_id,
            answer.raw,
            normalized,
            int(answer.skipped),
            verdict,
            graded_by,
            verdict_json,
            now_utc_iso(),
        ),
    )
    conn.commit()


def load_answers(store: Store, session_id: int) -> list[AnswerRow]:
    rows = (
        store.connection()
        .execute(
            "SELECT a.question_id, q.number, a.raw, a.normalized, a.skipped,"
            " a.verdict, a.graded_by, a.verdict_json"
            " FROM answers a JOIN questions q ON q.id = a.question_id"
            " WHERE a.session_id = ? ORDER BY q.number",
            (session_id,),
        )
        .fetchall()
    )
    return [
        AnswerRow(
            question_id=row["question_id"],
            number=row["number"],
            raw=row["raw"],
            skipped=bool(row["skipped"]),
            verdict=row["verdict"],
            graded_by=row["graded_by"],
            verdict_json=row["verdict_json"],
        )
        for row in rows
    ]


def count_session_answers(store: Store, session_id: int) -> int:
    return int(
        store.connection()
        .execute("SELECT COUNT(*) FROM answers WHERE session_id = ?", (session_id,))
        .fetchone()[0]
    )


def count_session_skipped(store: Store, session_id: int) -> int:
    return int(
        store.connection()
        .execute("SELECT COUNT(*) FROM answers WHERE session_id = ? AND skipped = 1", (session_id,))
        .fetchone()[0]
    )


# ---- tutor transcript -----------------------------------------------------------


def save_tutor_message(
    store: Store, session_id: int, question_id: int | None, role: str, content: str
) -> None:
    conn = store.connection()
    conn.execute(
        "INSERT INTO tutor_messages(session_id, question_id, role, content, created_at)"
        " VALUES (?,?,?,?,?)",
        (session_id, question_id, role, content, now_utc_iso()),
    )
    conn.commit()


def load_tutor_messages(store: Store, session_id: int) -> list[dict]:
    rows = (
        store.connection()
        .execute(
            "SELECT m.role, m.content, m.question_id, q.number AS question_number"
            " FROM tutor_messages m LEFT JOIN questions q ON q.id = m.question_id"
            " WHERE m.session_id = ? ORDER BY m.id",
            (session_id,),
        )
        .fetchall()
    )
    return [
        {
            "role": row["role"],
            "content": row["content"],
            "question_id": row["question_id"],
            "question_number": row["question_number"],
        }
        for row in rows
    ]


# ---- reports ----------------------------------------------------------------------


def save_report(store: Store, session_id: int, report) -> None:
    conn = store.connection()
    conn.execute(
        "INSERT INTO reports(session_id, score_correct, score_total, band_estimate,"
        " elapsed_seconds, report_json, created_at)"
        " VALUES (?,?,?,?,?,?,?)"
        " ON CONFLICT(session_id) DO UPDATE SET score_correct = excluded.score_correct,"
        "   score_total = excluded.score_total, band_estimate = excluded.band_estimate,"
        "   elapsed_seconds = excluded.elapsed_seconds, report_json = excluded.report_json",
        (
            session_id,
            report.score.raw_correct,
            report.score.total,
            report.score.band,
            report.elapsed_seconds,
            report.model_dump_json(),
            now_utc_iso(),
        ),
    )
    conn.commit()


def load_report(store: Store, session_id: int):
    row = (
        store.connection()
        .execute("SELECT report_json FROM reports WHERE session_id = ?", (session_id,))
        .fetchone()
    )
    if row is None:
        raise StateError(f"session {session_id} has no report yet")
    from ..models.grading import SessionReport

    return SessionReport.model_validate_json(row["report_json"])


# ---- history ----------------------------------------------------------------------


def list_sessions(store: Store, limit: int = 20) -> list[dict]:
    rows = (
        store.connection()
        .execute(
            "SELECT s.id, s.mode, s.status, s.started_at, s.elapsed_seconds, s.time_budget_min,"
            " r.score_correct, r.score_total, r.band_estimate,"
            " COALESCE(p.passage_title, a.title) AS passage_title"
            " FROM sessions s"
            " JOIN articles a ON a.id = s.article_id"
            " LEFT JOIN papers p ON p.id = s.paper_id"
            " LEFT JOIN reports r ON r.session_id = s.id"
            " ORDER BY s.id DESC LIMIT ?",
            (limit,),
        )
        .fetchall()
    )
    return [dict(row) for row in rows]


# ---- helpers ---------------------------------------------------------------------


def _fetch_one(store: Store, sql: str, params: tuple, what: str) -> sqlite3.Row:
    row = store.connection().execute(sql, params).fetchone()
    if row is None:
        raise StateError(f"{what} {params[0]} does not exist")
    return row


def _raise_if_missing_session(store: Store, session_id: int) -> None:
    row = (
        store.connection().execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
    )
    if row is None:
        raise StateError(f"session {session_id} does not exist")
