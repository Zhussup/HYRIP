"""Aggregations for `hyrip stats` (and the web/Qt history screens)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .db import Store


class TypeStat(BaseModel):
    question_type: str
    attempted: int
    correct: int

    @property
    def pct(self) -> float | None:
        if not self.attempted:
            return None
        return round(100.0 * self.correct / self.attempted, 1)


class SessionSummary(BaseModel):
    session_id: int
    mode: str
    status: str
    passage_title: str | None
    started_at: str
    elapsed_seconds: int | None
    score_correct: int | None
    score_total: int | None
    band_estimate: float | None


class Stats(BaseModel):
    per_type: list[TypeStat] = Field(default_factory=list)
    sessions: list[SessionSummary] = Field(default_factory=list)
    total_sessions: int = 0
    total_questions_answered: int = 0
    total_correct: int = 0


def compute_stats(store: Store, *, recent: int = 10) -> Stats:
    conn = store.connection()
    rows = conn.execute(
        "SELECT q.question_type AS t, COUNT(*) AS attempted,"
        " SUM(CASE WHEN a.verdict = 'correct' THEN 1 ELSE 0 END) AS correct"
        " FROM answers a JOIN questions q ON q.id = a.question_id"
        " WHERE a.verdict IS NOT NULL AND a.verdict != 'skipped'"
        " GROUP BY q.question_type ORDER BY q.question_type"
    ).fetchall()
    per_type = [
        TypeStat(question_type=row["t"], attempted=row["attempted"], correct=row["correct"])
        for row in rows
    ]

    session_rows = conn.execute(
        "SELECT s.id, s.mode, s.status, COALESCE(p.passage_title, a.title) AS passage_title,"
        " s.started_at, s.elapsed_seconds, r.score_correct, r.score_total, r.band_estimate"
        " FROM sessions s JOIN articles a ON a.id = s.article_id"
        " LEFT JOIN papers p ON p.id = s.paper_id"
        " LEFT JOIN reports r ON r.session_id = s.id"
        " WHERE s.status != 'generating'"
        " ORDER BY s.id DESC LIMIT ?",
        (recent,),
    ).fetchall()
    sessions = [
        SessionSummary(
            session_id=row["id"],
            mode=row["mode"],
            status=row["status"],
            passage_title=row["passage_title"],
            started_at=row["started_at"],
            elapsed_seconds=row["elapsed_seconds"],
            score_correct=row["score_correct"],
            score_total=row["score_total"],
            band_estimate=row["band_estimate"],
        )
        for row in session_rows
    ]
    totals = conn.execute(
        "SELECT COUNT(*) AS n, SUM(CASE WHEN verdict = 'correct' THEN 1 ELSE 0 END) AS correct"
        " FROM answers WHERE verdict IS NOT NULL AND verdict != 'skipped'"
    ).fetchone()
    return Stats(
        per_type=per_type,
        sessions=sessions,
        total_sessions=conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
        total_questions_answered=int(totals["n"] or 0),
        total_correct=int(totals["correct"] or 0),
    )
