"""SQLite schema, applied as an ordered migration list. No ORM."""

from __future__ import annotations

SCHEMA_VERSION = 1

MIGRATIONS: list[str] = [
    # v1
    """CREATE TABLE articles(
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        source TEXT NOT NULL,
        source_url TEXT,
        author TEXT,
        published TEXT,
        text TEXT NOT NULL,
        paragraphs_json TEXT NOT NULL,
        word_count INTEGER NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE papers(
        id INTEGER PRIMARY KEY,
        article_id INTEGER NOT NULL REFERENCES articles(id),
        model TEXT NOT NULL,
        passage_title TEXT,
        payload_json TEXT NOT NULL,
        raw_json TEXT,
        meta_json TEXT,
        created_at TEXT NOT NULL
    );
    CREATE TABLE sessions(
        id INTEGER PRIMARY KEY,
        article_id INTEGER NOT NULL REFERENCES articles(id),
        paper_id INTEGER REFERENCES papers(id),
        mode TEXT NOT NULL CHECK(mode IN ('exam','tutor')),
        status TEXT NOT NULL CHECK(status IN
            ('generating','reading','active','grading','report','abandoned','failed')),
        model TEXT NOT NULL,
        n_questions INTEGER,
        claims INTEGER NOT NULL DEFAULT 0,
        time_budget_min INTEGER NOT NULL DEFAULT 20,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        elapsed_seconds INTEGER
    );
    CREATE TABLE questions(
        id INTEGER PRIMARY KEY,
        paper_id INTEGER NOT NULL REFERENCES papers(id),
        number INTEGER NOT NULL,
        group_no INTEGER NOT NULL,
        question_type TEXT NOT NULL,
        answer_key TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        UNIQUE(paper_id, number)
    );
    CREATE TABLE answers(
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES sessions(id),
        question_id INTEGER NOT NULL REFERENCES questions(id),
        raw TEXT,
        normalized TEXT,
        skipped INTEGER NOT NULL DEFAULT 0,
        verdict TEXT,
        graded_by TEXT,
        verdict_json TEXT,
        created_at TEXT NOT NULL,
        UNIQUE(session_id, question_id)
    );
    CREATE TABLE tutor_messages(
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES sessions(id),
        question_id INTEGER,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE reports(
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL UNIQUE REFERENCES sessions(id),
        score_correct INTEGER,
        score_total INTEGER,
        band_estimate REAL,
        elapsed_seconds INTEGER,
        report_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE schema_meta(
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    CREATE INDEX idx_sessions_started ON sessions(started_at);
    CREATE INDEX idx_answers_session ON answers(session_id);""",
]
