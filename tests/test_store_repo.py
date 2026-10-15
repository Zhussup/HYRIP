"""Store round-trips: articles, papers, sessions, answers, reports, tutor transcript."""

from __future__ import annotations

import pytest

from hyrip.errors import StateError
from hyrip.models.grading import BandEstimate, SessionReport, UserAnswer
from hyrip.models.questions import build_paper
from hyrip.store.db import Store
from hyrip.store.repo import (
    annotate_session_started,
    count_session_answers,
    count_session_skipped,
    create_session,
    list_sessions,
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
    set_session_status,
    upsert_answer,
)
from hyrip.store.stats import compute_stats


@pytest.fixture
def paper(store, raw_paper, article):
    save_article(store, article)
    built = build_paper(raw_paper, article=article)
    from hyrip.models.questions import GenerationMeta

    built.meta = GenerationMeta(model="test-model", latency_s=1.0)
    save_paper(store, built)
    return built


def test_article_roundtrip(store, raw_paper, article):
    article_id = save_article(store, article)
    loaded = load_article(store, article_id)
    assert loaded.title == article.title
    assert loaded.paragraphs == article.paragraphs
    assert loaded.paragraph_labels == article.paragraph_labels
    assert loaded.word_count == article.word_count


def test_paper_roundtrip(store, paper):
    loaded = load_paper(store, paper.id)
    assert loaded.model_dump() == paper.model_dump()


def test_session_lifecycle(store, paper):
    session_id = create_session(
        store,
        paper.article_id,
        paper.id,
        mode="exam",
        model="test-model",
        time_budget_min=20,
    )
    session = load_session(store, session_id)
    assert session.status == "generating"
    assert session.mode == "exam"

    set_session_status(store, session_id, "reading")
    annotate_session_started(store, session_id)
    assert load_session(store, session_id).status == "active"

    set_session_status(store, session_id, "abandoned", elapsed_seconds=42)
    rows = list_sessions(store)
    assert rows[0]["status"] == "abandoned"
    assert rows[0]["elapsed_seconds"] == 42


def test_annotate_requires_reading_state(store, paper):
    session_id = create_session(
        store, paper.article_id, paper.id, mode="exam", model="m", time_budget_min=20
    )
    with pytest.raises(StateError, match="generating"):
        annotate_session_started(store, session_id)


def test_answer_upsert(store, paper):
    session_id = create_session(
        store, paper.article_id, paper.id, mode="exam", model="m", time_budget_min=20
    )
    q_id = question_id_for(store, paper.id, 3)
    upsert_answer(store, session_id, q_id, UserAnswer(number=3, raw="TRUE"))
    upsert_answer(store, session_id, q_id, UserAnswer(number=3, raw="FALSE"))
    rows = load_answers(store, session_id)
    assert len(rows) == 1
    assert rows[0].raw == "FALSE"
    assert count_session_answers(store, session_id) == 1


def test_question_id_for_unknown_raises(store, paper):
    with pytest.raises(StateError, match="no question"):
        question_id_for(store, paper.id, 99)


def test_report_roundtrip(store, paper):
    session_id = create_session(
        store, paper.article_id, paper.id, mode="exam", model="m", time_budget_min=20
    )
    report = SessionReport(
        score=BandEstimate(raw_correct=9, total=12, pct=75.0, band=7.0),
        elapsed_seconds=900,
        reviews=[],
        observations=[],
        next_focus=[],
    )
    save_report(store, session_id, report)
    assert load_report(store, session_id).score.band == 7.0


def test_session_persists_question_count(store, paper):
    with_count = create_session(
        store,
        paper.article_id,
        paper.id,
        mode="exam",
        model="m",
        time_budget_min=20,
        n_questions=9,
    )
    without = create_session(
        store, paper.article_id, paper.id, mode="exam", model="m", time_budget_min=20
    )
    assert load_session(store, with_count).n_questions == 9
    assert load_session(store, without).n_questions is None
    assert count_session_skipped(store, with_count) == 0


def test_tutor_messages(store, paper):
    session_id = create_session(store, paper.id, mode="tutor", model="m", time_budget_min=20)
    q_id = question_id_for(store, paper.id, 1)
    save_tutor_message(store, session_id, q_id, "user", "Q1 answered TRUE")
    save_tutor_message(store, session_id, q_id, "assistant", "Right — paragraph B settles it.")
    msgs = load_tutor_messages(store, session_id)
    assert [m["role"] for m in msgs] == ["user", "assistant"]


def test_missing_rows_raise_state_error(store, paper):
    with pytest.raises(StateError):
        load_paper(store, 999)
    with pytest.raises(StateError):
        load_article(store, 999)
    with pytest.raises(StateError):
        load_report(store, 123)


def test_stats_round_trip(store, paper):
    session_id = create_session(
        store, paper.article_id, paper.id, mode="exam", model="m", time_budget_min=20
    )
    set_session_status(store, session_id, "report")
    q_id = question_id_for(store, paper.id, 1)
    upsert_answer(
        store,
        session_id,
        q_id,
        UserAnswer(number=1, raw="A"),
        verdict="correct",
        graded_by="deterministic",
    )
    q_id2 = question_id_for(store, paper.id, 2)
    upsert_answer(
        store,
        session_id,
        q_id2,
        UserAnswer(number=2, raw="B"),
        verdict="incorrect",
        graded_by="deterministic",
    )
    stats = compute_stats(store)
    assert stats.total_questions_answered == 2
    assert stats.total_correct == 1
    by_type = {t.question_type: t for t in stats.per_type}
    assert by_type["multiple_choice"].correct == 1
    assert by_type["multiple_choice"].pct == 50.0
    assert stats.sessions[0].session_id == session_id


def test_schema_migration_refuses_newer(store, tmp_path):
    import sqlite3

    from hyrip.errors import StoreError

    store.close()
    newer = tmp_path / "newer.db"
    conn = sqlite3.connect(newer)
    conn.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    conn.execute("INSERT INTO schema_migrations VALUES (999, 'now')")
    conn.commit()
    conn.close()
    with pytest.raises(StoreError, match="newer"):
        Store(newer)


def test_store_independent_of_hy_data(config, article, raw_paper, tmp_path):
    save_article(store := Store(tmp_path / "x.db"), article)
    assert store.path.name == "x.db"
