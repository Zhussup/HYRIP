"""SessionService end-to-end on the fake LLM: state machine, both modes, finish, history."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from hyrip.errors import DataError, StateError
from hyrip.llm.client import ChatResult
from hyrip.models.grading import UserAnswer
from hyrip.service.session import SessionService
from hyrip.store.repo import load_session


@pytest.fixture
def service(store, config, fake_llm, article) -> SessionService:
    return SessionService(store, fake_llm, config)


@pytest.fixture
def saved_article(store, article) -> int:
    from hyrip.store.repo import save_article

    return save_article(store, article)


def paper_json(raw_paper) -> str:
    return raw_paper.model_dump_json()


def ready_service(service, saved_article, fake_llm, raw_paper, **options):
    fake_llm.responses.append(
        ChatResult(content=paper_json(raw_paper), finish_reason="stop", latency_s=0.1)
    )
    session_id = service.start_and_generate(saved_article, **options)
    return session_id


# ---- state machine ---------------------------------------------------------------


def test_full_exam_flow(service, fake_llm, raw_paper, saved_article):
    session_id = ready_service(service, saved_article, fake_llm, raw_paper, mode="exam")
    assert service.snapshot(session_id).status == "reading"
    service.start(session_id)
    assert service.snapshot(session_id).status == "active"

    exam_answer = service.record_answer(session_id, UserAnswer(number=1, raw="A"))
    assert exam_answer is None  # exam mode hides verdicts until finish
    service.record_answer(session_id, UserAnswer(number=3, raw="TRUE"))
    service.record_answer(session_id, UserAnswer(number=10, skipped=True))

    tally_before = service.tally(session_id)
    assert tally_before.answered == 2
    assert tally_before.skipped == 1

    # grading the open item needs one marker call
    fake_llm.responses.append(
        ChatResult(
            content=json.dumps(
                {"results": [{"number": 10, "verdict": "correct", "why": "figure matches"}]}
            ),
            finish_reason="stop",
        )
    )
    report = service.finish(session_id)
    assert report.score.total == 12
    assert report.score.raw_correct >= 2
    assert service.snapshot(session_id).status == "report"
    # history shows the finished session
    rows = service.list_history()
    assert rows[0]["id"] == session_id and rows[0]["status"] == "report"


def test_exam_finish_requires_active(service, fake_llm, raw_paper, saved_article):
    session_id = ready_service(service, saved_article, fake_llm, raw_paper, mode="exam")
    with pytest.raises(StateError, match="reading"):
        service.finish(session_id)


def test_record_requires_active(service, fake_llm, raw_paper, saved_article):
    session_id = ready_service(service, saved_article, fake_llm, raw_paper, mode="exam")
    with pytest.raises(StateError, match="reading"):
        service.record_answer(session_id, UserAnswer(number=1, raw="A"))


def test_answer_out_of_range_rejected(service, fake_llm, raw_paper, saved_article):
    session_id = ready_service(service, saved_article, fake_llm, raw_paper, mode="exam")
    service.start(session_id)
    with pytest.raises(StateError, match="outside"):
        service.record_answer(session_id, UserAnswer(number=13, raw="A"))


def test_generate_honors_session_question_count(
    service, fake_llm, raw_paper, saved_article, monkeypatch
):
    """--questions must reach the generator, not fall back to the config default."""
    import hyrip.service.session as session_module

    captured: dict = {}
    real_generate = session_module.generate_paper

    def spy(client, article_value, **kwargs):
        captured.update(kwargs)
        return real_generate(client, article_value, **kwargs)

    monkeypatch.setattr(session_module, "generate_paper", spy)

    fake_llm.responses.append(
        ChatResult(content=paper_json(raw_paper), finish_reason="stop", latency_s=0.1)
    )
    session_id = service.create_session(saved_article, mode="exam", n_questions=9)
    service.generate_for_session(session_id)
    assert captured["n_questions"] == 9
    assert load_session(service.store, session_id).n_questions == 9


def test_double_generate_rejected(service, fake_llm, raw_paper, saved_article):
    session_id = ready_service(service, saved_article, fake_llm, raw_paper, mode="exam")
    with pytest.raises(StateError, match="'reading'"):
        service.generate_for_session(session_id)


def test_generation_failure_marks_failed(service, fake_llm, saved_article):
    fake_llm.responses.append(ChatResult(content="junk", finish_reason="stop"))
    from hyrip.errors import GenerationError

    session_id = service.create_session(saved_article, mode="exam")
    with pytest.raises(GenerationError):
        service.generate_for_session(session_id)
    assert service.snapshot(session_id).status == "failed"


def test_config_error_also_marks_failed(service, fake_llm, saved_article):
    """A missing API key raises ConfigError — the session must not be stuck at generating."""
    from hyrip.errors import ConfigError

    def boom(*args, **kwargs):
        raise ConfigError("OLLAMA_API_KEY is not set.")

    fake_llm.chat = boom
    session_id = service.create_session(saved_article, mode="exam")
    with pytest.raises(ConfigError):
        service.generate_for_session(session_id)
    assert service.snapshot(session_id).status == "failed"


def test_key_stripping(service, fake_llm, raw_paper, saved_article):
    session_id = ready_service(service, saved_article, fake_llm, raw_paper, mode="exam")
    view = service.view(session_id)
    blob = json.dumps(view)
    assert "answer_key" not in blob and "explanation" not in blob
    assert "quote" not in blob
    mc = view["questions"][0]
    assert sorted(mc["options"]) == sorted(raw_paper.groups[0].options)
    bank = next(g["headings"] for g in view["groups"] if g["headings"])
    assert len(bank) == 8
    assert mc["question_type"] == "multiple_choice"
    completion = next(q for q in view["questions"] if q["question_type"] == "completion")
    assert completion["word_limit"] == 3
    assert view["answered"] == []


# ---- tutor mode -------------------------------------------------------------------


def test_tutor_flow_instant_verdict_and_stream(service, fake_llm, raw_paper, saved_article, store):
    session_id = ready_service(service, saved_article, fake_llm, raw_paper, mode="tutor")
    service.start(session_id)

    # submit-time marker call for the open question; closed items need no LLM
    fake_llm.responses.insert(
        0,
        ChatResult(
            content=json.dumps(
                {
                    "results": [
                        {
                            "number": 11,
                            "verdict": "partial",
                            "why": "nearly",
                            "teaching_note": "watch form",
                        }
                    ]
                }
            ),
            finish_reason="stop",
        ),
    )
    graded = service.record_answer(session_id, UserAnswer(number=11, raw="starve the pollinators"))
    assert graded is not None
    assert graded.verdict == "partial"  # tutor keeps partial
    assert graded.graded_by == "llm"

    closed = service.record_answer(session_id, UserAnswer(number=1, raw="A"))
    assert closed.verdict == "correct"
    assert closed.graded_by == "deterministic"

    fake_llm.streams = [["Nice,", " that", " works."]]
    chunks = list(service.tutor_feedback_stream(session_id, 1, closed))
    assert "".join(chunks) == "Nice, that works."

    fake_llm.streams = [["Because", " the key is B."]]
    follow = list(service.tutor_followup_stream(session_id, 1, "but why not A?"))
    assert "".join(follow) == "Because the key is B."

    transcript = service.transcript(session_id)
    assert [m["role"] for m in transcript] == ["user", "user", "assistant", "user", "assistant"]

    # finish reuses stored verdicts — no more marker calls in tutor mode
    report = service.finish(session_id)
    assert report.score.raw_correct >= 1
    saved = service.report_of(session_id)
    assert saved.score.raw_correct == report.score.raw_correct


def test_finish_idempotent(service, fake_llm, raw_paper, saved_article):
    session_id = ready_service(service, saved_article, fake_llm, raw_paper, mode="exam")
    service.start(session_id)
    fake_llm.responses.append(ChatResult(content="{}", finish_reason="stop"))
    report1 = service.finish(session_id)
    report2 = service.finish(session_id)
    assert report1 == report2


def test_elapsed_from_started_at(service, fake_llm, raw_paper, saved_article, store, monkeypatch):
    session_id = ready_service(service, saved_article, fake_llm, raw_paper, mode="exam")
    service.start(session_id)
    past = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    store.connection().execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?", (past, session_id)
    )
    store.connection().commit()
    snap = service.snapshot(session_id)
    assert snap.elapsed_seconds >= 290
    assert snap.remaining_seconds is not None and snap.remaining_seconds <= 20 * 60 - 290 + 5


def test_stats_and_history_round_trip(service, fake_llm, raw_paper, saved_article, store):
    session_id = ready_service(service, saved_article, fake_llm, raw_paper, mode="exam")
    service.start(session_id)
    service.record_answer(session_id, UserAnswer(number=1, raw="A"))
    fake_llm.responses.append(ChatResult(content="{}", finish_reason="stop"))
    service.finish(session_id)
    stats = service.stats()
    assert stats.sessions[0].session_id == session_id
    assert (
        service.list_history()[0]["passage_title"]
        == "Rooftop hives: the quiet boom in urban beekeeping"
    )


def test_article_intake_paths(service, tmp_path):
    session_a = service.create_article_from_text(
        "Rooftop bees arrive.\n\n"
        + "City hives grow quietly, and researchers measure yields.\n"
        + "word " * 130
    )
    assert session_a > 0
    file_path = tmp_path / "urban-bees.txt"
    file_path.write_text("City hives grow.\n\n" + "word " * 130, encoding="utf-8")
    imported = service.create_article_from_file(str(file_path))
    assert imported > 0
    from hyrip.store.repo import load_article

    assert load_article(service.store, imported).title == "City hives grow."
    with pytest.raises(DataError, match="No such file"):
        service.create_article_from_file(str(tmp_path / "nope.txt"))
