"""CLI layer: scripted run_session flows (exam + tutor), answer grammar, render checks."""

from __future__ import annotations

import io
import json
from collections.abc import Callable

import pytest
from rich.console import Console

from hyrip.cli.answers import Action, parse_answer
from hyrip.cli.session_ui import run_session
from hyrip.llm.client import ChatResult
from hyrip.models.grading import UserAnswer
from hyrip.service.session import SessionService
from hyrip.store.repo import save_article


@pytest.fixture
def service(store, config, fake_llm, article) -> SessionService:
    return SessionService(store, fake_llm, config)


@pytest.fixture
def saved_article(store, article) -> int:
    return save_article(store, article)


def ready_session(service, saved_article, fake_llm, raw_paper, **options) -> int:
    paper = raw_paper.model_dump_json()
    fake_llm.responses.append(ChatResult(content=paper, finish_reason="stop", latency_s=0.1))
    return service.start_and_generate(saved_article, **options)


@pytest.fixture
def buf_console():
    return Console(file=io.StringIO(), width=100, force_terminal=False)


def output_of(console: Console) -> str:
    return console.file.getvalue()


def script(*lines: str) -> Callable[[str], str]:
    """A scripted input_fn: pops one line per prompt; '' pads any extra prompts."""
    queue = list(lines)

    def _input(_prompt: str) -> str:
        return queue.pop(0) if queue else ""

    return _input


# ---- answer grammar ---------------------------------------------------------------


def test_answer_grammar():
    assert parse_answer("A").raw == "A"
    assert parse_answer("").kind == "value" and parse_answer("").raw == ""  # blank re-prompts
    assert parse_answer("p").kind == "back"
    assert parse_answer("\\").kind == "back"
    assert parse_answer("S").kind == "skip"
    assert parse_answer("f").kind == "finish"
    assert parse_answer("v").kind == "view"
    assert parse_answer("l").kind == "list"
    assert parse_answer("q").kind == "quit"
    assert parse_answer("?").kind == "help"
    # letters stay answers, including the reserved-for-TFNG 'b'
    assert parse_answer("b") == Action(kind="value", raw="b")
    assert parse_answer("TRUE").kind == "value"
    assert parse_answer("pakistan").kind == "value"  # only exact nav tokens


# ---- exam flow through run_session -------------------------------------------------


def test_run_session_exam_finishes_with_report(
    service, fake_llm, raw_paper, saved_article, buf_console
):
    session_id = ready_session(service, saved_article, fake_llm, raw_paper, mode="exam")
    service.start(session_id)
    # marker batch for the two open completion items at finish time
    fake_llm.responses.append(
        ChatResult(
            content=json.dumps(
                {
                    "results": [
                        {"number": 10, "verdict": "correct", "why": "figure matches"},
                        {"number": 11, "verdict": "partial", "issues": ["word form"]},
                    ]
                }
            ),
            finish_reason="stop",
        )
    )
    script_lines = [
        "A",  # Q1
        "C",  # Q2
        "TRUE",  # Q3
        "FALSE",  # Q4
        "ii",  # Q5
        "*",  # Q6 junk — recorded as an incorrect headings answer
        "",  # re-prompt for Q7 (blank)
        "iii",
        "H",  # Q8
        "D",  # Q9
        "30",  # Q10
        "s",  # skip Q11
        "B",  # Q12 main idea
        "f",
        "y",
    ]
    done = run_session(
        buf_console, service, session_id, article=None, input_fn=script(*script_lines)
    )
    assert done is True
    assert service.snapshot(session_id).status == "report"
    out = output_of(buf_console)
    assert "Question review" in out
    assert "band" in out
    rows = service.list_history()
    assert rows[0]["id"] == session_id and rows[0]["status"] == "report"


def test_run_session_quit_saves_abandoned(service, fake_llm, raw_paper, saved_article, buf_console):
    session_id = ready_session(service, saved_article, fake_llm, raw_paper, mode="exam")
    service.start(session_id)
    done = run_session(buf_console, service, session_id, article=None, input_fn=script("q"))
    assert done is False
    snap = service.snapshot(session_id)
    assert snap.status == "abandoned"
    assert "abandoned" in output_of(buf_console)


def test_run_session_eof_abandons(service, fake_llm, raw_paper, saved_article, buf_console):
    session_id = ready_session(service, saved_article, fake_llm, raw_paper, mode="exam")
    service.start(session_id)

    def raising(_prompt: str) -> str:
        raise EOFError

    done = run_session(buf_console, service, session_id, article=None, input_fn=raising)
    assert done is False
    assert service.snapshot(session_id).status == "abandoned"


def test_run_session_finish_needs_confirmation(
    service, fake_llm, raw_paper, saved_article, buf_console
):
    session_id = ready_session(service, saved_article, fake_llm, raw_paper, mode="exam")
    service.start(session_id)
    # say no to the finish confirm, answer Q1, then quit — still abandoned
    script_lines = ["f", "n", "A", "q"]
    done = run_session(
        buf_console, service, session_id, article=None, input_fn=script(*script_lines)
    )
    assert done is False
    snap = service.snapshot(session_id)
    assert snap.status == "abandoned" and snap.n_answered >= 1


# ---- tutor streaming through run_session -------------------------------------------


def test_run_session_tutor_streams_and_persists(
    service, fake_llm, raw_paper, saved_article, buf_console
):
    session_id = ready_session(service, saved_article, fake_llm, raw_paper, mode="tutor")
    service.start(session_id)
    fake_llm.streams.append(["Nice start — ", "Q1 checks the opening claim."])
    fake_llm.streams.append(["B is wrong because ", "the article never states it."])
    input_lines = ["A", "why is B wrong?", "", "f", "y"]
    done = run_session(
        buf_console, service, session_id, article=None, input_fn=script(*input_lines)
    )
    assert done is True
    out = output_of(buf_console)
    assert "Nice start" in out and "B is wrong" in out
    assert "(marked by deterministic)" in out
    transcript = service.transcript(session_id)
    roles = [m["role"] for m in transcript]
    assert roles.count("user") >= 2 and roles.count("assistant") >= 2
    # the stored user turn carries the tutor verdict for replay
    first_user = next(m for m in transcript if m["role"] == "user")
    assert "[answer" in first_user["content"]
    # the replay labels turns by QUESTION number (not the global row id) and does not
    # swallow "[answer ...]"-prefixed content as rich markup
    assert first_user["question_number"] == 1
    from hyrip.cli.render import print_transcript

    tbuf = io.StringIO()
    print_transcript(Console(file=tbuf, width=100), transcript)
    replay = tbuf.getvalue()
    assert "(Q1)" in replay
    assert "[answer" in replay  # literal, not eaten as rich markup


def test_run_session_tutor_llm_failure_degrades(
    service, fake_llm, raw_paper, saved_article, buf_console
):
    session_id = ready_session(service, saved_article, fake_llm, raw_paper, mode="tutor")
    service.start(session_id)
    # no streams queued → chat_stream yields empty → but feedback still recorded;
    # verdict still arrives deterministically
    session_id_answer = service.record_answer(session_id, UserAnswer(number=1, raw="A"))
    assert session_id_answer is not None
    input_lines = ["f", "y"]
    done = run_session(
        buf_console, service, session_id, article=None, input_fn=script(*input_lines)
    )
    assert done is True


# ---- main() entry point -------------------------------------------------------------


@pytest.fixture
def seeded_db(tmp_path, monkeypatch, config, fake_llm, raw_paper, article):
    """A generated session in the HYRIP_DATA_DIR that main() will open."""
    from hyrip.paths import db_path
    from hyrip.store.db import Store

    monkeypatch.setenv("HYRIP_DATA_DIR", str(tmp_path))
    store = Store(db_path())
    saved = save_article(store, article)
    fake_llm.responses.append(
        ChatResult(content=raw_paper.model_dump_json(), finish_reason="stop", latency_s=0.1)
    )
    return SessionService(store, fake_llm, config).start_and_generate(saved, mode="exam")


def test_main_history_lists_seeded_session(seeded_db, config, fake_llm, capsys):
    from hyrip.cli.app import main
    from hyrip.paths import db_path
    from hyrip.store.db import Store

    fake_llm.responses.append(
        ChatResult(
            content=json.dumps({"results": [{"number": 10, "verdict": "correct"}]}),
            finish_reason="stop",
        )
    )
    fresh = SessionService(Store(db_path()), fake_llm, config)
    fresh.start(seeded_db)
    fresh.finish(seeded_db)

    assert main(["history"]) == 0
    out = capsys.readouterr().out
    assert str(seeded_db) in out and "exam" in out and "report" in out


def test_main_stats_prints_accuracy(seeded_db, config, fake_llm, capsys):
    from hyrip.cli.app import main
    from hyrip.models.grading import UserAnswer
    from hyrip.paths import db_path
    from hyrip.store.db import Store

    fresh = SessionService(Store(db_path()), fake_llm, config)
    fresh.start(seeded_db)
    fresh.record_answer(seeded_db, UserAnswer(number=1, raw="A"))
    fresh.record_answer(seeded_db, UserAnswer(number=3, raw="TRUE"))
    fresh.record_answer(seeded_db, UserAnswer(number=10, skipped=True))
    fake_llm.responses.append(
        ChatResult(
            content=json.dumps({"results": [{"number": 10, "verdict": "correct"}]}),
            finish_reason="stop",
        )
    )
    fresh.finish(seeded_db)

    assert main(["stats"]) == 0
    out_value = capsys.readouterr().out
    assert "Accuracy by question type" in out_value
    assert str(seeded_db) in out_value


def test_main_history_unknown_id_fails(seeded_db, capsys):
    from hyrip.cli.app import main

    assert main(["history", "999"]) == 1
    assert "not in history" in capsys.readouterr().out


def test_main_bare_command_prints_help(capsys):
    from hyrip.cli.app import main

    assert main([]) == 0
