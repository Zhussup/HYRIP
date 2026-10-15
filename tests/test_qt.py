"""Minimal Qt checks, offscreen.

QT_QPA_PLATFORM=offscreen keeps the suite headless; pytest-qt's qtbot drives the event
loop so QThread signals round-trip for real.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # keep the suite headless

pytest.importorskip("PySide6", reason="qt extras not installed")

from PySide6.QtCore import QTimer


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None:
        return QApplication(["hyrip-test"])
    return QApplication.instance()


def test_run_on_thread_round_trips(qtbot, qapp):
    from hyrip.qt.workers import run_on_thread

    results = []
    run_on_thread(
        lambda: (41, 1),
        lambda value: results.append(value),
        lambda error: results.append(error),
    )
    qtbot.waitUntil(lambda: len(results) > 0, timeout=5000)
    assert results == [(41, 1)]


def test_run_on_thread_reports_error(qtbot, qapp):
    from hyrip.errors import LlmError
    from hyrip.qt.workers import run_on_thread

    errors = []

    def boom():
        raise LlmError("the key is rejected")

    run_on_thread(boom, lambda value: errors.append(value), lambda error: errors.append(error))
    qtbot.waitUntil(lambda: len(errors) > 0, timeout=5000)
    assert "key is rejected" in errors[0]


def test_main_window_pages(qapp, store, config, fake_llm):
    from hyrip.qt.main_window import MainWindow
    from hyrip.service.session import SessionService

    service = SessionService(store, fake_llm, config)
    window = MainWindow(service)
    assert window.stack.count() == 5
    qapp.processEvents()
    for title in ("input", "article", "questions", "report", "history"):
        assert (
            window.stack.widget(["input", "article", "questions", "report", "history"].index(title))
            is not None
        )


def test_history_page_loads(qapp, store, config, fake_llm, raw_paper, article):
    from hyrip.llm.client import ChatResult
    from hyrip.qt.main_window import MainWindow
    from hyrip.service.session import SessionService
    from hyrip.store.repo import save_article

    fake_llm.responses.append(
        ChatResult(content=raw_paper.model_dump_json(), finish_reason="stop", latency_s=0.1)
    )
    service = SessionService(store, fake_llm, config)
    saved = save_article(store, article)
    session_id = service.start_and_generate(saved, mode="exam")
    service.start(session_id)

    window = MainWindow(service)
    window.history_page.load()
    assert window.history_page.table.rowCount() >= 1

    QTimer.singleShot(0, qapp.processEvents)
    qapp.processEvents()
