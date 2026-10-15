"""Qt worker plumbing: engine calls run on a QThread, results arrive via queued signals."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

_ACTIVE_THREADS: set = set()  # keeps QThread+worker alive until their event loop exits


class LlmWorker(QObject):
    """Runs one sync callable on its own thread; emits the result or an error message."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[[], Any]) -> None:
        super().__init__()
        self._fn = fn

    def run(self) -> None:  # slot connected to QThread.started
        try:
            self.finished.emit(self._fn())
        except Exception as exc:
            message = exc.user_message() if hasattr(exc, "user_message") else str(exc)
            self.failed.emit(message)


def run_on_thread(fn, on_done, on_error, parent=None):
    """Start `fn` on a fresh QThread; queue `on_done`/`on_error` back to the caller thread."""
    del parent  # callers keep their own references (SessionController.threads)
    worker = LlmWorker(fn)
    thread = QThread()
    entry = (thread, worker)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(on_done)
    worker.failed.connect(on_error)
    # standard teardown: stop the loop when the work lands, delete objects afterwards
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    worker.failed.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.finished.connect(lambda: _ACTIVE_THREADS.discard(entry))
    _ACTIVE_THREADS.add(entry)
    thread.start()
    return thread
