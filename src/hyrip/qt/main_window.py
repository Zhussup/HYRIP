"""Qt main window: five pages over the same SessionService the CLI and web use."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..models.article import Article
from ..models.grading import GradedItem, SessionReport, UserAnswer
from ..service.session import SessionService
from . import theme
from .workers import run_on_thread

# Strictly monochrome: states are glyph + typography, never colour, so the
# light and dark palettes (qt/theme.py) both render them by shape.
VERDICT_STYLE = {
    "correct": "font-weight:bold;",
    "partial": "font-style:italic;",
    "incorrect": "",
    "skipped": "",
}
VERDICT_GLYPH = {"correct": "✓", "partial": "≈", "incorrect": "✗", "skipped": "·"}

ROMANS = [None, "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"]


# ---- controller ------------------------------------------------------------------


class SessionController(QObject):
    """Runs every engine call on a worker thread; UI reacts to queued signals."""

    paperReady = Signal(int, object, dict, object)  # session_id, article, view, snapshot
    paperFailed = Signal(str)
    verdictReady = Signal(int, object, object)  # number, GradedItem | None, raw
    reportReady = Signal(object)  # SessionReport
    failed = Signal(str)
    tutorChunk = Signal(int, str)  # number, token
    tutorDone = Signal(int, object)  # number, assembled text

    def __init__(self, service: SessionService, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self.threads: list = []

    def _start(self, fn, on_done, on_error=...):
        thread = run_on_thread(fn, on_done, on_error if on_error is not ... else self.failed.emit)
        self.threads.append(thread)
        return thread

    # -- session lifecycle ----

    def generate(
        self,
        article_id: int,
        *,
        mode: str,
        model: str | None,
        n_questions: int | None,
        claims: bool,
        time_budget_min: int | None,
    ) -> None:
        def job():
            session_id = self.service.start_and_generate(
                article_id,
                mode=mode,
                model=model,
                n_questions=n_questions,
                claims=claims,
                time_budget_min=time_budget_min,
            )
            from ..store.repo import load_article

            return (
                session_id,
                load_article(self.service.store, article_id),
                self.service.view(session_id),
                self.service.snapshot(session_id),
            )

        self._start(job, self._paper_ready)

    def _paper_ready(self, bundle) -> None:
        session_id, article, view, snapshot = bundle
        self.paperReady.emit(session_id, article, view, snapshot)

    def record(self, session_id: int, answer: UserAnswer, raw: str) -> None:
        self._start(
            lambda: self.service.record_answer(session_id, answer),
            lambda graded: self.verdictReady.emit(answer.number, graded, raw),
        )

    def finish(self, session_id: int) -> None:
        self._start(lambda: self.service.finish(session_id), self.reportReady.emit)

    def models(self, on_done) -> None:
        self._start(self.service.list_models, on_done)

    # -- tutor streams ----

    def tutor_feedback(self, session_id: int, number: int, graded: GradedItem) -> None:
        streamer = _TutorStreamer(
            lambda: self.service.tutor_feedback_stream(session_id, number, graded),
            number,
            parent=self,
        )
        streamer.chunk.connect(self.tutorChunk)
        streamer.done.connect(self.tutorDone)
        streamer.failed.connect(self.failed)
        self.threads.append(streamer)
        streamer.start_stream()

    def tutor_followup(self, session_id: int, number: int, text: str) -> None:
        streamer = _TutorStreamer(
            lambda: self.service.tutor_followup_stream(session_id, number, text),
            number,
            parent=self,
        )
        streamer.chunk.connect(self.tutorChunk)
        streamer.done.connect(self.tutorDone)
        streamer.failed.connect(self.failed)
        self.threads.append(streamer)
        streamer.start_stream()


class _TutorStreamer(QObject):
    """Iterates a sync chunk generator on a QThread, emitting one signal per chunk."""

    chunk = Signal(int, str)
    done = Signal(int, object)
    failed = Signal(str)

    def __init__(self, factory: Callable, number: int, parent=None):
        super().__init__(parent)
        self._factory = factory
        self._number = number
        self.chunks: list[str] = []

    def start_stream(self) -> None:
        run_on_thread(self._own_iterator, self._emit_done, self._emit_failed)

    def _own_iterator(self):  # runs on the worker thread; emits to the UI queue
        for token in self._factory():
            self.chunks.append(token)
            self.chunk.emit(self._number, token)
        return None

    def _emit_done(self, _) -> None:
        self.done.emit(self._number, "".join(self.chunks))

    def _emit_failed(self, message: str) -> None:
        self.failed.emit(message)


# ---- pages --------------------------------------------------------------------------


class InputPage(QWidget):
    """Paste / URL / file + session options; hands off to the controller."""

    def __init__(self, controller: SessionController, parent=None):
        super().__init__(parent)
        self.controller = controller
        grid = QGridLayout(self)
        grid.setContentsMargins(18, 12, 18, 12)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["exam", "tutor"])
        self.count_spin = QSpinBox()
        self.count_spin.setRange(8, 16)
        self.count_spin.setValue(12)
        self.count_spin.setSpecialValueText("default")
        self.time_spin = QSpinBox()
        self.time_spin.setRange(10, 90)
        self.time_spin.setValue(20)
        self.model_combo = QComboBox()
        self.claims_check = QCheckBox("YES/NO/NOT GIVEN phrasing")
        self.model_combo.addItem("default", None)

        options_row = QGroupBox("Session")
        opt_layout = QGridLayout(options_row)
        opt_layout.addWidget(QLabel("mode"), 0, 0)
        opt_layout.addWidget(self.mode_combo, 0, 1)
        opt_layout.addWidget(QLabel("questions"), 1, 0)
        opt_layout.addWidget(self.count_spin, 1, 1)
        opt_layout.addWidget(QLabel("budget (min)"), 0, 2)
        opt_layout.addWidget(self.time_spin, 0, 3)
        opt_layout.addWidget(QLabel("model"), 1, 2)
        opt_layout.addWidget(self.model_combo, 1, 3)
        opt_layout.addWidget(self.claims_check, 2, 0, 1, 4)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText("Paste the article text here…")
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("https://www.bbc.com/…")
        file_row = QHBoxLayout()
        file_edit = QLineEdit()
        file_edit.setReadOnly(True)
        self.file_edit = file_edit
        pick = QPushButton("Choose file…")
        pick.clicked.connect(self._pick_file)

        go = QPushButton("Write questions")
        go.clicked.connect(self._go)

        grid.addWidget(QLabel("<b>hyrip</b> — have you read it properly?"), 0, 0, 1, 2)
        grid.addWidget(options_row, 1, 0, 1, 2)
        grid.addWidget(self.text_edit, 2, 0, 3, 2)
        grid.addWidget(QLabel("or a web page:"), 5, 0)
        grid.addWidget(self.url_edit, 5, 1)
        file_row.addWidget(pick)
        file_row.addWidget(file_edit, 1)
        grid.addLayout(file_row, 6, 0, 1, 2)
        grid.addWidget(go, 7, 1)
        self._load_models()

    def _load_models(self):
        self.controller.models(lambda models: self._fill_models([m.name for m in models]))

    def _fill_models(self, names: list[str]):
        self.model_combo.addItem("default", None)
        for name in names:
            self.model_combo.addItem(name, name)

    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Article file", "", "Articles (*.txt *.md)")
        if path:
            self.file_edit.setText(path)

    def _go(self):
        text = self.text_edit.toPlainText().strip()
        url = self.url_edit.text().strip()
        path = self.file_edit.text().strip()
        try:
            if text:
                article_id = self.controller.service.create_article_from_text(text)
            elif path:
                article_id = self.controller.service.create_article_from_file(path)
            elif url:
                article_id = self.controller.service.create_article_from_url(url)
            else:
                QMessageBox.warning(self, "hyrip", "Paste an article, pick a file, or give a URL.")
                return
        except Exception as exc:
            QMessageBox.warning(
                self, "hyrip", exc.user_message() if hasattr(exc, "user_message") else str(exc)
            )
            return
        self.controller.generate(
            article_id,
            mode=self.mode_combo.currentText(),
            model=self.model_combo.currentData(),
            n_questions=(
                self.count_spin.value()
                if self.count_spin.value() != self.count_spin.minimum()
                else None
            ),
            claims=self.claims_check.isChecked(),
            time_budget_min=self.time_spin.value(),
        )


class ArticlePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.title = QLabel()
        self.title.setWordWrap(True)
        self.meta = QLabel()
        self.body = QTextBrowser()
        self.body.setOpenExternalLinks(False)
        go = QPushButton("I've read it — start answering")
        layout.addWidget(self.title)
        layout.addWidget(self.meta)
        layout.addWidget(self.body, 1)
        layout.addWidget(go, 0, Qt.AlignRight)
        self.start_button = go


class QuestionsPage(QWidget):
    answerCommitted = Signal(int, str, bool)  # number, raw, skipped

    def __init__(self, controller: SessionController, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.view = None
        self.article = None
        self.idx = 0
        self.session_id = None
        self._drafts = {}
        self.snapshot = None
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        self.group_label = QLabel()
        self.instructions = QLabel()
        self.instructions.setWordWrap(True)
        left = QVBoxLayout()
        left.addWidget(self.group_label)
        left.addWidget(self.instructions)
        top.addLayout(left, 1)
        layout.addLayout(top)

        self.headings = QTextBrowser()
        self.headings.setMaximumHeight(130)
        layout.addWidget(self.headings)

        self.qarea = QGroupBox()
        self.q_layout = QVBoxLayout(self.qarea)
        layout.addWidget(self.qarea)

        self.tutor = QTextBrowser()
        self.tutor.setVisible(False)
        layout.addWidget(self.tutor)
        self.followup_edit = QLineEdit()
        self.followup_edit.setPlaceholderText("Ask the tutor anything about this question…")
        self.followup_edit.setVisible(False)
        layout.addWidget(self.followup_edit)
        self.followup_edit.returnPressed.connect(self._send_followup)

        nav = QHBoxLayout()
        self.prev_btn = QPushButton("‹ Prev")
        self.next_btn = QPushButton("Next ›")
        self.skip_btn = QPushButton("Skip")
        self.finish_btn = QPushButton("Finish")
        quit_btn = QPushButton("quit saving")
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        nav.addWidget(self.skip_btn)
        nav.addWidget(self.finish_btn)
        nav.addWidget(quit_btn, 0, Qt.AlignRight)
        layout.addLayout(nav)

        self.jump_row = QHBoxLayout()
        layout.addLayout(self.jump_row)

    def set_view(self, session_id: int, article: Article, view: dict, snapshot) -> None:
        self.session_id = session_id
        self.article = article
        self.view = view
        self.snapshot = snapshot
        self.idx = 0
        self._drafts = {}
        self.render()

    def render(self):
        view = self.view
        question = view["questions"][self.idx]
        group = next(g for g in view["groups"] if g["group_no"] == question["group_no"])
        self.group_label.setText(f"<b>Questions {group['group_no']}</b>")
        self.instructions.setText(group["instructions"])
        if group.get("headings"):
            self.headings.setPlainText("\n".join(group["headings"]))
            self.headings.setVisible(True)
        else:
            self.headings.setVisible(False)

        # The group box and its layout must survive every render — a QVBoxLayout dies with
        # the widget it manages, so throwing the box away per question (and reusing the
        # layout of the detached box) crashes shiboken on the next render. The box is a
        # permanent shell; only its children are rebuilt.
        while self.q_layout.count():
            item = self.q_layout.takeAt(0)
            if widget := item.widget():
                widget.deleteLater()
        self.qarea.setTitle(f"Q{question['number']}")
        self._radio_group = QButtonGroup()  # unparented: replaced (and collected) per render

        saved = self._drafts.get(question["number"], "")
        prompt = QLabel(question["prompt"])
        prompt.setWordWrap(True)
        self.q_layout.addWidget(prompt)
        input_widget = self._build_input(question, saved, self.q_layout)
        self._current_input = input_widget
        self.q_layout.setAlignment(Qt.AlignTop)

        self.prev_btn.setEnabled(self.idx > 0)
        self.tutor.clear()
        self.tutor_done_text = ""
        if self.snapshot.mode == "tutor":
            self.followup_edit.setVisible(True)
        else:
            self.followup_edit.setVisible(False)

    def _build_input(self, question: dict, saved: str, parent_layout: QVBoxLayout):
        n = question["number"]
        if question["question_type"] in ("multiple_choice", "summary_choice"):
            return self._radio_options(
                question, zip("ABCD", question["options"], strict=False), saved, parent_layout
            )
        if question["question_type"] == "true_false_not_given":
            return self._radio_options(
                question, [(v, v) for v in ("TRUE", "FALSE", "NOT GIVEN")], saved, parent_layout
            )
        if question["question_type"] == "yes_no_not_given":
            return self._radio_options(
                question, [(v, v) for v in ("YES", "NO", "NOT GIVEN")], saved, parent_layout
            )
        if question["question_type"] == "matching_headings":
            note = QLabel(f"paragraph {question['paragraph_label']} — answer with a numeral")
            parent_layout.addWidget(note)
            return self._combo(self._heading_pairs(), saved, parent_layout)
        if question["question_type"] == "match_paragraphs":
            pairs = [(label, label) for label in self.article.paragraph_labels]
            return self._combo(pairs, saved, parent_layout)
        if "word_limit" in question:
            hint = QLabel(f"No more than {question['word_limit']} words.")
            parent_layout.addWidget(hint)
        edit = QLineEdit()
        edit.setText(saved)
        parent_layout.addWidget(edit)
        if not self._drafts.get(n):
            self._drafts[n] = saved
        return edit

    def _heading_pairs(self):
        group = next(
            g
            for g in self.view["groups"]
            if g["group_no"] == self.view["questions"][self.idx]["group_no"]
        )
        pairs = []
        for i, text in enumerate(group["headings"] or []):
            roman = ROMANS[i + 1] if i + 1 < len(ROMANS) else str(i + 1)
            pairs.append((roman, f"{roman} — {text}"))
        return pairs

    def _radio_options(self, question, pairs, saved, parent_layout):
        for value, text in pairs:
            radio = QRadioButton(f"{value}) {text}" if text != value else value)
            if saved and saved.strip().upper() == value.upper():
                radio.setChecked(True)
            radio.value = value
            self._radio_group.addButton(radio)
            parent_layout.addWidget(radio)
        return self._radio_group

    def _combo(self, pairs, saved, parent_layout):
        from PySide6.QtWidgets import QComboBox

        combo = QComboBox()
        combo.addItem("— choose —", "")
        for value, text in pairs:
            combo.addItem(text, value)
        if saved:
            for i in range(combo.count()):
                if combo.itemData(i) == saved.strip().upper():
                    combo.setCurrentIndex(i)
        parent_layout.addWidget(combo)
        return combo

    def current_answer(self) -> tuple[str, bool]:
        """(raw, skipped) from the current input widget."""
        widget = getattr(self, "_current_input", None)
        if isinstance(widget, QButtonGroup):
            button = widget.checkedButton()
            return (button.value, False) if button else ("", False)
        if isinstance(widget, QComboBox):
            return widget.currentData() or "", False
        edit = widget
        return edit.text(), not edit.text().strip()

    def next_index(self):
        return min(self.idx + 1, len(self.view["questions"]) - 1)

    def prev_index(self):
        return max(self.idx - 1, 0)

    def set_tutor(self, verdict_line: str | None) -> None:
        self.tutor.setVisible(True)
        if verdict_line:
            self.tutor.setHtml(verdict_line)

    def append_tutor(self, token: str) -> None:
        self.tutor_done_text += token
        self.tutor.setHtml(f"<pre style='white-space:pre-wrap'>{self.tutor_done_text}</pre>")

    def _send_followup(self) -> None:
        text = self.followup_edit.text().strip()
        if not text or not self.session_id:
            return
        self.followup_edit.clear()
        question = self.view["questions"][self.idx]
        self.tutor_done_text = ""
        self.controller.tutor_followup(self.session_id, question["number"], text)


class ReportPage(QWidget):
    def __init__(self, controller: SessionController, parent=None):
        super().__init__(parent)
        self.controller = controller
        layout = QVBoxLayout(self)
        self.score = QLabel()
        self.score.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.meta = QLabel()
        self.meta.setWordWrap(True)
        self.body = QTextBrowser()
        go = QPushButton("Try another article")
        layout.addWidget(self.score)
        layout.addWidget(self.meta)
        layout.addWidget(self.body, 1)
        layout.addWidget(go, 0, Qt.AlignRight)

    def show_report(self, report: SessionReport, *, mode: str) -> None:
        self.score.setText(
            f"{report.score.raw_correct}/{report.score.total}  —  band ≈ {report.score.band:.1f}"
        )
        minutes = report.elapsed_seconds // 60
        self.meta.setText(
            f"{minutes} min{'' if mode == 'exam' else ' · tutor'} — {report.score.disclaimer}"
        )
        lines = []
        if report.observations:
            lines.append("<b>What your reading did</b>")
            for o in report.observations:
                lines.append(
                    f"<b>{o.title}</b> "
                    f"<span>(Q{', Q'.join(map(str, o.question_numbers))})</span><br>{o.detail}"
                )
        if report.next_focus:
            lines.append("<b>Next focus</b>")
            lines.extend(f"• {item}" for item in report.next_focus)
        lines.append("<hr><b>Question review</b>")
        for review in report.reviews:
            style = VERDICT_STYLE.get(review.verdict, "")
            glyph = VERDICT_GLYPH.get(review.verdict, "·")
            lines.append(
                f"<p><span style='{style}'>{glyph} {review.verdict.upper()}</span> "
                f"Q{review.number} <i>{review.question_type}</i><br>"
                f"you: {review.user_raw or '(skipped)'} · "
                f"answer: {review.answer_key}<br>"
            )
            if review.why:
                lines[-1] += f"why: {review.why}<br>"
            if review.evidence:
                ev = review.evidence[0]
                mark = "✓" if ev.quote_verified else "≈"
                lines[-1] += f"evidence {mark} ({ev.paragraph}): <i>“{ev.quote}”</i>"
            lines[-1] += "</p>"
        self.body.setHtml("<br>".join(lines))


class HistoryPage(QWidget):
    def __init__(self, controller: SessionController, parent=None):
        super().__init__(parent)
        self._service = None
        layout = QVBoxLayout(self)
        self.stats_label = QLabel("No data yet.")
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["id", "mode", "status", "score", "band", "elapsed"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.load)
        layout.addWidget(self.stats_label)
        layout.addWidget(QLabel("History"))
        layout.addWidget(self.table, 1)
        layout.addWidget(QLabel("Band trend"))
        self.spark = QTextBrowser()
        self.spark.setMaximumHeight(120)
        layout.addWidget(self.spark)
        layout.addWidget(refresh)

    def load(self):
        if self._service is None:
            return
        rows = self._service.list_history()
        stats = self._service.stats()
        self.stats_label.setText(
            f"sessions: {stats.total_sessions} · answers: {stats.total_questions_answered}"
            f" · correct: {stats.total_correct}"
        )
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            values = [
                str(row["id"]),
                row["mode"],
                row["status"],
                f"{row['score_correct']}/{row['score_total']}"
                if row["score_correct"] is not None
                else "—",
                f"{row['band_estimate']:.1f}" if row["band_estimate"] is not None else "—",
                (lambda m, s: f"{m:02d}:{s:02d}")(*divmod(row["elapsed_seconds"], 60))
                if row["elapsed_seconds"] is not None
                else "—",
            ]
            for col, value in enumerate(values):
                self.table.setItem(i, col, QTableWidgetItem(value))
        bands = [row["band_estimate"] for row in rows if row["band_estimate"] is not None]
        blocks = ["▁", "▂", "▃", "▄", "▅", "▆", "▇"]
        bars = "".join(
            blocks[max(0, min(int((band - 3.5) * 3), len(blocks) - 1))] for band in bands
        )
        self.spark.setHtml(
            f"<pre style='font-size:22px'>{bars}</pre>"
            "<small>band estimates, oldest → newest</small>"
        )

    def set_service(self, service: SessionService) -> None:
        self._service = service
        self.load()


class MainWindow(QMainWindow):
    def __init__(self, service: SessionService):
        super().__init__()
        self.service = service
        self.controller = SessionController(service, parent=self)
        self.setWindowTitle("hyrip — have you read it properly?")
        self.resize(880, 640)

        self.stack = QStackedWidget()
        self.input_page = InputPage(self.controller)
        self.article_page = ArticlePage()
        self.questions_page = QuestionsPage(self.controller)
        self.report_page = ReportPage(self.controller)
        self.history_page = HistoryPage(self.controller)
        self.history_page.set_service(service)
        for page in (
            self.input_page,
            self.article_page,
            self.questions_page,
            self.report_page,
            self.history_page,
        ):
            self.stack.addWidget(page)
        self.setCentralWidget(self.stack)

        self.controller.paperReady.connect(self._on_paper)
        self.controller.paperFailed.connect(self._on_failed)
        self.controller.failed.connect(self._on_failed)
        self.controller.verdictReady.connect(self._on_verdict)
        self.controller.reportReady.connect(self._on_report)
        self.controller.tutorChunk.connect(self._on_tutor_chunk)
        self.controller.tutorDone.connect(self._on_tutor_done)

        self.article_page.start_button.clicked.connect(self._start_answering)
        self.questions_page.next_btn.clicked.connect(self._go_next)
        self.questions_page.prev_btn.clicked.connect(self._go_prev)
        self.questions_page.skip_btn.clicked.connect(self._skip)
        self.questions_page.finish_btn.clicked.connect(self._finish)

        self._theme_actions: dict[str, QAction] = self._build_theme_menu()

    # -- theme menu (system / light / dark, persisted) ----

    def _build_theme_menu(self) -> dict[str, QAction]:
        menu = self.menuBar().addMenu("theme")
        group = QActionGroup(self)
        group.setExclusive(True)
        actions = {}
        for pref, label in (
            (theme.SYSTEM, "system — follow the desktop"),
            (theme.LIGHT, "light"),
            (theme.DARK, "dark"),
        ):
            action = QAction(label, self, checkable=True)
            action.triggered.connect(lambda checked=False, p=pref: self._set_theme(p))
            group.addAction(action)
            menu.addAction(action)
            actions[pref] = action
        pref = theme.stored_preference()
        actions[pref].setChecked(True)
        return actions

    def _set_theme(self, pref: str) -> None:
        theme.set_preference(pref)
        theme.apply(pref, QApplication.instance())
        self._theme_actions[pref].setChecked(True)

    # -- navigation ----

    def go(self, page: QWidget) -> None:
        self.stack.setCurrentWidget(page)

    # -- controller reactions ----

    def _on_paper(self, session_id, article, view, snapshot) -> None:
        self.session_id = session_id
        self.article = article
        self.questions_page.set_view(session_id, article, view, snapshot)
        self.article_page.title.setText(f"<b>{article.title}</b>")
        self.article_page.meta.setText(
            f"{article.word_count} words · source: {article.source.value}"
            + (f" · {article.source_url}" if article.source_url else "")
        )
        body = "<br><br>".join(
            f"<b>{label}.</b> {para}"
            for label, para in zip(article.paragraph_labels, article.paragraphs, strict=True)
        )
        self.article_page.body.setHtml(f"<h2>{article.title}</h2>" + body)
        self.go(self.article_page)

    def _on_failed(self, message: str) -> None:
        QMessageBox.warning(self, "hyrip", message)
        self.go(self.input_page)

    def _start_answering(self) -> None:
        def job():
            self.service.start(self.session_id)
            return None

        run_on_thread(job, lambda _: self.go(self.questions_page), self._on_failed)

    def _go_next(self) -> None:
        raw, skipped = self.questions_page.current_answer()
        if skipped:
            QMessageBox.information(self, "hyrip", "Answer first, or press Skip.")
            return
        if raw:
            self.questions_page._drafts[
                self.questions_page.view["questions"][self.questions_page.idx]["number"]
            ] = raw
        question = self.questions_page.view["questions"][self.questions_page.idx]
        answer = UserAnswer(number=question["number"], raw=raw, skipped=False)
        self.questions_page.controller.record(self.session_id, answer, raw)
        if self.questions_page.idx + 1 >= len(self.questions_page.view["questions"]):
            self._maybe_finish_hint()
            return

        self.questions_page.idx += 1
        self.questions_page.render()

    def _go_prev(self) -> None:
        self.questions_page.idx = self.questions_page.prev_index()
        self.questions_page.render()

    def _skip(self) -> None:
        question = self.questions_page.view["questions"][self.questions_page.idx]
        self.controller.record(
            self.session_id, UserAnswer(number=question["number"], skipped=True), ""
        )
        if self.questions_page.idx + 1 >= len(self.questions_page.view["questions"]):
            self._maybe_finish_hint()
        else:
            self.questions_page.idx += 1
            self.questions_page.render()

    def _maybe_finish_hint(self) -> None:
        QMessageBox.information(self, "hyrip", "That was the last question — press Finish.")

    def _on_verdict(self, number: int, graded, raw: str) -> None:
        if graded is None:
            return
        page = self.questions_page
        glyph = VERDICT_GLYPH.get(graded.verdict, "·")
        page.set_tutor(f"<b>{glyph} {graded.verdict}</b> (marked by {graded.graded_by})")
        page.controller.tutor_feedback(self.session_id, number, graded)

    def _on_tutor_chunk(self, number: int, token: str) -> None:
        self.questions_page.append_tutor(token)

    def _on_tutor_done(self, number: int, text: str) -> None:
        pass  # the streamed text is already on screen

    def _finish(self) -> None:
        reply = QMessageBox.question(self, "hyrip", "Finish and mark the paper?")
        if reply == QMessageBox.Yes:
            self.controller.finish(self.session_id)

    def _on_report(self, report: SessionReport) -> None:
        mode = self.questions_page.snapshot.mode if self.questions_page.snapshot else "exam"
        self.report_page.show_report(report, mode=mode)
        self.history_page.load()
        self.go(self.report_page)
