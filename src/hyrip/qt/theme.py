"""Monochrome theming for the Qt interface.

The same two grey palettes the web UI uses — light + dark, nothing beyond
grayscale. Verdicts render by glyph and typography (✓ ≈ ✗ ·), not colour, so
either palette adapts itself and switching theme never desynchronises content.

A preference (system / light / dark) is persisted in QSettings; "system" follows
the desktop colour scheme live via styleHints().colorSchemeChanged.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

SYSTEM = "system"
LIGHT = "light"
DARK = "dark"
CHOICES = (SYSTEM, LIGHT, DARK)

# Same tokens as web/static/style.css — keep the two files in step.
TOKENS = {
    LIGHT: {
        "ink": "#171717",  # foreground / fills
        "muted": "#6b6b6b",  # secondary text
        "faint": "#a3a3a3",  # tertiary text, disabled
        "bg": "#fafafa",
        "card": "#ffffff",
        "line": "#d6d6d6",
        "soft": "#f0f0f0",  # tinted surfaces: table head, hover
    },
    DARK: {
        "ink": "#ededed",
        "muted": "#9c9c9c",
        "faint": "#6b6b6b",
        "bg": "#121212",
        "card": "#1c1c1c",
        "line": "#363636",
        "soft": "#262626",
    },
}


def stylesheet(t: dict[str, str]) -> str:
    """One QSS string for a token set; no hue appears anywhere."""
    line, soft, card, ink = t["line"], t["soft"], t["card"], t["ink"]
    return f"""
    QMainWindow, QWidget {{ background: {t["bg"]}; color: {t["ink"]}; }}
    QLabel {{ color: {t["ink"]}; }}

    QGroupBox {{
        border: 1px solid {line}; border-radius: 8px; margin-top: 10px; padding-top: 6px;
    }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; color: {t["muted"]}; }}

    QLineEdit, QPlainTextEdit, QComboBox, QSpinBox {{
        background: {card}; color: {t["ink"]}; border: 1px solid {line};
        border-radius: 6px; padding: 4px 6px;
    }}
    QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus {{
        border-color: {ink};
    }}
    QComboBox QAbstractItemView {{
        background: {card}; color: {t["ink"]};
        selection-background-color: {soft}; selection-color: {t["ink"]};
    }}

    QPushButton {{
        background: {card}; color: {t["ink"]}; border: 1px solid {line};
        border-radius: 6px; padding: 6px 14px;
    }}
    QPushButton:hover:enabled {{ border-color: {ink}; }}
    QPushButton:pressed:enabled {{ background: {soft}; }}
    QPushButton:disabled {{ color: {t["faint"]}; border-color: {line}; }}

    QTextBrowser {{ background: {card}; color: {t["ink"]}; border: 1px solid {line}; }}

    QTableWidget {{
        background: {card}; color: {t["ink"]}; border: 1px solid {line};
        gridline-color: {line};
        selection-background-color: {soft}; selection-color: {t["ink"]};
    }}
    QHeaderView::section {{
        background: {soft}; color: {t["ink"]}; border: none;
        border-right: 1px solid {line}; border-bottom: 1px solid {line}; padding: 4px 6px;
    }}

    QMenu {{ background: {card}; color: {t["ink"]}; border: 1px solid {line}; }}
    QMenu::item:selected {{ background: {soft}; }}
    QMenuBar {{ background: {t["bg"]}; color: {t["ink"]}; }}
    QMenuBar::item:selected {{ background: {soft}; }}

    * {{ selection-background-color: {ink}; selection-color: {t["bg"]}; }}
    """


def _palette(t: dict[str, str]) -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(t["bg"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(t["ink"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(t["card"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(t["soft"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(t["ink"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(t["card"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(t["ink"]))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(t["card"]))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(t["ink"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(t["soft"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(t["ink"]))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(t["faint"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(t["faint"]))
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(t["faint"])
    )
    return palette


def stored_preference() -> str:
    """The persisted preference, or "system" when nothing (valid) is stored."""
    from PySide6.QtCore import QSettings

    value = QSettings("hyrip", "hyrip").value("theme", SYSTEM)
    return value if value in CHOICES else SYSTEM


def set_preference(pref: str) -> None:
    from PySide6.QtCore import QSettings

    QSettings("hyrip", "hyrip").setValue("theme", pref)


def resolve(pref: str, app: QApplication) -> str:
    """Preference → effective palette key; "system" reads the desktop scheme."""
    if pref in (LIGHT, DARK):
        return pref
    scheme = app.styleHints().colorScheme()
    return DARK if scheme == Qt.ColorScheme.Dark else LIGHT


def apply(pref: str, app: QApplication) -> str:
    """Install Fusion + the monochrome palette; returns the effective key."""
    tokens = TOKENS[resolve(pref, app)]
    app.setStyle("Fusion")
    app.setPalette(_palette(tokens))
    app.setStyleSheet(stylesheet(tokens))
    return resolve(pref, app)


def install(app: QApplication) -> str:
    """Apply the stored preference now and follow the desktop scheme live."""
    pref = stored_preference()
    effective = apply(pref, app)
    hints = app.styleHints()

    def _follow(_scheme) -> None:
        # re-read, so a later menu choice stops the follow even with a stale signal
        if stored_preference() == SYSTEM:
            apply(SYSTEM, app)

    hints.colorSchemeChanged.connect(_follow)
    return effective
