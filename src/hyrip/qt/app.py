"""Qt app bootstrap: `hyrip gui`."""


def run_gui(service) -> None:
    """Run the desktop interface against a ready SessionService (blocks until quit)."""
    import sys

    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    sys_argv = sys.argv
    app = QApplication(sys_argv)
    app.setApplicationName("hyrip")
    app.setOrganizationName("hyrip")
    window = MainWindow(service)
    window.show()
    raise SystemExit(app.exec())
